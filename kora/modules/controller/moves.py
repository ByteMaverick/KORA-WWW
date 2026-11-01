"""A strategy's pieces become a stuck task's children; their answers become the parent's."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from kora.provenance.models import (
    EntityRef,
    TaskBindings,
    TaskContract,
    TaskNode,
    TaskStatus,
)
from kora.provenance.state import KoraState
from kora.modules.query.models import MAX_ANCHORS
from kora.modules.reduction.models import Operate

__all__ = ["Ask", "Moves", "compose", "grow", "open_pieces", "operate_spec"]


class Ask(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    goal: str
    anchors: list[str]
    after: list[str] = Field(default_factory=list)
    expects_kind: str = ""
    counts_as: Literal["answer", "superset"] = "answer"
    why: str = ""
    satisfies: list[str] = Field(default_factory=list)
    inherits: list[str] = Field(default_factory=list)


class Moves(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pieces: list[Ask | Operate] = Field(default_factory=list)
    combine: Literal["last", "union", "intersect"] = "last"
    rationale: str = ""


def open_pieces(state: KoraState, task: TaskNode, moves: Moves, *,
                log: list[str]) -> list[TaskNode]:
    child_of: dict[str, str] = {}
    children: list[TaskNode] = []
    for piece in moves.pieces:
        child_id = f"{task.id}.{len(task.children) + len(children) + 1}"
        if isinstance(piece, Ask):
            if _restates(state, task, piece, child_of):
                if piece.id:
                    child_of[piece.id] = task.id
                log.append(f"open {task.id}: piece {piece.id!r} restates its parent; skipped")
                continue
            child = _ask_child(state, task, child_id, piece, child_of)
            if child is None:
                log.append(f"open {task.id}: piece {piece.id!r} stood on nothing held; skipped")
                continue
        else:
            child = _operate_child(state, task, child_id, piece, child_of)
        if piece.id:
            child_of[piece.id] = child_id
        children.append(child)
    if children:
        state.task_split(task.id, children, combine=moves.combine,
                         rationale=moves.rationale or "pieces from the strategy")
    return children


def grow(ctl, state: KoraState, task: TaskNode, result) -> bool:
    if operate_spec(task) is not None:
        return False
    moves = ctl.strategy.next(state, task)
    made = open_pieces(state, task, moves, log=result.log)
    result.decompositions += bool(made)
    result.log.append(f"grow {task.id}: {len(made)} piece(s), combine {moves.combine}")
    return bool(made)


def _restates(state: KoraState, task: TaskNode, ask: Ask, child_of: dict[str, str]) -> bool:
    parent = state.tasks.get(task.parent_id or "")
    if parent is None or not _same_goal(task.goal, parent.goal):
        return False
    held = {i for i in ask.anchors if i in state.grounded_entities}
    waits_on_a_sibling = any(child_of.get(a) not in (None, task.id) for a in ask.after)
    return (_same_goal(ask.goal, task.goal) and not waits_on_a_sibling
            and held <= {e.id for e in task.grounded_inputs})


def _same_goal(a: str, b: str) -> bool:
    return " ".join(a.lower().split()) == " ".join(b.lower().split())


def _ask_child(state: KoraState, task: TaskNode, child_id: str, ask: Ask,
               child_of: dict[str, str]) -> TaskNode | None:
    after = [child_of[a] for a in ask.after if a in child_of and child_of[a] != task.id]
    held = [state.grounded_entities[i] for i in ask.anchors if i in state.grounded_entities]
    anchors = [EntityRef(id=e.id, label=e.label, entity_type=e.entity_type) for e in held]
    stated = {e.id for e in state.question.root_entities}
    if any(e.id not in stated for e in anchors):
        anchors = _across_the_set(state, anchors)
    if not anchors and not after:
        return None
    kind = _kind(ask.expects_kind)
    found: dict[str, str | list[str]] = {"piece": ask.id, "for": list(ask.satisfies)}
    if ask.inherits:
        found["inherits"] = list(ask.inherits)
    return TaskNode(
        id=child_id, goal=ask.goal, dependencies=after, grounded_inputs=anchors,
        bindings=TaskBindings(roles={(e.label or e.id): [e] for e in anchors}),
        output_name=child_id, counts_as=ask.counts_as, expected_output=kind,
        contract=TaskContract(goal=ask.goal, expected_entity_type=kind or None),
        found=found)


def _piece(state: KoraState, task: TaskNode, child_of: dict[str, str], name: str) -> str | None:
    if name in child_of:
        return child_of[name]
    matches = [c for c in task.children if state.tasks[c].found.get("piece") == name]
    if matches:
        return matches[-1]
    if name in state.tasks:
        return name
    if name in state.sets:
        return name
    return None


def _operate_child(state: KoraState, task: TaskNode, child_id: str, op: Operate,
                   child_of: dict[str, str]) -> TaskNode:
    named = [*getattr(op, "after", []), *op.over]
    resolved = {name: _piece(state, task, child_of, name) for name in named}
    over = [resolved[name] if resolved[name] is not None else name for name in op.over]
    after = [resolved[name] if resolved[name] is not None else name for name in op.after]
    deps = [resolved[n] for n in named
            if resolved[n] is not None and resolved[n] != n and resolved[n] != task.id]
    spec = op.model_copy(update={"over": over, "after": after})
    return TaskNode(
        id=child_id, goal=f"operate: {spec.op}({', '.join([*(e.label or e.id for e in spec.named), *spec.over])})",
        dependencies=list(dict.fromkeys(deps)),
        output_name=child_id, counts_as=spec.counts_as,
        found={"operate": spec.model_dump_json(), "piece": op.id})


def operate_spec(task: TaskNode) -> Operate | None:
    raw = task.found.get("operate")
    return Operate.model_validate_json(raw) if raw else None


def compose(state: KoraState, task: TaskNode) -> list[EntityRef]:
    sets = [state.produced_by(c) for c in task.latest_children
            if state.tasks[c].status is TaskStatus.RESOLVED]
    if not sets:
        return []
    if task.combine == "last":
        return list(sets[-1])
    if task.combine == "intersect":
        rest = [{e.id for e in s} for s in sets[1:]]
        return [e for e in sets[0] if all(e.id in ids for ids in rest)]
    union: dict[str, EntityRef] = {}
    for s in sets:
        for e in s:
            union.setdefault(e.id, e)
    return list(union.values())


def _across_the_set(state: KoraState, anchors: list[EntityRef]) -> list[EntityRef]:
    held = {e.id for e in anchors}
    for record in state.sets.values():
        members = [e for e in record.entities if not record.refused]
        if len(members) < 2 or not (held & {e.id for e in members}):
            continue
        room = MAX_ANCHORS - len(anchors)
        if room <= 0:
            break
        anchors = anchors + [e for e in members if e.id not in held][:room]
        held = {e.id for e in anchors}
    return anchors


def _kind(kind: str | None) -> str:
    kind = (kind or "").strip()
    return "" if kind.lower() in ("?", "unknown", "none", "n/a", "any") else kind
