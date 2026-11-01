"""The structural decomposer behind one `next(state, task) -> Moves`."""

from __future__ import annotations

import re

from kora.provenance.models import EntityRef, TaskNode
from kora.provenance.state import KoraState
from kora.modules.controller import ingest
from kora.modules.controller.moves import Ask, Moves
from kora.modules.decomposition import view as projection
from kora.modules.decomposition.decomposer import DecompositionError, DecompositionRequest
from kora.modules.decomposition.structural.compile import compile_cut, to_moves
from kora.modules.decomposition.structural.decomposer import StructuralDecomposer
from kora.modules.decomposition.structural.graph import (
    SubQueryFailureContext,
    cut_fingerprint,
    failure_cut_fingerprints,
)
from kora.modules.decomposition.structural.hypertree import HypertreeError
from kora.modules.decomposition.structural.operations import PROJECTED, placed, relations

__all__ = ["StructuralStrategy", "check_arm", "strategy_for"]

_CUT = re.compile(r"(?:detachment_)?(cut_\?[A-Za-z_][A-Za-z0-9_]*)")


class StructuralStrategy:
    """Cuts a stuck task's plan."""

    def __init__(self, llm, kg):
        self.decomposer, self.kg = StructuralDecomposer(llm), kg

    def next(self, state: KoraState, task: TaskNode) -> Moves:
        return self.cut(state, task)

    def cut(self, state: KoraState, task: TaskNode) -> Moves:
        cuts = state.metadata.setdefault(f"cuts:{task.id}", [])
        failed = SubQueryFailureContext(subtask_id=task.id, failed_node_id=task.id, failure_type="FAILED",
                                        diagnosis=task.diagnosis, attempt_count=task.attempts,
                                        prior_cuts=list(cuts))
        request = DecompositionRequest(task_id=task.id)
        whole, plan = state.graph_of(task.id), state.plan_of(task.id)
        view = view_of(state, task.id, self.kg)
        claims = list(view.claims)
        graph, view = relations(plan, whole, view)
        try:
            found = self.decomposer.decompose(request, view=view, graph=graph, failure_context=failed)
        except (DecompositionError, HypertreeError) as exc:
            return _once(state, task, _alone(state, task) or Moves(rationale=str(exc)[:200]))
        state.budget_consumed(tokens=found.spend.tokens, seconds=found.spend.seconds, llm_calls=found.spend.llm_calls)
        if not found.applicable:
            return _once(state, task, _alone(state, task) or Moves(rationale=found.rationale))
        if fingerprint := _cut_fingerprint(found, graph, view, failed):
            cuts.append(fingerprint)
        outputs = {s.id: f"?{s.output.name}" for s in found.subtasks}
        moves = _inheriting(to_moves(found, state, task), found)
        return _once(state, task, placed(moves, plan, outputs, _constants(whole), claims))


def _once(state: KoraState, task: TaskNode, moves: Moves) -> Moves:
    opened = state.metadata.setdefault(f"opened:{task.id}", [])
    shape = [p.goal if isinstance(p, Ask) else f"{p.op}:{p.prop}" for p in moves.pieces]
    if shape and shape in opened:
        return Moves(rationale="the same cut was opened before and did not answer")
    if shape:
        opened.append(shape)
    return moves


def strategy_for(arm: str, llm, kg):
    return STRATEGIES[check_arm(arm)](llm, kg)


def check_arm(arm: str) -> str:
    if arm not in STRATEGIES:
        raise ValueError(f"unknown arm {arm!r} (KORA_DECOMPOSER): one of {', '.join(STRATEGIES)}")
    return arm


STRATEGIES = {"structural": StructuralStrategy}


def _alone(state: KoraState, task: TaskNode) -> Moves | None:
    plan = state.plan_of(task.id)
    if plan is None or task.id != ingest.ROOT_ID:
        return None
    asked = [c for c in plan.constraints if not c.op]
    if len(asked) > 1 or (len(asked) == len(plan.constraints) and plan.answer.projection not in PROJECTED):
        return None
    pieces, outputs = [], {}
    for c in asked:
        variables = [a for a in c.args if a.startswith("?")]
        pieces.append(Ask(id=c.id, goal=c.says.replace("{", "").replace("}", ""),
                          anchors=[e.id for e in task.grounded_inputs], why="the plan's one relation"))
        wanted = [plan.answer.variable, *(o.over for o in plan.constraints if o.op)]
        outputs[c.id] = next((v for v in wanted if v in variables), variables[0] if variables else "")
    moves = Moves(pieces=pieces, combine="last", rationale="one relation, then the operations over it")
    return placed(moves, plan, outputs, _constants(state.graph_of(task.id)))


def _constants(graph) -> dict[str, EntityRef]:
    return dict(getattr(graph, "grounded", None) or {})


def view_of(state: KoraState, task_id: str, kg):
    task = state.tasks.get(task_id)
    statistics: dict[str, float] = {}
    if task is not None and hasattr(kg, "is_hub"):
        for entity in task.grounded_inputs[:10]:
            try:
                statistics[f"hub:{entity.label or entity.id}"] = float(kg.is_hub(entity.id))
            except Exception:
                continue
    return projection.of(state, task_id, statistics=statistics)


def _cut_fingerprint(result, graph, view, failure_context: SubQueryFailureContext) -> str:
    fingerprint = _fingerprint_from_rationale(result.rationale)
    if fingerprint or graph is None:
        return fingerprint
    residual = graph.scoped_to(view.claims).residualise(
        domains={f"?{role}": list(refs) for role, refs in view.bindings.items() if refs},
        resolved=[*graph.resolved_requirements, *view.resolved],
    )
    requirements = {r.id: r for r in view.requirements}
    for cut in residual.factorisations(forbidden_cuts=failure_cut_fingerprints(failure_context)):
        compiled = compile_cut(cut, residual, task_id=result.task_id, requirements=requirements)
        if _decomposition_shape(compiled) == _decomposition_shape(result):
            return cut_fingerprint(cut)
    return ""


def _decomposition_shape(result) -> tuple:
    return (
        tuple((tuple(s.required_inputs), tuple(s.satisfies), s.output.name,
               s.output.type, s.output.entity_type) for s in result.subtasks),
        tuple((d.from_id, d.to_id) for d in result.dependencies),
        tuple(result.completion), result.combine, result.recomposition,
    )


def _fingerprint_from_rationale(rationale: str) -> str:
    match = _CUT.search(rationale or "")
    return match.group(1) if match else ""


def _inheriting(moves: Moves, found) -> Moves:
    carried = {s.id: list(s.satisfies) for s in found.subtasks}
    pieces = []
    for p in moves.pieces:
        if isinstance(p, Ask) and carried.get(p.id):
            p = p.model_copy(update={"inherits": carried[p.id]})
        pieces.append(p)
    return moves.model_copy(update={"pieces": pieces})
