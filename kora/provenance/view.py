"""What an agent is allowed to see."""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field

from kora.provenance.models import (
    Constraint,
    EntityRef,
    EvidenceRecord,
    QueryAttempt,
    RequirementState,
    TaskStatus,
    labels_for,
)
from kora.tools.dialect import parse_query

ROOT = "T0"


SHOWN = 5


def set_name(role: str) -> str:
    """How a bound role is named in a query: `@` and the role, made safe."""
    return "@" + re.sub(r"[^A-Za-z0-9_.]", "_", role)


class TaskView(BaseModel):
    """One task, and only what writing a query for it requires."""

    model_config = ConfigDict(extra="forbid")

    goal: str
    task_id: str | None = None

    question: str = ""
    projection: str = "identity"
    expected_entity_type: str = ""
    final: bool = False
    hint: str = ""

    anchors: list[EntityRef] = Field(default_factory=list)
    sets: dict[str, list[str]] = Field(default_factory=dict)

    claims: list[str] = Field(default_factory=list)
    requirements: list[str] = Field(default_factory=list)
    variable: str = ""
    expected_count: str = "many"
    held: list[str] = Field(default_factory=list)
    chain: list[dict] = Field(default_factory=list)
    tried: list[tuple[str, str, str]] = Field(default_factory=list)
    closing: bool = False
    page_limit: int = 0

    @property
    def anchor_ids(self) -> list[str]:
        return list(dict.fromkeys(entity.id for entity in self.anchors))

    @property
    def labels(self) -> dict[str, str]:
        return {e.id: e.label for e in self.anchors if e.label}


class SufficiencyView(BaseModel):
    """What sufficiency gets to see of one target: the root question or one task."""

    model_config = ConfigDict(extra="forbid")

    target_id: str
    planned: bool = False
    goal: str
    question: str = ""
    expected_output_type: str = ""
    expected_entity_type: str = ""
    accepted_entity_types: list[str] = Field(default_factory=list)
    requirements: list[RequirementState] = Field(default_factory=list)
    satisfies: list[str] = Field(default_factory=list)
    child_outputs: dict[str, list[EntityRef]] = Field(default_factory=dict)
    anchors: list[EntityRef] = Field(default_factory=list)
    anchor_labels: dict[str, str] = Field(default_factory=dict)
    bindings: dict[str, list[EntityRef]] = Field(default_factory=dict)
    current_output: list[EntityRef] = Field(default_factory=list)
    consumers: list[str] = Field(default_factory=list)
    claims: list[str] = Field(default_factory=list)
    path: list[str] = Field(default_factory=list)
    wanted_count: str = "many"


def fact(record: EvidenceRecord, labels: dict[str, str | None]) -> str:
    """One evidence record as a line a model can read."""
    subject = labels.get(record.subject_id) or record.subject_id
    obj = record.object_value or labels.get(record.object_id) or record.object_id or ""
    return f"{subject} {record.relation} {obj}".strip()


def under(state, task_id: str) -> set[str]:
    """The tasks whose evidence belongs to this one: itself, what it waits on, and everything beneath it."""
    task = state.tasks.get(task_id)
    if task is None:
        return {task_id}
    allowed = {task_id} | set(task.dependencies)
    frontier = list(task.children)
    while frontier:
        child = state.tasks.get(frontier.pop())
        if child is not None:
            allowed.add(child.id)
            frontier.extend(child.children)
    return allowed


def task_view(state, task_id: str | None = None) -> TaskView:
    """The Query Agent's view: one task, and what writing a query for it needs."""
    common = {"question": state.question.text,
              "closing": bool(state.resources.soft_seconds
                              and state.resources.wall_seconds() >= state.resources.soft_seconds),
              "projection": state.plan.answer.projection if state.plan else "identity"}

    if task_id is None:
        mine, _ = _constraints_for(state)
        return TaskView(
            goal=state.question.text,
            anchors=list(state.question.root_entities),
            expected_entity_type=(state.contract.answer_entity_type
                                  if state.contract else ""),
            claims=unfilled_claims(mine, state.question.text),
            requirements=requirement_sentences(state, None, state.question.text),
            variable=state.plan.answer.variable if state.plan else "",
            expected_count=state.contract.answer_count if state.contract else "many",
            **common,
        )

    task = state.task(task_id)
    contract = task.contract
    if state.regime == "fold" and task_id != ROOT:
        common["question"] = (contract.goal if contract and contract.goal else task.goal)
    task_attempts = (list(state.task_states[task_id].attempts)
                     if task_id in state.task_states else [])
    goal = contract.goal if contract and contract.goal else task.goal
    mine, asked = _constraints_for(state, task)
    variable = produced_variable(state, task, asked)
    claims = unfilled_claims(mine, goal)
    if task_id != ROOT and not variable:
        claims = [f"c1: {goal}"]
    return TaskView(
        task_id=task_id,
        goal=goal,
        anchors=list(task.grounded_inputs),
        final=state.is_final(task_id),
        hint=task.diagnosis,
        expected_entity_type=((contract.expected_entity_type if contract else None)
                              or task.expected_output or ""),
        sets={set_name(role): [e.id for e in refs]
              for role, refs in task.bindings.roles.items() if len(refs) >= 2},
        claims=claims,
        requirements=requirement_sentences(state, set(task.requirement_ids), goal),
        variable=variable,
        expected_count=((state.contract.answer_count if state.contract else "many")
                        if task_id == ROOT else "many"),
        held=held_of(task),
        chain=chain_of(state, task),
        tried=tried_of(task_attempts, {e.id for e in task.grounded_inputs}),
        **common,
    )


def sufficiency_view(state, target_id: str | None = None) -> SufficiencyView:
    """Sufficiency's view of one target: what it stands behind, and on what."""
    target_id = target_id or ROOT
    if target_id in ("ROOT", "root"):
        target_id = ROOT
    task = state.tasks.get(target_id)
    contract_requirements = list(state.contract.requirements) if state.contract else []
    wanted_count = ((state.contract.answer_count if state.contract else "many")
                    if target_id == ROOT else "many")

    if task is None:
        # No task graph yet: the question itself is the target.
        return SufficiencyView(
            target_id=target_id, goal=state.question.text, question=state.question.text,
            expected_entity_type=(state.contract.answer_entity_type
                                  if state.contract else "") or "",
            accepted_entity_types=(list(state.contract.accepted_entity_types)
                                   if state.contract else []),
            requirements=contract_requirements,
            anchors=list(state.question.root_entities),
            anchor_labels=labels_for(None, state.question.root_entities),
            current_output=[],
            wanted_count=wanted_count,
        )

    wanted = set(task.requirement_ids)
    contract = task.contract

    requirements = (contract_requirements if target_id == ROOT
                    else [r for r in contract_requirements if r.id in wanted])
    expected_entity_type = (contract.expected_entity_type if contract else None) or ""
    bindings = {role: list(refs) for role, refs in task.bindings.roles.items()}
    contract_only = state.regime == "fold"
    output = (state.projected(state.produced_by(target_id))
              if target_id == ROOT else state.produced_by(target_id))
    return SufficiencyView(
        target_id=target_id,
        planned=state.plan is not None,
        goal=task.goal,
        # Under fold a child is judged against ITS question, not the root's.
        question=(task.goal if contract_only and target_id != ROOT else state.question.text),
        expected_output_type=((contract.expected_output_type if contract else None)
                              or task.expected_output or ""),
        expected_entity_type=expected_entity_type,
        requirements=requirements,
        accepted_entity_types=(list(contract.accepted_entity_types) if contract else []),
        satisfies=[] if target_id == ROOT else list(task.satisfies),
        child_outputs=({} if contract_only
                       else {c: state.produced_by(c) for c in task.latest_children}),
        anchors=list(task.grounded_inputs),
        anchor_labels=labels_for(task, state.question.root_entities
                                 if target_id == ROOT else ()),
        bindings=bindings,
        current_output=output,
        consumers=([] if contract_only else
                   sorted(t.id for t in state.tasks.values()
                          if target_id in t.dependencies
                          and t.status is not TaskStatus.FAILED)),
        claims=claims_of(state, task, output),
        path=path_of(state, task),
        wanted_count=wanted_count,
    )


def _constraints_for(state, task=None) -> tuple[list[Constraint], set[str]]:
    constraints = list(state.contract.constraints) if state.contract else []
    asked = set(task.found.get("for") or []) if task is not None and task.id != ROOT else set()
    mine = [c for c in constraints if c.id in asked] or constraints
    return mine, asked


def claims_of(state, task, output: list[EntityRef]) -> list[str]:
    mine, asked = _constraints_for(state, task)
    variable = produced_variable(state, task, asked)
    filled = members([e.label or e.id for e in output])
    return [f"{c.id}: " + (c.says.replace("{" + variable + "}", "{" + f"{variable} = {filled}" + "}")
                           if variable else c.says) for c in mine]


def produced_variable(state, task, asked: set[str]) -> str:
    if task.id == ROOT:
        return state.plan.answer.variable if state.plan else ""
    needs = state.plan.recipe.needs if state.plan else []
    covering = [n.variable for n in needs if asked and asked <= set(n.from_constraints)]
    return covering[-1] if covering else ""


def members(names: list[str]) -> str:
    return ", ".join(names[:SHOWN]) + (f" … and {len(names) - SHOWN} more" if len(names) > SHOWN else "")


def requirement_sentences(state, wanted: set[str] | None, goal: str) -> list[str]:
    requirements = state.contract.requirements if state.contract else []
    found = [r.description for r in requirements if r.description and (wanted is None or r.id in wanted)]
    return found or [goal]


def unfilled_claims(mine: list[Constraint], goal: str) -> list[str]:
    return [f"{c.id}: {c.says}" for c in mine] if mine else [f"c1: {goal}"]


_UNRESOLVED_OUTCOMES = frozenset(
    {"EMPTY_RESULT", "EXECUTION_ERROR", "INSUFFICIENT_RESULT", "NO_VALID_CANDIDATE"})


def tried_of(attempts: list[QueryAttempt], anchor_ids: set[str]) -> list[tuple[str, str, str]]:
    found: list[tuple[str, str, str]] = []
    for attempt in attempts:
        for candidate in attempt.candidates:
            if candidate.outcome not in _UNRESOLVED_OUTCOMES:
                continue
            parsed = parse_query(candidate.query)
            if not parsed:
                continue
            anchor = next((term for pattern in parsed[1] for term in pattern
                          if term in anchor_ids), None)
            if anchor is None:
                continue
            found.extend((anchor, relation, candidate.outcome) for relation in candidate.relations)
    return found


def chain_of(state, task, _seen: set | None = None) -> list[dict]:
    """The queries whose outputs this task stands on, in walking order, as {query, ids, n}."""
    seen = _seen if _seen is not None else set()
    if task.id in seen:
        return []
    seen.add(task.id)
    feeders = [state.tasks[t] for t in dict.fromkeys([*task.dependencies, *task.latest_children]) if t in state.tasks]
    ordered, placed = [], set()
    while feeders:
        ready = [f for f in feeders if all(d in placed or d not in {x.id for x in feeders} for d in f.dependencies)]
        if not ready:
            ready = feeders[:1]
        for f in ready:
            ordered.append(f); placed.add(f.id); feeders.remove(f)
    links: list[dict] = []
    for feeder in ordered:
        if "operate" in feeder.found:
            return []
        links += chain_of(state, feeder, seen)
        attempts = state.task_states[feeder.id].attempts if feeder.id in state.task_states else []
        query = next((a.executed_query for a in reversed(attempts) if a.executed_query and a.result_count), "")
        found = state.produced_by(feeder.id)
        if query and found:
            ids = list(dict.fromkeys(e.id for e in found))
            links.append({"query": query, "ids": ids[:30], "n": len(ids)})
    return links


def held_of(task) -> list[str]:
    return [f"{set_name(role)}: {len(refs)} entities, e.g. "
            f"{members([e.label or e.id for e in refs])}"
            for role, refs in task.bindings.roles.items() if len(refs) >= 2]


def path_of(state, task) -> list[str]:
    lines: list[str] = []
    _draw(state, task, lines)
    _descend(state, task, lines)
    return lines


def _draw(state, task, lines: list[str]) -> None:
    line = _step(state, task)
    if line:
        lines.append(line)


def _descend(state, task, lines: list[str]) -> None:
    for child_id in task.latest_children:
        child = state.tasks.get(child_id)
        if child is None:
            continue
        if child.status is TaskStatus.RESOLVED:
            _draw(state, child, lines)
        _descend(state, child, lines)


def _step(state, task) -> str:
    if "operate" in task.found:
        return _operate_step(state, task)
    if task.composed:
        return ""
    return _query_step(state, task)


def _operate_step(state, task) -> str:
    found = state.produced_by(task.id)
    if not found:
        return ""
    names = names_line(found)
    start = ", ".join(e.label or e.id for e in task.grounded_inputs[:3]) or task.id
    return f"{start} -[{task.goal}]-> {{{names}}} ({len(found)})"


def _query_step(state, task) -> str:
    attempts = state.task_states[task.id].attempts if task.id in state.task_states else []
    query = next((a.executed_query for a in reversed(attempts) if a.executed_query and a.result_count), "")
    if not query:
        return ""
    parsed = parse_query(query)
    relation = " / ".join(p[1] for p in parsed[1]) if parsed else ""
    if not relation:
        return ""
    found = state.produced_by(task.id)
    if not found:
        return ""
    start = ", ".join(e.label or e.id for e in task.grounded_inputs[:3]) or task.id
    return f"{start} -[{relation}]-> {{{names_line(found)}}} ({len(found)})"


def names_line(found: list[EntityRef]) -> str:
    names = [e.label or e.id for e in found[:3]]
    if len(found) > 3:
        return ", ".join(names) + f" … and {len(found) - 3} more"
    return ", ".join(names)


def planning_view(state, task_id: str | None = None):
    from kora.modules.planning.planner import PlanRequest

    return PlanRequest(
        question=state.question.text,
        grounded_entities=list(state.question.root_entities),
        anchor_labels=labels_for(state.tasks.get(ROOT), state.question.root_entities),
    )


def decomposition_view(state, task_id: str | None = None):
    """The Decomposer's view: the planning view plus the question's structure and what was tried."""
    from kora.modules.decomposition import view as decomposition

    return decomposition.of(state, task_id or ROOT)


BY_PURPOSE = {
    "query": task_view,
    "sufficiency": sufficiency_view,
    "planning": planning_view,
    "decompose": decomposition_view,
}
