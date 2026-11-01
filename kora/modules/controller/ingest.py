"""The question becomes the root task, standing on its entities and the names it states."""

from __future__ import annotations

import re

from kora.provenance.models import (
    EntityRef,
    Plan,
    TaskBindings,
    TaskContract,
    TaskNode,
    answer_entity_types,
)
from kora.provenance.state import KoraState
from kora.modules.controller.reconcile import a_place_to_stand
from kora.modules.planning.planner import _FILLER, PlannerError, to_graph

ROOT_ID = "T0"
_WORD = re.compile(r"[a-z0-9]+")


def root_task(state: KoraState, plan: Plan | None = None) -> TaskNode:
    entities = list(state.question.root_entities)
    contract = None
    if plan is not None:
        accepted = (list(state.contract.accepted_entity_types)
                    if state.contract is not None and state.contract.accepted_entity_types
                    else answer_entity_types(state.question.text, plan.answer.kind or None))
        contract = TaskContract(goal=state.question.text, expected_output_type=plan.answer.type,
                                expected_entity_type=accepted[0] if accepted else None,
                                accepted_entity_types=accepted)
    requirement_ids = [r.id for r in (state.contract.requirements if state.contract else [])]
    return TaskNode(id=ROOT_ID, goal=state.question.text, requirement_ids=requirement_ids,
                    satisfies=list(requirement_ids), grounded_inputs=entities,
                    bindings=TaskBindings(roles={(e.label or e.id): [e] for e in entities}),
                    output_name="answer", depth=0, contract=contract,
                    expected_output=plan.answer.type if plan else "")


def open_root(state: KoraState, planner, log: list[str]) -> None:
    state.regime = "fold"
    try:
        planner.run(state)
    except PlannerError as exc:
        log.append(f"plan root: rejected -- {str(exc)[:120]}")
    state.task_created(root_task(state, state.plan), rationale="the question as the root task")
    if state.plan is not None:
        grounded = {(e.label or e.id): e for e in state.question.root_entities}
        state.canonicalised(to_graph(state.plan, grounded), rationale="the plan, read as a graph")


def ground_stated_names(state: KoraState, names, resolver, kg, log: list[str]) -> None:
    if names is None or resolver is None:
        return
    before = names.spend.model_copy()
    stated = names(state)
    state.budget_consumed(tokens=names.spend.tokens - before.tokens, seconds=names.spend.seconds - before.seconds,
                          llm_calls=names.spend.llm_calls - before.llm_calls)
    held = {e.id for e in state.question.root_entities}
    labels = {e.label or "" for e in state.question.root_entities}
    added: list[EntityRef] = []
    for name in stated:
        if any(same_name(name, x) for x in labels if x):
            continue
        if not says(state.question.text, name):
            continue
        try:
            matches = resolver.resolve(name)
        except Exception:
            continue
        if not matches:
            continue
        ref = EntityRef(id=matches[0].id, label=getattr(matches[0], "name", None) or name)
        if ref.id in held or not a_place_to_stand(ref, kg):
            continue
        held.add(ref.id)
        labels.add(ref.label or "")
        state.question.root_entities.append(ref)
        state.entity_grounded(ref, rationale="stated in the question")
        added.append(ref)
    if added:
        log.append("grounded stated names: " + ", ".join(e.label or e.id for e in added))


def says(question: str, name: str) -> bool:
    """Whether the question itself says this name: every word of it, or its initials ("jfk")."""
    said = set(_WORD.findall(question.lower()))
    words = [w for w in _WORD.findall(name.lower()) if w not in _FILLER]
    return bool(words) and (all(w in said for w in words) or "".join(w[0] for w in words) in said)


def same_name(a: str, b: str) -> bool:
    ta, tb = set(_WORD.findall(a.lower())), set(_WORD.findall(b.lower()))
    return bool(ta and tb) and (ta <= tb or tb <= ta)
