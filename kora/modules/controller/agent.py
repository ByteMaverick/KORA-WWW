"""One task to the query agent, and what it did written down."""

from __future__ import annotations

from kora.provenance.models import EntityRef, TaskNode
from kora.provenance.state import KoraState
from kora.modules.controller import reconcile
from kora.modules.controller.ingest import ROOT_ID
from kora.modules.query.models import Outcome, QueryResult

__all__ = ["TaskAgent"]


class TaskAgent:
    def __init__(self, agent, kg):
        self.agent, self.kg = agent, kg

    def solve(self, state: KoraState, task_id: str):
        view = None
        try:
            view = state.get_view(task_id)
            return view, self.agent.solve(view)
        except Exception as exc:
            return view, QueryResult(status="FAILED", failure_type=Outcome.EXECUTION_ERROR,
                                     diagnosis=f"{type(exc).__name__}: {str(exc)[:200]}")

    def record(self, state: KoraState, task_id: str, outcome) -> bool:
        view, result = outcome
        task = state.task(task_id)
        if view is not None:
            self.agent.record(state, view, result, decide=False)
        reconcile.ground(state, task, result.entities, self.kg)
        entities = state.produced_by(task_id) or _returned_by_attempts(self.kg, state, task, result)
        counts_as = "superset" if task_id == ROOT_ID or not result.ok else (task.counts_as or "answer")
        kind = (task.contract.expected_entity_type if task.contract else None) or _kind_of(entities)
        state.set_produced(task_id, entities, kind=kind, produced_by=task_id, counts_as=counts_as,
                           rationale=f"what {task_id} returned")
        if not result.ok:
            task.diagnosis = result.diagnosis or ""
        return bool(result.ok)


def _returned_by_attempts(kg, state: KoraState, task: TaskNode, outcome) -> list[EntityRef]:
    toward = [e for a in outcome.attempts if a.on_the_way for e in a.entities]
    returned = list(dict.fromkeys(toward or outcome.entities
                                  or [e for a in outcome.attempts for e in a.entities]))[:50]
    found: list[EntityRef] = []
    for n, entity_id in enumerate(returned):
        known = state.grounded_entities.get(entity_id)
        if known is not None:
            found.append(EntityRef(id=known.id, label=known.label, entity_type=known.entity_type))
            continue
        ref = EntityRef(id=entity_id, label=_label(kg, entity_id) if n < reconcile.LABELLED else None)
        state.entity_grounded(ref, task_id=task.id, rationale=f"returned by an attempt of {task.id}")
        found.append(ref)
    return found


def _label(kg, entity_id: str) -> str | None:
    try:
        return kg.label(entity_id)
    except Exception:
        return None


def _kind_of(entities: list[EntityRef]) -> str:
    kinds: dict[str, int] = {}
    for entity in entities:
        if entity.entity_type:
            kinds[entity.entity_type] = kinds.get(entity.entity_type, 0) + 1
    return max(kinds.items(), key=lambda pair: pair[1])[0] if kinds else ""
