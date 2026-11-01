"""What a query returned becomes grounded; what a task produced is bound into the tasks that wait on it."""

from __future__ import annotations

import re

from kora.provenance.models import EntityRef, TaskNode, TaskStatus
from kora.provenance.state import KoraState
from kora.modules.query.models import MAX_ANCHORS

LABELLED = 25
_VALUE = re.compile(r"^[+-]?\d[\d,.:/T\- ]*Z?$|\s|^\"")


def ground(state: KoraState, task: TaskNode, entity_ids: list[str], kg) -> list[EntityRef]:
    refs: list[EntityRef] = []
    for n, entity_id in enumerate(entity_ids):
        label = kg.label(entity_id) if n < LABELLED and hasattr(kg, "label") else None
        ref = EntityRef(id=entity_id, label=label)
        if label is None and n < LABELLED and looks_like_value(entity_id) \
                and hasattr(kg, "exists") and not kg.exists(entity_id):
            ref = EntityRef(id=entity_id, label=entity_id, entity_type="value")
        state.entity_grounded(ref, task_id=task.id, rationale="query result")
        refs.append(ref)
    return refs


def propagate(state: KoraState, producer: TaskNode, kg=None) -> list[str]:
    output = state.produced_by(producer.id)
    standable = [e for e in output if a_place_to_stand(e, None)]
    bound: list[str] = []
    for task in state.tasks.values():
        if producer.id not in task.dependencies or task.status in (
            TaskStatus.RESOLVED, TaskStatus.FAILED, TaskStatus.DECOMPOSED
        ):
            continue
        held = {e.id for e in task.grounded_inputs}
        room = max(0, MAX_ANCHORS - len(held))
        anchors = [e for e in standable if e.id not in held][:room]
        state.task_inputs_bound(task.id, producer.output_name or producer.id, output,
                                anchors=anchors, rationale=f"output of {producer.id}")
        bound.append(task.id)
    return bound


def looks_like_value(text: str) -> bool:
    return bool(_VALUE.search(text or ""))


def a_place_to_stand(entity: EntityRef, kg) -> bool:
    if entity.entity_type == "value":
        return False
    if kg is None or not hasattr(kg, "is_hub"):
        return True
    try:
        return not kg.is_hub(entity.id)
    except Exception:
        return False
