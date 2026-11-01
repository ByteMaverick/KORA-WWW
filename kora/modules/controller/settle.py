"""A done task offers its set to the root, then walks up one edge."""

from __future__ import annotations

from kora.provenance.models import TaskNode, TaskStatus
from kora.provenance.state import KoraState
from kora.modules.controller import ingest, reconcile
from kora.modules.controller.moves import compose


def settle(ctl, state: KoraState, task: TaskNode, result) -> None:
    state.task_done(task.id, rationale="answered")
    reconcile.propagate(state, task, ctl.kg)
    root = state.task(ingest.ROOT_ID)
    if task.id != root.id and root.status is not TaskStatus.RESOLVED:
        offer(ctl, state, task, root, result)
    parent = state.tasks.get(task.parent_id or "")
    if parent is None or task.id not in parent.latest_children or parent.status is TaskStatus.RESOLVED:
        return
    if root.status is TaskStatus.RESOLVED or not state.split_done(parent.id):
        return
    state.task_composed(parent.id, compose(state, parent))
    verdict = judge(ctl, state, parent, result)
    result.log.append(f"composed {parent.id}: {parent.combine} -> {len(parent.composed)}, "
                      f"judged {'yes' if verdict.sufficient else 'no'} {verdict.reason[:80]}")
    if verdict.sufficient:
        settle(ctl, state, parent, result)
    else:
        state.task_stuck(parent.id, diagnosis=f"judge: {verdict.reason[:200]}")


def offer(ctl, state: KoraState, task: TaskNode, root: TaskNode, result) -> None:
    found = state.produced_by(task.id)
    if not found:
        return
    composed = list(root.composed)
    owned = {i for i, grounded in state.grounded_entities.items() if grounded.produced_by_task(root.id)}
    state.task_composed(root.id, found, rationale=f"offered by {task.id}")
    verdict = judge(ctl, state, root, result)
    result.log.append(f"offer {task.id}: {len(found)} -> root, judged "
                      f"{'yes' if verdict.sufficient else 'no'} -- {verdict.reason[:80]}")
    if verdict.sufficient:
        state.task_done(root.id, rationale=f"answered by {task.id}")
        return
    state.task_output_withdrawn(root.id, [e.id for e in found if e.id not in owned],
                                rationale=f"offer from {task.id} judged no")
    state.task_composed(root.id, composed, rationale=f"what the root held before the offer from {task.id}")


def judge(ctl, state: KoraState, task: TaskNode, result):
    verdict = ctl.judge.check(state, task.id)
    state.judged(task.id, verdict, rationale=verdict.basis or "judge")
    absorb(state, task, verdict, result)
    if verdict.spend:
        state.budget_consumed(tokens=verdict.spend.tokens, seconds=verdict.spend.seconds,
                              llm_calls=verdict.spend.llm_calls)
    return verdict


def absorb(state: KoraState, task: TaskNode, verdict, result) -> None:
    known = {r.id for r in (state.contract.requirements if state.contract else [])}
    claimed = [r for r in task.satisfies if r in known]
    if not claimed:
        return
    if not verdict.sufficient:
        for requirement_id in claimed:
            state.requirement_blocked(requirement_id, task.id, diagnosis=verdict.reason, rationale=verdict.basis)
        result.log.append(f"blocked {', '.join(claimed)} on {task.id}: {verdict.reason[:60]} (still open)")
        return
    settled = set(verdict.satisfied_requirements) or set(claimed)
    closed = [r for r in claimed if r in settled]
    for requirement_id in closed:
        state.requirement_resolved(requirement_id, rationale=verdict.reason)
    if closed:
        result.log.append(f"closed {', '.join(closed)} on {task.id}; coverage {state.coverage():.0%}")
