"""A wave of ready tasks: asks to the query agent in parallel, operates run in code; each result settles or sticks."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import partial

from kora.provenance.models import TaskNode, TaskStatus
from kora.provenance.state import KoraState
from kora.modules.controller.ingest import ROOT_ID
from kora.modules.controller.models import ControlResult
from kora.modules.controller.moves import operate_spec
from kora.modules.controller.settle import settle
from kora.modules.reduction.models import Operate
from kora.modules.reduction.reduction import lookup_on, operate


def run_wave(ctl, pool: ThreadPoolExecutor, state: KoraState, wave: list[str],
             result: ControlResult) -> None:
    operates = [t for t in wave if operate_spec(state.task(t)) is not None]
    for task_id in operates:
        if not answered(state):
            state.task_started(task_id, rationale="dispatched")
            run_operate(ctl, state, state.task(task_id), operate_spec(state.task(task_id)), result)
    asks = [] if answered(state) else [t for t in wave if t not in operates]
    for task_id in asks:
        state.task_started(task_id, rationale="dispatched")
    result.dispatches += len(asks)
    outcomes = list(pool.map(lambda task_id: ctl.agent.solve(state, task_id), asks))
    for task_id, outcome in zip(asks, outcomes):
        if ctl.agent.record(state, task_id, outcome):
            settle(ctl, state, state.task(task_id), result)
        else:
            state.task_stuck(task_id, rationale="the agent did not answer")


def answered(state: KoraState) -> bool:
    return state.task(ROOT_ID).status is TaskStatus.RESOLVED


def run_operate(ctl, state: KoraState, task: TaskNode, spec: Operate,
                result: ControlResult) -> None:
    for name in [*spec.after, *spec.over]:
        if name not in state.tasks and name not in state.sets:
            reason = (f"no piece, earlier piece, task or held set named {name!r} "
                     f"(looked in this reply, earlier replies under {task.parent_id}, "
                     f"task ids, held sets)")
            return failed(state, task, "refused", reason, result)
    sets = {name: state.produced_by(name) if name in state.tasks else state.sets[name].entities
            for name in spec.over if name in state.tasks or name in state.sets}
    chooser, choose, before = getattr(ctl, "chooser", None), None, 0
    if chooser is not None:
        kind = spec.expects_kind or next((state.sets[n].kind for n in spec.over if n in state.sets), "")
        choose, before = partial(chooser, kind=kind, says=claim_of(state, task, spec)), len(chooser.choices)
    try:
        anchors = tuple(dict.fromkeys(e.id for name in spec.over if name in state.tasks
                                      for e in state.task(name).grounded_inputs))
        found = operate(spec, sets, lookup_on(ctl.kg, embedder_of(ctl.agent), anchors),
                        choose=choose)
    except (KeyError, ValueError) as exc:
        return failed(state, task, "refused", str(exc), result)
    finally:
        if chooser is not None:
            charged(state, task, chooser.choices[before:])
    if not found:
        return failed(state, task, "empty", f"operate {spec.op} over {', '.join(spec.over)} produced nothing", result)
    state.task_composed(task.id, found, rationale=f"{spec.op} over {', '.join(spec.over)}")
    state.set_produced(task.id, found, produced_by=task.id, counts_as=spec.counts_as)
    settle(ctl, state, task, result)


def claim_of(state: KoraState, task: TaskNode, spec: Operate) -> str:
    """The plan's words for the operate's restriction: its own constraint, or the one holding the projection's key."""
    plan = state.plan_of(task.parent_id) if task.parent_id else state.plan
    if plan is None:
        return ""
    key = plan.answer.key.strip() if spec.op == plan.answer.projection else ""
    for c in plan.constraints:
        if c.id == spec.id or c.id in spec.for_ids or (key.startswith("?") and key in c.args):
            return c.says
    return ""


def charged(state: KoraState, task: TaskNode, choices: list) -> None:
    """Each property choice's model call on the budget, and the choice kept for the row."""
    for choice in choices:
        if choice.called:
            state.budget_consumed(tokens=choice.spend.tokens, seconds=choice.spend.seconds,
                                  llm_calls=choice.spend.llm_calls, rationale=f"the property for {task.id}")
        state.metadata.setdefault("property", []).append(
            {"task": task.id, "called": choice.called, "fallback": choice.fallback, "path": choice.path})


def failed(state: KoraState, task: TaskNode, how: str, diagnosis: str, result: ControlResult) -> None:
    state.task_stuck(task.id, diagnosis=diagnosis)
    parent = state.task(task.parent_id)
    if task.id in parent.latest_children and parent.status is not TaskStatus.RESOLVED:
        state.task_stuck(parent.id, diagnosis=diagnosis, rationale=f"its operate {task.id} {how}")
    result.log.append(f"operate {task.id}: {how} -- {diagnosis}")


def embedder_of(agent):
    return getattr(agent, "embedder", None) or getattr(getattr(agent, "agent", None), "embedder", None)
