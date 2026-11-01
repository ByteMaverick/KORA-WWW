"""The controller, rebuilt: an AND-tree, one loop, only the clock stops it."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Protocol

from kora.provenance.models import EntityRef, TaskNode, TaskStatus
from kora.provenance.state import KoraState
from kora.modules.controller import clock, dispatch, ingest, moves
from kora.modules.controller.models import Budget, ControlResult

__all__ = ["Controller", "Strategy"]


class Strategy(Protocol):
    def next(self, state: KoraState, task: TaskNode) -> moves.Moves: ...


class Controller:
    def __init__(self, agent, planner, judge, strategy: Strategy, *, kg,
                 budget: Budget | None = None, parallel: int = 4, names=None, resolver=None,
                 chooser=None, fallback: str = "held"):
        self.agent, self.planner, self.judge, self.strategy, self.kg = agent, planner, judge, strategy, kg
        self.budget, self.parallel, self.names, self.resolver = budget, max(1, parallel), names, resolver
        self.chooser = chooser
        self.fallback = fallback

    def run(self, state: KoraState) -> ControlResult:
        result = ControlResult()
        from kora.modules.sufficiency import estimator as _estimator
        _estimator.reset_best()
        if self.budget is not None:
            state.resources.token_budget = self.budget.tokens
            state.resources.time_budget_seconds = self.budget.seconds
            state.resources.soft_seconds = self.budget.soft_seconds
        if ingest.ROOT_ID not in state.tasks:
            ingest.ground_stated_names(state, self.names, self.resolver, self.kg, result.log)
            ingest.open_root(state, self.planner, result.log)
        root = state.task(ingest.ROOT_ID)
        with ThreadPoolExecutor(max_workers=self.parallel) as pool:
            while clock.phase(state) != "closed" and root.status is not TaskStatus.RESOLVED:
                wave = state.ready()
                result.waves.append(wave)
                result.rounds += 1
                dispatch.run_wave(self, pool, state, wave, result)
                if clock.phase(state) == "closed" or root.status is TaskStatus.RESOLVED:
                    break
                grown = [t for t in state.stuck() if moves.grow(self, state, state.task(t), result)]
                if not wave and not grown:
                    return self.finish(state, result, "idle", "no task is ready and the strategy has no moves")
        if root.status is TaskStatus.RESOLVED:
            return self.finish(state, result, "solved", "the root is done")
        return self.finish(state, result, "timed_out", "the wall clock ended the run")

    @staticmethod
    def best_rejected_offer(state: KoraState):
        rejected = [v for v in state.verdicts.get(ingest.ROOT_ID, []) if not v.sufficient and v.answer_entities]
        return max(rejected, key=lambda v: v.confidence) if rejected else None

    def finish(self, state: KoraState, result: ControlResult, status: str, reason: str) -> ControlResult:
        root = state.task(ingest.ROOT_ID)
        partial = moves.compose(state, root) if status != "solved" else []
        if partial:
            state.task_composed(root.id, partial, rationale=f"partial at {status}")
        if status != "solved" and self.fallback == "best":
            best = self.best_rejected_offer(state)
            from kora.modules.sufficiency import estimator as _estimator
            seen = _estimator.best(root.id) if not partial else None   # what the root holds comes first
            if seen is not None and (best is None or seen[0] > best.confidence):
                p, ids, names = seen
                entities = [EntityRef(id=i, label=n or None) for i, n in zip(ids, names)]
                state.task_composed(root.id, entities, rationale=f"best candidate the judge saw at {status} (p={p:.2f})")
                result.log.append(f"finish: answering with the best candidate the judge saw (p={p:.2f}, "
                                  f"{len(entities)} entities) instead of what the root held")
            elif best is not None:
                state.task_composed(root.id, list(best.answer_entities),
                                    rationale=f"best rejected offer at {status} (p={best.confidence:.2f})")
                result.log.append(f"finish: answering with the best rejected offer (p={best.confidence:.2f}, "
                                  f"{len(best.answer_entities)} entities) instead of what the root held")
        result.answer_entities = state.produced_by(root.id)
        still_open = [t.id for t in state.tasks.values() if t.status is not TaskStatus.RESOLVED]
        result.status, result.phase_at_finish = status, clock.phase(state)
        result.stop_reason = reason + (f"; open: {', '.join(still_open)}" if still_open else "")
        result.log.append(f"finish: {status} -- {result.stop_reason}")
        return result
