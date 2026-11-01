from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context

from kora.modules.query.models import Candidate, Context, Execution


def execute_all(kg, candidates: list[Candidate], ctx: Context) -> list[tuple[Candidate, Execution]]:
    if not candidates:
        return []
    with ThreadPoolExecutor(max_workers=len(candidates)) as pool:
        runs = [pool.submit(copy_context().run, Execution.run, kg, c, ctx.task.sets) for c in candidates]
        executions = [run.result() for run in runs]
    ctx.usage.kg_queries += len(executions)
    ctx.usage.seconds += max(e.latency_seconds for e in executions)
    return list(zip(candidates, executions, strict=True))
