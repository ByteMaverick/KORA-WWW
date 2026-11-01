"""Cost and latency per question, from the rows the harness wrote (README: efficiency)."""
from __future__ import annotations

import numpy as np

from harness.metrics.groundedness import hook
from harness.metrics.groundedness.grounding import last_attempt, log_path
from harness.metrics.reasoning import miss_kind
from harness.settings import TIMEOUT_SECONDS


def question_cost(system: str, row: dict, source: str, shard, cell_dir) -> dict:
    fw = row.get("fw_metrics") or {}
    wall = float(row.get("wall_time_s") or 0.0)
    kind = miss_kind(row)
    if kind in ("over budget", "timeout"):
        wall = float(TIMEOUT_SECONDS)
    graph = None
    if cell_dir:
        lines, _ = hook.read_lines(log_path(cell_dir, shard, str(row.get("qid"))))
        graph = len(last_attempt(lines, row.get("wall_time_s"))[0])
    return {"calls": float(fw.get("num_llm_calls") or 0), "input": float(fw.get("total_prompt_tokens") or 0),
            "output": float(fw.get("total_completion_tokens") or 0), "latency": wall, "graph": graph}


def cell_costs(system: str, cell) -> list:
    """The cost of every question of a cell that has a row."""
    return [question_cost(system, row, source, shard, d) for row, source, shard, d in cell.rows.values()
            if row is not None]


def pool(costs: list, n_questions: int) -> dict:
    """Means, latency median and p90, graph calls (None when any row has no log), over pooled question costs."""
    if not costs:
        return {"rows": 0, "questions": n_questions}
    lat = [c["latency"] for c in costs]
    graph = [c["graph"] for c in costs]
    return {"rows": len(costs), "questions": n_questions,
            **{k: float(np.mean([c[k] for c in costs])) for k in ("calls", "input", "output")},
            "tokens": float(np.mean([c["input"] + c["output"] for c in costs])),
            "latency_median": float(np.median(lat)), "latency_p90": float(np.percentile(lat, 90)),
            "graph": float(np.mean(graph)) if None not in graph else None,
            "graph_no_log_rows": sum(g is None for g in graph)}
