"""The wall clock: open, closing past soft_seconds, closed past the hard limit."""

from __future__ import annotations

import time


def now() -> float:
    return time.monotonic()


def phase(state) -> str:
    left = state.resources
    wall = now() - left.started_at
    if left.time_budget_seconds and wall >= left.time_budget_seconds:
        return "closed"
    if left.soft_seconds and wall >= left.soft_seconds:
        return "closing"
    return "open"
