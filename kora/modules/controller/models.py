"""What the Controller is given to spend, and what it hands back."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from kora.provenance.models import EntityRef


class Budget(BaseModel):
    """The run's clock in seconds, zero for none; tokens is recorded on the state, never a limit."""

    model_config = ConfigDict(extra="forbid")

    tokens: int = 0
    seconds: float = 0.0
    soft_seconds: float = 0.0


class ControlResult(BaseModel):
    """What one run of the Controller came to: the root done, or the clock or idleness, and why."""

    model_config = ConfigDict(extra="forbid")

    status: str = "timed_out"
    stop_reason: str = ""
    answer_entities: list[EntityRef] = Field(default_factory=list)
    rounds: int = 0
    decompositions: int = 0
    dispatches: int = 0
    waves: list[list[str]] = Field(default_factory=list)
    phase_at_finish: str = "open"
    log: list[str] = Field(default_factory=list)
