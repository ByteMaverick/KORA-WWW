"""What the decomposer is allowed to see."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from kora.provenance.models import EntityRef, RequirementState, RequirementStatus

__all__ = ["DecompositionView", "OpenRequirement"]

SHOWN = 12


class OpenRequirement(BaseModel):
    """One line of the contract this task is accountable for, unresolved."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    goal: str
    required_inputs: list[str] = Field(default_factory=list)
    expected_output: str = ""
    expected_entity_type: str = ""
    critical: bool = True
    weight: float = 1.0
    blocked_by: list[str] = Field(default_factory=list)
    diagnosis: str = ""


class DecompositionView(BaseModel):
    """One task, and what Kora currently knows that bears on splitting it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str = ""

    requirements: list[OpenRequirement] = Field(default_factory=list)
    resolved: dict[str, list[str]] = Field(default_factory=dict)

    bindings: dict[str, list[EntityRef]] = Field(default_factory=dict)

    claims: list[str] = Field(default_factory=list)

    statistics: dict[str, float] = Field(default_factory=dict)

    @property
    def open_requirement_ids(self) -> list[str]:
        return [r.id for r in self.requirements]


def _open(requirement: RequirementState) -> OpenRequirement:
    return OpenRequirement(
        id=requirement.id, goal=requirement.description,
        critical=requirement.critical, weight=requirement.weight,
        blocked_by=list(requirement.blocking_task_ids),
        diagnosis=requirement.diagnosis or "",
    )


def of(state, task_id: str, statistics: dict | None = None) -> DecompositionView:
    """The projection, built by the Controller."""
    task = state.task(task_id)
    contract = state.contract
    wanted = set(task.requirement_ids)
    requirements = [
        _open(r)
        for r in (contract.requirements if contract else [])
        if r.id in wanted and r.status is not RequirementStatus.RESOLVED
    ]
    resolved = {
        r.id: [str(r.resolved_value)] if r.resolved_value is not None else []
        for r in (contract.requirements if contract else [])
        if r.id in wanted and r.status is RequirementStatus.RESOLVED
    }

    bindings = {}
    for role, refs in task.bindings.roles.items():
        if not refs:
            continue
        bindings[role] = list(refs[:SHOWN])

    return DecompositionView(
        task_id=task_id,
        requirements=requirements,
        claims=list(task.satisfies),
        resolved=resolved,
        bindings=bindings,
        statistics=dict(statistics or {}),
    )
