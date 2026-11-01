"""The types a decomposition is written in: what a task asks for, and the children that come back."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from kora.provenance.models import Spend


class DecompositionError(RuntimeError):
    """The decomposer could not turn this task into runnable children."""

    spend: Spend | None = None


class DecompositionRequest(BaseModel):
    """One task and what is known around it."""

    model_config = ConfigDict(extra="forbid")

    task_id: str


class OutputSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = ""
    type: str = "entity_set"
    entity_type: str | None = None


class Dependency(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    from_id: str = Field(alias="from")
    to_id: str = Field(alias="to")


class Subtask(BaseModel):
    """One child goal."""

    model_config = ConfigDict(extra="forbid")

    id: str
    goal: str
    required_inputs: list[str] = Field(default_factory=list)
    satisfies: list[str] = Field(default_factory=list)
    output: OutputSpec = Field(default_factory=OutputSpec)

    @property
    def consumes(self) -> list[str]:
        return [x[len("output:"):] for x in self.required_inputs if x.startswith("output:")]

    @property
    def roles(self) -> list[str]:
        return [x for x in self.required_inputs if not x.startswith("output:")]


class DecompositionResult(BaseModel):
    """The children of one task, and what producing them cost."""

    model_config = ConfigDict(extra="forbid")

    applicable: bool = True
    recomposition: str = ""

    task_id: str
    subtasks: list[Subtask] = Field(default_factory=list)
    dependencies: list[Dependency] = Field(default_factory=list)
    completion: list[str] = Field(default_factory=list)
    combine: str = "union"
    rationale: str = ""
    spend: Spend = Field(default_factory=Spend)

    def edges(self) -> list[tuple[str, str]]:
        stated = [(d.from_id, d.to_id) for d in self.dependencies]
        implied = [(p, c.id) for c in self.subtasks for p in c.consumes]
        return list(dict.fromkeys(stated + implied))

    def _waits(self) -> dict[str, set[str]]:
        waits: dict[str, set[str]] = {c.id: set() for c in self.subtasks}
        for before, after in self.edges():
            if before in waits and after in waits:
                waits[after].add(before)
        return waits

    def ordered(self) -> list[str]:
        waits, done = self._waits(), []
        while len(done) < len(waits):
            ready = [c for c in waits if c not in done and waits[c] <= set(done)]
            if not ready:
                break
            done.extend(ready)
        return done
