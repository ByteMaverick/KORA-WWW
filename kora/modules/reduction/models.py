from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from kora.provenance.models import EntityRef, Spend

OPERATIONS = ("argmax", "argmin", "count", "filter", "intersect", "difference", "verify")
YES, NO = EntityRef(id="value:yes", label="yes", entity_type="value"), EntityRef(id="value:no", label="no", entity_type="value")

class Typed(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_by_name=True, validate_by_alias=True)

class Move(Typed):
    id: str = ""
    after: list[str] = Field(default_factory=list)
    for_ids: list[str] = Field(default_factory=list, alias="for")
    expects_kind: str = ""
    counts_as: Literal["answer", "superset"] = "answer"
    why: str = ""

class Operate(Move):
    kind: Literal["operate"] = "operate"
    op: str
    prop: str = ""
    cmp: str = ""
    value: str = ""
    over: list[str] = Field(default_factory=list)
    named: list[EntityRef] = Field(default_factory=list)

class Candidate(Typed):
    """A property an operate may read: `relation` or `relation / field`."""
    path: str
    samples: list[str] = Field(default_factory=list)
    coverage: int = 0
    sampled: int = 0

class Choice(Typed):
    """What one choice came to: the path, or None and the word match runs."""
    path: str | None = None
    why: str = ""
    error: str = ""
    called: bool = False
    cached: bool = False
    spend: Spend = Field(default_factory=Spend)

    @property
    def fallback(self) -> bool:
        return self.path is None
