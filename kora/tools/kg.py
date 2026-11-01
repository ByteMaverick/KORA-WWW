"""The Kora-facing graph capability."""

from __future__ import annotations

import json
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from kora.prompts import PROMPTS
from kora.provenance.models import Example, LocalSchema, PathSchema, SchemaEntry

__all__ = ["Example", "KGError", "KGProvider", "KGResult", "LocalSchema",
           "PathSchema", "SchemaEntry", "Triple", "load_schema", "load_schema_notes"]


def _read_schema_file(name: str) -> dict:
    """A graph's notes and relations, from the prompt section kg.<name>_notes ({} for a graph without one)."""
    # A file artifacts/prompts/kg/<name>_notes.yaml (metaqa_notes.yaml) overrides the shipped section (registry.py).
    prompt_id = f"kg.{name}_notes"
    if prompt_id not in PROMPTS:
        return {}
    return json.loads(PROMPTS.get(prompt_id).template.format())


def load_schema(name: str) -> list[dict]:
    return _read_schema_file(name).get("relations", [])


def load_schema_notes(name: str) -> list[str]:
    return _read_schema_file(name).get("notes", [])


class KGError(RuntimeError):
    """Transport or connectivity failure."""


class Triple(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subject: str
    predicate: str
    object: str


class KGResult(BaseModel):
    """Outcome of one structured query."""

    model_config = ConfigDict(extra="forbid")

    query: str
    rows: list[dict[str, str]] = Field(default_factory=list)
    error: str | None = None
    latency_seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def is_empty(self) -> bool:
        return self.ok and not self.rows

    def column(self, name: str) -> list[str]:
        seen: dict[str, None] = {}
        for row in self.rows:
            value = row.get(name)
            if value is not None:
                seen.setdefault(value, None)
        return list(seen)


@runtime_checkable
class KGProvider(Protocol):
    """Contract every graph backend satisfies."""

    def execute(self, query: str, sets: dict[str, list[str]] | None = None) -> KGResult: ...

    def local_schema(self, entity_id: str, limit: int = 20, wanted: str = "",
                     first: list[str] | None = None) -> LocalSchema: ...

    def label(self, entity_id: str) -> str | None: ...

    def properties(self, entity_id: str) -> list[SchemaEntry]: ...

    def opened(self, entity_id: str, relations: list[str],
               failed: set[tuple[str, str]] | None = None) -> LocalSchema: ...

    def labels(self, ids: list[str]) -> dict[str, str | None]: ...

    def known_types(self) -> set[str]: ...

    def schema(self) -> list[dict]: ...

    def schema_notes(self) -> list[str]: ...


def value_of(kg, entity_id: str, prop: str) -> list[str]:
    for query in (f"SELECT ?x WHERE {{ {entity_id} {prop} ?x }}", f"SELECT ?x WHERE {{ ?x {prop} {entity_id} }}"):
        found = list(dict.fromkeys(str(row["?x"]) for row in kg.execute(query).rows if row.get("?x")))
        if found:
            return found
    return []
