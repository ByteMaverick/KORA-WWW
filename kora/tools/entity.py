"""Entity resolution: surface name -> grounded graph identifier."""

from __future__ import annotations

import time

from pydantic import BaseModel, ConfigDict

from kora.tools.kg import KGError, KGProvider

FAILURE_COOLDOWN_SECONDS = 60.0


class EntityMatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str | None = None
    score: float = 0.0
    prominence: float = 0.0


class EntityResolver:
    """Resolves names against a graph's name property."""

    def __init__(
        self,
        kg: KGProvider,
        name_property: str = "name",
        alias_properties: tuple[str, ...] = (),
        id_property: str = "mid",
        limit: int = 5,
        anchor_label: str = "",
        prominence_expr: str = "",
    ):
        self.kg = kg
        self.name_property = name_property
        self.alias_properties = tuple(alias_properties)
        self.id_property = id_property
        self.limit = limit
        self.anchor_label = anchor_label or getattr(kg, "anchor_label", "") or ""
        self.prominence_expr = prominence_expr or getattr(kg, "prominence_expr", "") or ""
        self._cache: dict[str, list[EntityMatch]] = {}
        self._failed: dict[str, float] = {}

    @property
    def _order(self) -> str:
        return f"ORDER BY {self.prominence_expr} DESC\n" if self.prominence_expr else ""

    @property
    def _return(self) -> str:
        extra = f", {self.prominence_expr} AS prominence" if self.prominence_expr else ""
        return (f"RETURN n.{self.id_property} AS id, "
                f"n.{self.name_property} AS name{extra}\n")

    def _matches(self, rows, score: float) -> list[EntityMatch]:
        return [
            EntityMatch(id=str(r["id"]), name=r.get("name"), score=score,
                        prominence=float(r.get("prominence") or 0.0))
            for r in rows if r.get("id")
        ]

    def resolve(self, name: str) -> list[EntityMatch]:
        key = name.strip().lower()
        if not key:
            return []
        if key in self._cache:
            return self._cache[key]
        failed_at = self._failed.get(key)
        if failed_at is not None and time.monotonic() - failed_at < FAILURE_COOLDOWN_SECONDS:
            return []   # asked and failed moments ago; do not stall on it again

        try:
            matches = self._lookup(name)
        except KGError:
            self._failed[key] = time.monotonic()
            return []
        self._failed.pop(key, None)
        self._cache[key] = matches
        return matches

    def resolve_one(self, name: str) -> str | None:
        matches = self.resolve(name)
        return matches[0].id if matches else None

    def _lookup(self, name: str) -> list[EntityMatch]:
        native = getattr(self.kg, "resolve_name", None)
        if native is not None:
            # A `KGError` propagates to `resolve`, which must not cache it.
            return [EntityMatch(id=mid, name=found, score=1.0)
                    for mid, found in native(name, self.limit)]

        runner = getattr(self.kg, "_run", None)
        if runner is None:
            label = getattr(self.kg, "label", None)
            if label is not None and label(name):
                return [EntityMatch(id=name, name=name, score=1.0)]
            return []

        node = f"(n:{self.anchor_label.split('|')[0]})" if self.anchor_label else "(n)"
        exact_cypher = (
            f"MATCH {node} WHERE n.{self.name_property} = $name\n"
            f"{self._return}"
            f"{self._order}"
            f"LIMIT {self.limit}"
        )
        pooled: dict[str, dict] = {}
        for variant in dict.fromkeys([name, name.title(), name.capitalize(), name.upper()]):
            rows = runner(exact_cypher, name=variant)
            for row in rows:
                if row.get("id") and str(row["id"]) not in pooled:
                    pooled[str(row["id"])] = row

        if not pooled:
            for alias in self.alias_properties:
                alias_cypher = (f"MATCH {node} WHERE n.{alias} = $name\n"
                                f"{self._return}{self._order}LIMIT {self.limit}")
                for row in runner(alias_cypher, name=name):
                    if row.get("id") and str(row["id"]) not in pooled:
                        pooled[str(row["id"])] = row
                if pooled:
                    break

        matches = self._matches(pooled.values(), 1.0)
        matches.sort(key=lambda m: -m.prominence)
        return matches[: self.limit]
