"""Which executable cut to take."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from kora.modules.decomposition.structural.graph import ConstraintGraph, Cut, Leaf

__all__ = [
    "CapabilityThreshold",
    "HeuristicSolvability",
    "Scored",
    "Solvability",
    "TieBreakPolicy",
    "choose",
    "score_all",
]

COMFORTABLE = 1
WIDE = 25


@runtime_checkable
class Solvability(Protocol):
    def score(self, leaf: Leaf, graph: ConstraintGraph,
              statistics: dict[str, float]) -> float: ...


class HeuristicSolvability:
    """Arithmetic over what the view already carries."""

    def score(self, leaf: Leaf, graph: ConstraintGraph,
              statistics: dict[str, float] | None = None) -> float:
        statistics = statistics or {}
        s = 1.0
        extra = max(0, len(leaf.constraint_ids) - COMFORTABLE)
        s *= 0.6 ** extra

        constants = {a for cid in leaf.constraint_ids
                     for a in graph.constraint(cid).constants}
        if not (constants & set(graph.grounded)) and not graph.domains.get(leaf.output):
            s *= 0.5

        unknowns = {v for cid in leaf.constraint_ids
                    for v in graph.constraint(cid).variables} - {leaf.output}
        s *= 0.75 ** len(unknowns - set(graph.domains))

        widest = max([len(graph.domains.get(v, [])) for v in leaf.inputs] or [0])
        if widest > WIDE:
            s *= WIDE / widest

        hubness = max([statistics.get(f"hub:{a}", 0.0) for a in constants] or [0.0])
        s *= 1.0 - min(0.9, hubness)
        return round(max(0.0, min(1.0, s)), 4)


class CapabilityThreshold(BaseModel):
    """The executor's capability boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    at: float = 0.7

    def clears(self, score: float) -> bool:
        return score >= self.at


class TieBreakPolicy(BaseModel):
    """Lexicographic, not weighted."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    by: tuple[str, ...] = ("cost", "width", "depth")


class Scored(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    cut: Cut
    scores: dict[str, float] = Field(default_factory=dict)
    weakest: float = 0.0
    cost: int = 0
    width: int = 0

    @property
    def depth(self) -> int:
        return self.cut.depth


def score_all(cuts: list[Cut], graph: ConstraintGraph,
              solvability: Solvability | None = None,
              statistics: dict[str, float] | None = None,
              per_leaf_tokens: int = 40_000) -> list[Scored]:
    solvability = solvability or HeuristicSolvability()
    scored: list[Scored] = []
    for cut in cuts:
        scores = {leaf.id: solvability.score(leaf, graph, statistics or {})
                  for leaf in cut.leaves}
        width = max([len(graph.domains.get(leaf.output, [])) for leaf in cut.leaves] or [1])
        scored.append(Scored(cut=cut, scores=scores,
                             weakest=min(scores.values()) if scores else 0.0,
                             cost=per_leaf_tokens * len(cut.leaves),
                             width=max(1, width)))
    return scored


def choose(scored: list[Scored], threshold: CapabilityThreshold | None = None,
           tie_break: TieBreakPolicy | None = None) -> Scored | None:
    """Maximin, then lexicographic."""
    if not scored:
        return None
    threshold = threshold or CapabilityThreshold()
    tie_break = tie_break or TieBreakPolicy()
    clears = [s for s in scored if threshold.clears(s.weakest)]
    if not clears:
        clears = [max(scored, key=lambda s: s.weakest)]
    order = {"cost": lambda s: s.cost, "width": lambda s: s.width,
             "depth": lambda s: s.depth}
    return min(clears, key=lambda s: tuple(order[name](s) for name in tie_break.by))
