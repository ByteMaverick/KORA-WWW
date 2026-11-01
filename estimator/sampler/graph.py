"""What the sampler needs from a graph, read-only: ids and names, nothing more."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Protocol

from .rows import Constraint, Hop, Node, Path


@dataclass(frozen=True)
class Rel:
    relation: str
    direction: str      # "out" | "in"
    count: int = 0      # edges from the node(s) through it, when known


class Graph(Protocol):
    bench: str

    def relations(self, nodes: list[Node]) -> list[Rel]:
        """The relations leaving (and entering) these nodes, deduplicated, counts summed where known."""

    def follow(self, nodes: list[Node], rel: Rel, limit: int = 200) -> list[Node]:
        """The nodes reached from `nodes` through `rel`, with names (None for unnamed nodes)."""

    def same_name(self, node: Node, limit: int = 3) -> list[Node]:
        """Other nodes carrying the same name (a different film called Halloween); [] when none."""

    def spine(self, question: dict) -> Path | None:
        """The gold query executed: its chain of hops ending on the gold set, or None when the gold cannot be run."""

    def spine_variants(self, question: dict, spine: Path) -> list[Path]:
        """The spine with one constraint dropped, one per constraint (source 'drop_constraint')."""


class Scorer(Protocol):
    def __call__(self, question: str, names: list[str]) -> list[float]:
        """A similarity of the question to each relation name, in order; higher is closer."""


def lexical_scorer(question: str, names: list[str]) -> list[float]:
    """Word overlap, for tests and for graphs without an embedder."""
    q = {w for w in question.lower().replace("?", " ").split() if len(w) > 2}
    out = []
    for n in names:
        words = {w for w in n.lower().replace(".", " ").replace("_", " ").replace("/", " ").split() if len(w) > 2}
        out.append(len(q & words) / (len(words) or 1))
    return out


class SimpleGraph:
    """An in-memory graph for tests: edges as (subject, relation, object), names by id, gold spines by qid."""
    bench = "test"

    def __init__(self, edges: list[tuple[str, str, str]], names: dict[str, str | None], spines: dict[str, list] | None = None):
        self.out: dict[str, dict[str, list[str]]] = {}
        self.inn: dict[str, dict[str, list[str]]] = {}
        for s, r, o in edges:
            self.out.setdefault(s, {}).setdefault(r, []).append(o)
            self.inn.setdefault(o, {}).setdefault(r, []).append(s)
        self.names = names
        self.spines = spines or {}

    def node(self, i: str) -> Node:
        return Node(i, self.names.get(i))

    def relations(self, nodes: list[Node]) -> list[Rel]:
        c: dict[tuple[str, str], int] = {}
        for n in nodes:
            for r, objs in self.out.get(n.id, {}).items():
                c[(r, "out")] = c.get((r, "out"), 0) + len(objs)
            for r, subs in self.inn.get(n.id, {}).items():
                c[(r, "in")] = c.get((r, "in"), 0) + len(subs)
        return [Rel(r, d, k) for (r, d), k in sorted(c.items(), key=lambda kv: -kv[1])]

    def follow(self, nodes: list[Node], rel: Rel, limit: int = 200) -> list[Node]:
        table = self.out if rel.direction == "out" else self.inn
        found: list[str] = []
        for n in nodes:
            found += table.get(n.id, {}).get(rel.relation, [])
        return [self.node(i) for i in dict.fromkeys(found)][:limit]

    def same_name(self, node: Node, limit: int = 3) -> list[Node]:
        return [self.node(i) for i, nm in self.names.items() if nm and nm == node.name and i != node.id][:limit]

    def _walk(self, question: dict, chain: list, source: str, skip_filter: int | None = None) -> Path | None:
        """Walk `chain`: (relation, direction) items, or (relation, direction, {"rel": r, "value": id}) for a constraint."""
        anchors = [self.node(i) for i in question["topic_entity_ids"]]
        hops, current, constraints = [], anchors, []
        for k, item in enumerate(chain):
            relation, direction = item[0], item[1]
            objs = self.follow(current, Rel(relation, direction))
            if len(item) > 2 and k != skip_filter:
                f = item[2]
                objs = [o for o in objs if f["value"] in {n.id for n in self.follow([o], Rel(f["rel"], "out"))}]
                constraints.append(Constraint(k + 1, f["rel"], self.names.get(f["value"]) or f["value"]))
            if not objs:
                return None
            hops.append(Hop(current, relation, direction, objs)); current = objs
        return Path(anchors, hops, source, constraints=constraints)

    def spine(self, question: dict) -> Path | None:
        chain = self.spines.get(question["id"])
        return self._walk(question, chain, "spine") if chain else None

    def spine_variants(self, question: dict, spine: Path) -> list[Path]:
        chain = self.spines.get(question["id"]) or []
        out = []
        for k, item in enumerate(chain):
            if len(item) > 2:
                p = self._walk(question, chain, "drop_constraint", skip_filter=k)
                if p is not None:
                    out.append(p)
        return out


def random_walk(graph: Graph, anchors: list[Node], length: int, rng: random.Random, per_hop_limit: int = 200) -> Path | None:
    hops, current = [], anchors
    for _ in range(max(length, 1)):
        rels = graph.relations(current)
        if not rels:
            break
        rel = rng.choice(rels)
        objs = graph.follow(current, rel, per_hop_limit)
        if not objs:
            break
        hops.append(Hop(current, rel.relation, rel.direction, objs)); current = objs
    return Path(anchors, hops, "random") if hops else None
