"""MetaQA on Neo4j, over `kora.tools.neo4j.Neo4jKG`.

Nodes are `Entity`; a node's id is its `id` property and MetaQA's ids are their names, so `Node.name` is the node's
`name` (falling back to the id). Relations are the nine edge types, walked in both directions. No mediators, no
constraints: the spine is the gold hops from the topic entities.

The graph is read from the server's default database `neo4j`, overridable with NEO4J_METAQA_DATABASE; the server is
NEO4J_METAQA_URI.
"""
from __future__ import annotations

import ast
import os
import re

from ..graph import Rel
from ..rows import Hop, Node, Path

URI = "bolt://localhost:7687"
DATABASE = "neo4j"
_REL = re.compile(r"[^A-Za-z0-9_]")
#: Edge rows read per direction when counting relations; the count of a hub is then a floor, never unbounded.
COUNT_ROWS = 20000
MAX_NODES = 200
SPINE_LIMIT = 5000


def _rel(name: str) -> str:
    return "`" + _REL.sub("", name) + "`"


class MetaQAGraph:
    def __init__(self, kg, bench: str = "metaqa"):
        self.kg = kg
        self.bench = bench

    def _node(self, id_, name) -> Node:
        return Node(str(id_), str(name) if name is not None else str(id_))

    def relations(self, nodes: list[Node]) -> list[Rel]:
        ids = list(dict.fromkeys(n.id for n in nodes))[:MAX_NODES]
        if not ids:
            return []
        counts: dict[tuple[str, str], int] = {}
        for direction, pattern in (("out", "(n)-[r]->()"), ("in", "(n)<-[r]-()")):
            cypher = (f"MATCH (n:Entity) WHERE n.id IN $ids WITH n MATCH {pattern} "
                      f"WITH type(r) AS t LIMIT {COUNT_ROWS} RETURN t, count(*) AS c")
            for row in self.kg.ask(cypher, ids=ids):
                counts[(str(row["t"]), direction)] = int(row["c"])
        return [Rel(r, d, c) for (r, d), c in sorted(counts.items(), key=lambda kv: -kv[1])]

    def follow(self, nodes: list[Node], rel: Rel, limit: int = 200) -> list[Node]:
        ids = list(dict.fromkeys(n.id for n in nodes))[:MAX_NODES * 10]
        if not ids:
            return []
        pattern = (f"(n)-[:{_rel(rel.relation)}]->(m:Entity)" if rel.direction == "out"
                   else f"(n)<-[:{_rel(rel.relation)}]-(m:Entity)")
        cypher = (f"MATCH (n:Entity) WHERE n.id IN $ids MATCH {pattern} "
                  f"RETURN DISTINCT m.id AS id, m.name AS name LIMIT {int(limit)}")
        return [self._node(r["id"], r["name"]) for r in self.kg.ask(cypher, ids=ids) if r["id"] is not None]

    def same_name(self, node: Node, limit: int = 3) -> list[Node]:
        # MetaQA has one node per name (checked: no name is shared), so there is nothing to confuse.
        return []

    def _gold(self, question: dict) -> dict | None:
        gold = question.get("gold")
        if isinstance(gold, str):
            try:
                gold = ast.literal_eval(gold)
            except (ValueError, SyntaxError):
                return None
        return gold if isinstance(gold, dict) else None

    def _anchors(self, question: dict) -> list[Node]:
        te = question.get("topic_entity") or {}
        return [Node(str(i), str(n or i)) for i, n in te.items()]

    def spine(self, question: dict) -> Path | None:
        gold = self._gold(question)
        hops_spec = (gold or {}).get("hops") or []
        anchors = self._anchors(question)
        if not hops_spec or not anchors:
            return None
        hops, current = [], anchors
        for h in hops_spec:
            rel, direction = h.get("relation"), h.get("direction")
            if not rel or direction not in ("in", "out"):
                return None
            objs = self.follow(current, Rel(rel, direction), SPINE_LIMIT)
            if not objs:
                return None
            hops.append(Hop(current, rel, direction, objs)); current = objs
        return Path(anchors, hops, "spine")

    def spine_variants(self, question: dict, spine: Path) -> list[Path]:
        return []


def make_graph(bench: str) -> MetaQAGraph:
    from kora.tools.neo4j import Neo4jKG
    kg = Neo4jKG(uri=os.environ.get("NEO4J_METAQA_URI", URI),
                 database=os.environ.get("NEO4J_METAQA_DATABASE", DATABASE), id_property="id",
                 anchor_label="Entity", hub_relations=set(), prominence_expr="", schema_name="metaqa")
    return MetaQAGraph(kg, bench)
