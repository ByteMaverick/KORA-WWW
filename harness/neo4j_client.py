"""The MetaQA graph over Bolt, also in the shape ToG's and PoG's code reads; the graph-error rule (README)."""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from neo4j import GraphDatabase

NEO4J_URI      = os.getenv("NEO4J_METAQA_URI", "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "korakora")
NEO4J_DATABASE = os.getenv("NEO4J_METAQA_DATABASE", "metaqa-db")

_driver = None


def get_driver():
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    return _driver


def run_cypher(query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Execute one Cypher query against metaqa-db and return a list of row dicts."""
    driver = get_driver()
    with driver.session(database=NEO4J_DATABASE) as session:
        result = session.run(query, params or {})
        return [dict(record) for record in result]


def close_driver() -> None:
    global _driver
    if _driver is not None:
        _driver.close()
        _driver = None


def entity_exists(entity_id: str) -> bool:
    rows = run_cypher(
        "MATCH (n:Entity {id: $id}) RETURN count(n) AS c", {"id": entity_id}
    )
    return bool(rows) and rows[0]["c"] > 0


def get_neighbours(entity_id: str, relation: Optional[str] = None, limit: int = 200) -> List[dict]:
    """One-hop neighbours both ways, at most `limit` per direction (a LIMIT after a UNION binds its last branch)."""
    if relation:
        query = (
            "MATCH (n:Entity {id: $id})-[r:" + relation + "]->(m:Entity) "
            "RETURN m.id AS id, m.name AS name, type(r) AS relation, 'out' AS direction "
            "LIMIT $limit "
            "UNION "
            "MATCH (n:Entity {id: $id})<-[r:" + relation + "]-(m:Entity) "
            "RETURN m.id AS id, m.name AS name, type(r) AS relation, 'in' AS direction "
            "LIMIT $limit"
        )
    else:
        query = (
            "MATCH (n:Entity {id: $id})-[r]->(m:Entity) "
            "RETURN m.id AS id, m.name AS name, type(r) AS relation, 'out' AS direction "
            "LIMIT $limit "
            "UNION "
            "MATCH (n:Entity {id: $id})<-[r]-(m:Entity) "
            "RETURN m.id AS id, m.name AS name, type(r) AS relation, 'in' AS direction "
            "LIMIT $limit"
        )
    return run_cypher(query, {"id": entity_id, "limit": limit})


def get_relations(entity_id: str) -> List[str]:
    """Distinct relation types touching entity_id, either direction."""
    rows = run_cypher(
        "MATCH (n:Entity {id: $id})-[r]-() RETURN DISTINCT type(r) AS relation",
        {"id": entity_id},
    )
    return sorted({r["relation"] for r in rows})


def get_label(entity_id: str) -> str:
    rows = run_cypher(
        "MATCH (n:Entity {id: $id}) RETURN n.name AS name LIMIT 1", {"id": entity_id}
    )
    return rows[0]["name"] if rows else entity_id


def search_entities(term: str, limit: int = 20) -> List[dict]:
    """CONTAINS search over entity names (case-insensitive)."""
    rows = run_cypher(
        "MATCH (n:Entity) WHERE toLower(n.name) CONTAINS toLower($term) "
        "RETURN n.id AS id, n.name AS name LIMIT $limit",
        {"term": term, "limit": limit},
    )
    return rows


def rejected(error: BaseException) -> bool:
    """The graph refused the query itself (malformed), as opposed to failing to answer it."""
    try:
        from SPARQLWrapper.SPARQLExceptions import QueryBadFormed
        if isinstance(error, QueryBadFormed):
            return True
    except ImportError:
        pass
    try:
        from neo4j.exceptions import CypherSyntaxError
        if isinstance(error, CypherSyntaxError):
            return True
    except ImportError:
        pass
    try:
        import requests
    except ImportError:
        return False
    response = getattr(error, "response", None)
    return isinstance(error, requests.HTTPError) and response is not None and response.status_code == 400


def id2entity_name_or_type(entity_id: str, kg_name: str = None) -> str:
    """An entity's name (on MetaQA, its id); "UnName_Entity" for an unknown id, as ToG's code expects."""
    if not entity_id or entity_id in ("[FINISH_ID]",):
        return "UnName_Entity"
    if not entity_exists(entity_id):
        return "UnName_Entity"
    return get_label(entity_id)


def abandon_rels(relation: str, kg_name: str = None) -> bool:
    """Never: MetaQA has no infrastructure relations for ToG to drop."""
    return False


def get_head_relations(entity_id: str) -> list:
    """Relation types for which entity_id is the SUBJECT (outgoing edges)."""
    neighbours = get_neighbours(entity_id)
    return sorted({n["relation"] for n in neighbours if n["direction"] == "out"})


def get_tail_relations(entity_id: str) -> list:
    """Relation types for which entity_id is the OBJECT (incoming edges)."""
    neighbours = get_neighbours(entity_id)
    return sorted({n["relation"] for n in neighbours if n["direction"] == "in"})


def get_tail_entities(head_id: str, relation: str) -> list:
    """Entities head_id -[relation]-> ?."""
    neighbours = get_neighbours(head_id, relation=relation)
    return [n["id"] for n in neighbours if n["direction"] == "out"]


def get_head_entities(tail_id: str, relation: str) -> list:
    """Entities ? -[relation]-> tail_id."""
    neighbours = get_neighbours(tail_id, relation=relation)
    return [n["id"] for n in neighbours if n["direction"] == "in"]
