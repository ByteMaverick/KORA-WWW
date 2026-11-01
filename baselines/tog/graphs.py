"""
tog/graphs.py
--------------
The authors' ToG reaches the graph through four things in freebase_func.py: its SPARQL templates, execurte_sparql,
entity_search (which keeps only Freebase ids, "m."), and id2entity_name_or_type. On Freebase they run as written,
at the harness's endpoint. On MetaQA (Neo4j) the same four are answered by the harness's graph helpers
(harness/neo4j_client.py), returning the same shapes the authors' code reads:
[{"relation": {"value": ...}}], [{"tailEntity": {"value": ...}}], a name or "UnName_Entity". Everything else in the
authors' code is untouched.

Graph errors. A query the graph rejects as malformed (a relation the model wrote: HTTP 400, Neo4j's
CypherSyntaxError) finds nothing, as in the authors' code, and is counted (REJECTED, the row's graph_rejected). Any
other error is a graph that cannot answer: it is raised, so the question is an error and can be rerun rather than
answered from memory over "no edges". FAILURE keeps it for PoG, whose loop swallows exceptions.
"""
from __future__ import annotations

import os

from harness.neo4j_client import rejected

#: The released entity_search fills its incoming-direction template as (entity, relation) where the template wants
#: (relation, entity), so a search along an incoming relation finds nothing; PoG's authors fixed the same line.
#: "paper" swaps the two, as the paper's §2.1.2 describes; run.py sets it from harness/settings.py.
HEAD_ORDER = "paper"
_KG = {"name": "freebase", "tog": False}
MARK = "\t"
QUERIES = [0]          # graph requests of the question in flight (the harness's count, not the authors')
REJECTED = [0]         # of those, the ones the graph rejected as malformed (they found nothing)
FAILURE: list = [None]  # the error of a graph that could not answer, in the question in flight


def reset() -> None:
    """A new question: its counts and its failure start empty."""
    QUERIES[0], REJECTED[0], FAILURE[0] = 0, 0, None


def is_node(entity_id) -> bool:
    """An entity id on a non-Freebase graph (the authors' "m." test): MetaQA has nodes only."""
    return _KG["name"] == "metaqa"


def guarded(call, *args, empty=list):
    """One graph request: a rejected query returns empty() (counted); any other error is raised."""
    try:
        return call(*args)
    except Exception as e:
        if not rejected(e):
            FAILURE[0] = e
            raise
        REJECTED[0] += 1
        print(f"  [official graph] rejected, finds nothing: {type(e).__name__}: {str(e)[:160]}")
        return empty()


def _metaqa():
    from harness import neo4j_client as kg
    return kg


def _execute(query: str) -> list:
    """Their execurte_sparql for MetaQA: the templates below are tab-separated requests, answered under the
    graph-error rule (guarded)."""
    QUERIES[0] += 1
    return guarded(_answer, query)


def _answer(query: str) -> list:
    kind, *parts = query.split(MARK)
    if kind == "HEAD_ENT" and _KG["tog"] and HEAD_ORDER == "paper":
        parts = [parts[1], parts[0]]          # _fix_head_order sends (relation, entity)
    m = _metaqa()
    if kind == "HEAD_REL":
        return [{"relation": {"value": r}} for r in m.get_head_relations(parts[0])]
    if kind == "TAIL_REL":
        return [{"relation": {"value": r}} for r in m.get_tail_relations(parts[0])]
    relation = parts[1]
    if kind == "TAIL_ENT":
        return [{"tailEntity": {"value": e}} for e in m.get_tail_entities(parts[0], relation)]
    if kind == "HEAD_ENT":
        if _KG["tog"] and HEAD_ORDER == "as_released":
            return []        # the released ToG's query here asks "?x <entity> <relation>", which finds nothing
        return [{"tailEntity": {"value": e}} for e in m.get_head_entities(parts[0], relation)]
    raise ValueError(f"[tog] unknown graph request {kind!r}")


def _entity_search(fb):
    def entity_search(entity, relation, head=True):
        # The authors' entity_search with its Freebase filter (entity.startswith("m.")) read as is_node: entities
        # only, literal values dropped, as on Freebase.
        if head:
            entities = fb.execurte_sparql(fb.sparql_tail_entities_extract % (entity, relation))
        else:
            entities = fb.execurte_sparql(fb.sparql_head_entities_extract % (entity, relation))
        return [e for e in fb.replace_entities_prefix(entities) if is_node(e)]
    return entity_search


def _fix_head_order(fb):
    """HEAD_ORDER "paper": the authors' entity_search with the incoming-direction arguments in the template's order."""
    original = fb.entity_search

    def entity_search(entity, relation, head=True):
        if head:
            return original(entity, relation, head)
        return fb.replace_entities_prefix(fb.execurte_sparql(fb.sparql_head_entities_extract % (relation, entity)))
    fb.entity_search = entity_search


def _id2name(entity_id):
    QUERIES[0] += 1
    return _metaqa().id2entity_name_or_type(entity_id) or "UnName_Entity"


def install(fb, kg_name: str) -> None:
    """Point the authors' freebase_func module at the question's graph."""
    _KG["name"], _KG["tog"] = kg_name, True
    if not hasattr(fb, "_tog_original"):
        fb._tog_original = {k: getattr(fb, k) for k in ("sparql_head_relations", "sparql_tail_relations",
                                                         "sparql_tail_entities_extract", "sparql_head_entities_extract",
                                                         "execurte_sparql", "entity_search", "id2entity_name_or_type")}
    if kg_name == "freebase":
        for k, v in fb._tog_original.items():
            setattr(fb, k, v)
        fb.SPARQLPATH = os.getenv("SPARQL_ENDPOINT", "http://localhost:8890/sparql")   # harness: the endpoint
        original = fb._tog_original["execurte_sparql"]

        def counted(sparql_query):
            QUERIES[0] += 1
            return guarded(original, sparql_query)     # rejected: nothing; an outage raises
        fb.execurte_sparql = counted
        if HEAD_ORDER == "paper":
            _fix_head_order(fb)
        return
    if kg_name != "metaqa":
        raise NotImplementedError(f"[tog] no graph for kg_name={kg_name!r}: Freebase or MetaQA")
    fb.sparql_head_relations = f"HEAD_REL{MARK}%s"
    fb.sparql_tail_relations = f"TAIL_REL{MARK}%s"
    fb.sparql_tail_entities_extract = f"TAIL_ENT{MARK}%s{MARK}%s"
    fb.sparql_head_entities_extract = f"HEAD_ENT{MARK}%s{MARK}%s"
    fb.execurte_sparql = _execute
    fb.entity_search = _entity_search(fb)
    if HEAD_ORDER == "paper":
        _fix_head_order(fb)
    fb.id2entity_name_or_type = _id2name
