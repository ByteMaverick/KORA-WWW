"""
pog/graphs.py
--------------
The authors' PoG reaches the graph through the same SPARQL templates and execurte_sparql as ToG, plus
id2entity_name_or_type (it returns the id itself when there is no name) and provide_triple (names for Freebase
ids, "m.", the value as written otherwise). On Freebase they run as written at the harness's endpoint. On MetaQA the
requests are answered by the same helpers as tog/graphs.py, and provide_triple's "m." test reads "is an entity node"
(is_entity).

Graph errors as in tog/graphs.py: a query the graph rejects finds nothing (a name lookup keeps the id); a graph
that cannot answer raises, and run.py raises it again after their loop, whose bare except swallows it.
"""
from __future__ import annotations

import os

from baselines.tog import graphs as tog_graphs

QUERIES = tog_graphs.QUERIES
_KG = tog_graphs._KG


def is_entity(entity_id: str) -> bool:
    """The authors' test for an entity id worth expanding: Freebase "m." ids; on MetaQA an entity node
    (tog_graphs.is_node), never their "[FINISH_ID]"."""
    if _KG["name"] == "freebase":
        return str(entity_id).startswith("m.")
    return tog_graphs.is_node(entity_id) and not str(entity_id).startswith("[FINISH")


def _id2name(entity_id):
    QUERIES[0] += 1
    name = tog_graphs.guarded(tog_graphs._metaqa().id2entity_name_or_type, entity_id, empty=lambda: None)
    return name or entity_id                       # the authors' PoG returns the id when there is no name


def _provide_triple(fb):
    def provide_triple(entity_candidates_id, relation):
        # The authors' provide_triple with its Freebase test ("m.") read as "is an entity" (is_entity above).
        entity_candidates = []
        for entity_id in entity_candidates_id:
            if is_entity(entity_id):
                entity_candidates.append(fb.id2entity_name_or_type(entity_id))
            else:
                entity_candidates.append(entity_id)
        if len(entity_candidates) <= 1:
            return entity_candidates, entity_candidates_id
        ent_id_dict = dict(sorted(zip(entity_candidates, entity_candidates_id)))
        entity_candidates, entity_candidates_id = list(ent_id_dict.keys()), list(ent_id_dict.values())
        return entity_candidates, entity_candidates_id
    return provide_triple


def install(fb, kg_name: str) -> None:
    """Point the authors' freebase_func module at the question's graph."""
    _KG["name"], _KG["tog"] = kg_name, False
    keys = ("sparql_head_relations", "sparql_tail_relations", "sparql_tail_entities_extract",
            "sparql_head_entities_extract", "execurte_sparql", "id2entity_name_or_type", "provide_triple")
    if not hasattr(fb, "_pog_original"):
        fb._pog_original = {k: getattr(fb, k) for k in keys}
    for k, v in fb._pog_original.items():
        setattr(fb, k, v)
    if kg_name == "freebase":
        fb.SPARQLPATH = os.getenv("SPARQL_ENDPOINT", "http://localhost:8890/sparql")   # harness: the endpoint
        original = fb._pog_original["execurte_sparql"]

        def counted(sparql_query):
            QUERIES[0] += 1
            return tog_graphs.guarded(original, sparql_query)   # rejected: nothing; an outage raises
        fb.execurte_sparql = counted
        names = fb._pog_original["id2entity_name_or_type"]

        def id2name(entity_id):            # their lookup opens its own SPARQLWrapper: the same rule
            return tog_graphs.guarded(names, entity_id, empty=lambda: entity_id)
        fb.id2entity_name_or_type = id2name
        return
    if kg_name != "metaqa":
        raise NotImplementedError(f"[pog] no graph for kg_name={kg_name!r}: Freebase or MetaQA")
    mark = tog_graphs.MARK
    fb.sparql_head_relations = f"HEAD_REL{mark}%s"
    fb.sparql_tail_relations = f"TAIL_REL{mark}%s"
    fb.sparql_tail_entities_extract = f"TAIL_ENT{mark}%s{mark}%s"
    # PoG fills this template as (relation, entity): the request reads HEAD_ENT <relation> <entity>
    fb.sparql_head_entities_extract = f"HEAD_ENT_RE{mark}%s{mark}%s"
    fb.execurte_sparql = _execute
    fb.id2entity_name_or_type = _id2name
    fb.provide_triple = _provide_triple(fb)


def _execute(query: str) -> list:
    kind, *parts = query.split(tog_graphs.MARK)
    if kind == "HEAD_ENT_RE":                      # PoG's argument order, swapped to the helpers' (entity, relation)
        query = tog_graphs.MARK.join(["HEAD_ENT", parts[1], parts[0]])
    return tog_graphs._execute(query)
