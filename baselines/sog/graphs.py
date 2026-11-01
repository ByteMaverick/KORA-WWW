"""
sog/graphs.py
--------------
SoG's one tool, get_adjacent_relations_and_entities(entity, direction, properties_to_filter_for), on both graphs
the harness serves. Freebase is the authors' own function (upstream/few_shot_llm_tools_unified_helper.py),
unchanged except that the endpoint is the harness's (harness:). MetaQA (Neo4j) is ours: the rows come from the
harness's graph helper and go through the authors' own post-processing, so the model sees the same table with the
same rules as on Freebase:

  * columns property | propertyLabel | value | valueLabel (upstream convert_tool_result_to_table);
  * duplicates removed (remove_duplicates_from_sparql_result);
  * more than 50 rows and no properties_to_filter_for: only the distinct properties are returned
    (convert_properties_to_table), and the caller then asks the model to filter (upstream main.py);
  * at most 2,000 rows (the authors' code, top_k = 2000; the paper's p is 1,000).

Graph errors (`rejected` in harness/neo4j_client.py). A query the graph rejects as malformed (HTTP 400, CypherSyntaxError) finds
nothing, as in the authors' Freebase tool, whose execute_sparql_query retries 3 times then returns [], and is counted
(rejected(), the row's graph_rejected). A graph that cannot answer is raised, on Freebase after their retries, and
kept (failure()): their loop would hand the exception to the model as a message, so run.py sends no model turn after
a failure and raises it.
"""
from __future__ import annotations

import os
import threading

from SPARQLWrapper import SPARQLWrapper

from baselines.sog import upstream_path  # noqa: F401  (puts the authors' folder on sys.path)
import few_shot_llm_tools_unified_helper as helper
from harness.neo4j_client import rejected as _is_rejected

HIGH_DEGREE = 50      # the authors' k: above it, properties only
MAX_ROWS = 2000       # the authors' top_k

#: Tool calls of the question in flight. The authors run a turn's tool calls on worker threads, so the count is
#: one per process (the harness answers one question at a time in a process), under a lock.
_lock = threading.Lock()
_count = [0]
_rejected = [0]                # of the tool calls, those whose query the graph rejected (they found nothing)
_failure: list = [None]        # the error of a graph that could not answer, in the question in flight
_last = threading.local()      # the error of this thread's last Freebase request (None when it succeeded)


def queries() -> int:
    return _count[0]


def rejected() -> int:
    return _rejected[0]


def failure():
    return _failure[0]


def reset() -> None:
    with _lock:
        _count[0], _rejected[0], _failure[0] = 0, 0, None


def _counted() -> None:
    with _lock:
        _count[0] += 1


def _graph_error(error: BaseException) -> None:
    """A tool call's graph error: a rejection is counted (the call finds nothing); anything else is kept and
    raised."""
    with _lock:
        if _is_rejected(error):
            _rejected[0] += 1
            print(f"  [sog graph] rejected, finds nothing: {type(error).__name__}: {str(error)[:160]}")
            return
        if _failure[0] is None:
            _failure[0] = error
    raise error


class _Recorded(SPARQLWrapper):
    """harness: their helper's SPARQLWrapper, which records each request's error for this thread, so that after their
    retries (execute_sparql_query returns [] for any error) a graph that could not answer is told from no rows."""

    def query(self):
        try:
            result = super().query()
        except Exception as e:
            _last.error = e
            raise
        _last.error = None
        return result


helper.SPARQLWrapper = _Recorded


def _freebase(entity, direction, properties_to_filter_for=None, sparql_endpoint=None, return_format="markdown-short"):
    # harness: the endpoint is the harness's (SPARQL_ENDPOINT), not the authors' hard-coded localhost:8890.
    _counted()
    _last.error = None
    table = helper.get_adjacent_relations_and_entities_freebase(
        entity=entity, direction=direction, properties_to_filter_for=properties_to_filter_for,
        sparql_endpoint=os.getenv("SPARQL_ENDPOINT", "http://localhost:8890/sparql"), return_format=return_format)
    if _last.error is not None:           # their last attempt failed and they returned []: rejected, or raise
        _graph_error(_last.error)
    return table


def _rows_metaqa(entity: str, direction: str) -> list[dict]:
    from harness.neo4j_client import get_neighbours
    want = "out" if direction == "outgoing" else "in"
    return [{"property": r["relation"], "propertyLabel": r["relation"].replace("_", " ").lower(),
             "value": r["id"], "valueLabel": r["name"] if r["name"] != r["id"] else ""}
            for r in get_neighbours(entity, relation=None, limit=MAX_ROWS * 2) if r["direction"] == want]


def _ours(rows_of):
    def get_adjacent(entity, direction, properties_to_filter_for=None, sparql_endpoint=None,
                     return_format="markdown-short"):
        if direction not in ("outgoing", "incoming"):
            raise ValueError("direction must be 'outgoing' or 'incoming'.")
        _counted()
        try:
            rows = rows_of(str(entity).strip(), direction)
        except Exception as e:
            _graph_error(e)               # raises unless the graph rejected the query
            rows = []
        if properties_to_filter_for:
            wanted = {p.strip() for p in properties_to_filter_for}
            rows = [r for r in rows if r["property"] in wanted]
        result = helper.remove_duplicates_from_sparql_result(
            {"head": {"vars": ["property", "propertyLabel", "value", "valueLabel"]}, "results": {"bindings": rows}})
        bindings = result["results"]["bindings"]
        if len(bindings) > HIGH_DEGREE and not properties_to_filter_for and return_format == "markdown-short":
            unique = {}
            for b in bindings:
                unique.setdefault(b["property"], b["propertyLabel"])
            return helper.convert_properties_to_table(unique)
        result["results"]["bindings"] = bindings[:MAX_ROWS]
        return helper.convert_tool_result_to_table(result)
    return get_adjacent


def get_adjacent_fn(kg_name: str):
    if kg_name == "freebase":
        return _freebase
    if kg_name == "metaqa":
        return _ours(_rows_metaqa)
    raise NotImplementedError(f"[sog] no graph tool for kg_name={kg_name!r}")
