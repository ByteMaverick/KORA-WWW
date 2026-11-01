"""Neo4j / Cypher knowledge graph backend."""

from __future__ import annotations

import os
import re
import threading
import time

from kora.tools.dialect import (COMPARISONS, comparisons, count_variable, literal_kind, parse_modifiers, parse_query,
                                property_absences, property_reads, set_anchors, type_constraints,
                                without_type_constraints)
from kora.tools.kg import (Example, KGError, KGResult, LocalSchema, PathSchema, SchemaEntry, load_schema,
                           load_schema_notes)

UNAVAILABLE = (ImportError, AttributeError, TypeError, ValueError, OSError, RuntimeError)
_SAFE = re.compile(r"[^A-Za-z0-9_]")
MID = re.compile(r"^[mg]\.[0-9a-z_]+$")
GENERIC_LABELS = ["Entity", "CVT", "common/topic", "Resource"]
TYPE_SAMPLE = 25
MEDIATORS_PER_RELATION = 10
HUB_DEGREE = 50_000
HUB_SAMPLE_EDGES = 4_000
EXAMPLES = 6
FETCH_CAP = 500
#: The first label of `one` that is not generic: the page's and the mediator routes' target type.
_ONE_TYPE = ("       [l IN labels(one) WHERE NOT l IN $generic "
             "        AND NOT l STARTS WITH 'base/' "
             "        AND NOT l STARTS WITH 'user/'][0] AS target_type")


class KGQueryRejected(KGError):
    """The graph rejected a statement the translator accepted."""


class CypherTranslationError(ValueError):
    """The dialect query could not be expressed in Cypher."""


def _var(token: str) -> str:
    return "v_" + _SAFE.sub("_", token[1:])


def _quote_type(predicate: str) -> str:
    return "`" + predicate.replace("`", "") + "`"


def to_cypher(query: str, id_property: str = "mid", limit: int = 500, anchor_label: str = "",
              name_property: str = "name", sets: dict[str, list[str]] | None = None) -> str:
    """Translate the compact triple dialect into a Cypher read query."""
    parsed = parse_query(query)
    if parsed is None:
        raise CypherTranslationError("expected SELECT ?vars WHERE { subject predicate object . ... }")
    variables, all_patterns = parsed
    wanted_types = type_constraints(all_patterns)
    reads = property_reads(all_patterns)
    patterns = without_type_constraints(all_patterns)
    if not variables or not patterns:
        raise CypherTranslationError("query selects nothing or matches nothing")

    node_names: dict[str, str] = {}
    conditions: list[str] = []
    rel_names: dict[str, str] = {}
    computed: dict[str, str] = {}
    constants = 0

    def node(term: str) -> str:
        nonlocal constants
        if term in node_names:
            return node_names[term]
        if term.startswith("?"):
            name = _var(term)
        else:
            name = f"c{constants}"
            constants += 1
            escaped = term.replace("\\", "\\\\").replace("'", "\\'")
            conditions.append(f"{name}.{id_property} = '{escaped}'")
            if anchor_label:
                name = f"{name}:{anchor_label}"
        conditions.extend(_type_condition(name.split(":")[0], kind) for kind in wanted_types.get(term, []))
        node_names[term] = name
        return name

    def ref(term: str) -> str:
        return node(term).split(":")[0]

    matches: list[str] = []
    for index, (subject, predicate, obj) in enumerate(patterns):
        left, right = node(subject), node(obj)
        if predicate.startswith("?"):
            rel = f"r{index}"
            rel_names[predicate] = rel
            matches.append(f"({left})-[{rel}]->({right})")
        else:
            matches.append(f"({left})-[:{_quote_type(predicate)}]->({right})")

    for subject, key, target in reads:
        if subject not in node_names:
            raise CypherTranslationError(f"{subject} reads property {key} but is never matched")
        computed[target] = _property_read(ref(subject), key)

    for variable, name in set_anchors(all_patterns):
        members = (sets or {}).get(name)
        if members is None:
            raise CypherTranslationError(f"set anchor {name} names a set this task does not hold")
        node(variable)
        literals = ", ".join("'" + m.replace("'", "\\'") + "'" for m in members)
        conditions.append(f"{ref(variable)}.{id_property} IN [{literals}]")

    for variable, op, literal in comparisons(all_patterns):
        symbol = COMPARISONS[op]
        source = computed.get(variable) or (f"{ref(variable)}.{name_property}" if variable in node_names else None)
        if source is None:
            raise CypherTranslationError(f"{variable} is compared but never bound")
        if literal_kind(literal) == "number":
            conditions.append(f"toFloat({source}) {symbol} {literal}")
        else:
            escaped = literal.replace("'", "\\'")
            conditions.append(f"{source} {symbol} '{escaped}'")

    for subject, key in property_absences(all_patterns):
        if subject not in node_names:
            raise CypherTranslationError(f"{subject} requires no {key} but is never matched")
        escaped = key.replace("'", "\\'")
        conditions.append(f"NOT '{escaped}' IN coalesce({ref(subject)}.prop_keys, [])")

    projections: list[str] = []
    for variable in variables:
        alias = variable[1:]
        if variable in rel_names:
            projections.append(f"type({rel_names[variable]}) AS {alias}")
        elif variable in computed:
            projections.append(f"{computed[variable]} AS {alias}")
        elif variable in node_names:
            projections.append(f"{node_names[variable].split(':')[0]}.{id_property} AS {alias}")
        else:
            raise CypherTranslationError(f"selected variable {variable} is unbound")

    counted = count_variable(query)
    if counted:
        if counted in rel_names or counted in computed:
            raise CypherTranslationError(f"COUNT({counted}) must count a graph node, "
                                         "not a relation or a property value")
        projections = [f"count(DISTINCT {ref(counted)}.{id_property}) AS {counted[1:]}"]
        keyword = "RETURN"
    else:
        keyword = "RETURN DISTINCT"

    where = f"\nWHERE {' AND '.join(conditions)}" if conditions else ""
    mods = parse_modifiers(query)
    order = ""
    if mods["order"] and not counted:
        var, direction = mods["order"]
        # A property read is ordered by its value, a node by its name; either is projected under its own column.
        if var in computed:
            column, value = var[1:], computed[var]
        elif var in node_names:
            column, value = f"{var[1:]}_name", f"{ref(var)}.{name_property}"
        else:
            raise CypherTranslationError(f"ORDER BY {var} is unbound")
        if not any(proj.endswith(f" AS {column}") for proj in projections):
            projections.append(f"{value} AS {column}")
        order = f"\nORDER BY {column} {direction}"
    take = mods["limit"] if mods["limit"] is not None else limit
    return f"MATCH {', '.join(matches)}{where}\n{keyword} {', '.join(projections)}{order}\nLIMIT {take}"


def _property_read(node: str, key: str) -> str:
    escaped = key.replace("'", "\\'")
    return (f"[i IN range(0, size(coalesce({node}.prop_keys, [])) - 1) "
            f"WHERE {node}.prop_keys[i] = '{escaped}' | {node}.prop_values[i]][0]")


def _type_condition(node: str, kind: str) -> str:
    escaped = kind.replace("`", "")
    return f"({node}:`{escaped}` OR {_property_read(node, 'type')} = '{escaped}')"


class Neo4jKG:
    """A :class:`~kora.tools.kg.KGProvider` backed by a Bolt connection."""

    def __init__(self, uri: str | None = None, user: str | None = None, password: str | None = None,
                 database: str | None = None, id_property: str = "mid", name_property: str = "name",
                 limit: int = 500, timeout: float = 60.0, anchor_label: str = "",
                 hub_relations: set[str] | None = None, prominence_expr: str = "", schema_name: str = ""):
        self.uri = uri or os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.environ.get("NEO4J_USER", "neo4j")
        self.password = password or os.environ.get("NEO4J_PASSWORD", "korakora")
        self.database = database or os.environ.get("NEO4J_DATABASE", "neo4j")
        self.schema_name = schema_name
        self.id_property = id_property
        self.name_property = name_property
        self.limit = limit
        self.anchor_label = anchor_label
        self.prominence_expr = prominence_expr
        self.hub_relations: set[str] = set(hub_relations or ())
        self._types: set[str] | None = None
        self._relations: list[str] | None = None
        self.timeout = timeout
        self.queries: list[str] = []
        self.calls = 0
        self._counter_lock = threading.Lock()
        self._probes: dict[str, LocalSchema] = {}
        self._probes_lock = threading.Lock()
        self._driver = None

    @property
    def driver(self):
        if self._driver is None:
            try:
                from neo4j import GraphDatabase
            except ImportError:
                raise KGError("the neo4j driver is missing; pip install neo4j") from None
            try:
                self._driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password),
                                                    notifications_min_severity="OFF")
            except UNAVAILABLE as exc:
                raise KGError(f"cannot connect to {self.uri}: {exc}") from None
        return self._driver

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    def health(self) -> bool:
        try:
            return self.ask("RETURN 1 AS ok")[0]["ok"] == 1
        except (KGError, IndexError, KeyError):
            return False

    def ask(self, cypher: str, **params) -> list[dict]:
        return self._run(cypher, **params)

    def _run(self, cypher: str, **params) -> list[dict]:
        with self._counter_lock:
            self.calls += 1
        try:
            with self.driver.session(database=self.database) as session:
                return [dict(record) for record in session.run(cypher, **params)]
        except KGError:
            raise
        except Exception as exc:
            if type(exc).__name__ == "ClientError" or "Neo.ClientError" in str(exc):
                raise KGQueryRejected(str(exc)) from None
            raise KGError(f"neo4j query failed on {self.uri}: {exc}") from None

    def execute(self, query: str, sets: dict[str, list[str]] | None = None) -> KGResult:
        self.queries.append(query)
        result = KGResult(query=query)
        try:
            cypher = to_cypher(query, self.id_property, self.limit, self.anchor_label, sets=sets)
        except CypherTranslationError as exc:
            result.error = f"malformed query: {exc}"
            return result
        started = time.monotonic()
        try:
            rows = self.ask(cypher)
        except KGQueryRejected as exc:
            result.error = f"the graph rejected the query: {str(exc)[:300]}"
            result.latency_seconds = time.monotonic() - started
            return result
        result.latency_seconds = time.monotonic() - started
        result.rows = [{f"?{key}": ("" if value is None else str(value)) for key, value in row.items()}
                       for row in rows]
        return result

    @property
    def _anchor(self) -> str:
        return f"n:{self.anchor_label}" if self.anchor_label else "n"

    @property
    def _me(self) -> str:
        """The anchor node matched by `$id`."""
        return f"({self._anchor} {{{self.id_property}: $id}})"

    def relation_types(self) -> list[str]:
        if self._relations is None:
            try:
                rows = self.ask("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType AS t")
                self._relations = sorted(str(r["t"]) for r in rows)
            except KGError:
                self._relations = []
        return self._relations

    def vocabulary(self) -> set[str]:
        return set(self.relation_types())

    def schema(self) -> list[dict]:
        return load_schema(self.schema_name or self.database)

    def schema_notes(self) -> list[str]:
        return load_schema_notes(self.schema_name or self.database)

    def is_hub(self, entity_id: str, cap: int = HUB_DEGREE) -> bool:
        cypher = f"MATCH {self._me}-[]-()\nWITH 1 AS one LIMIT {cap + 1}\nRETURN count(one) AS n"
        try:
            rows = self.ask(cypher, id=entity_id)
        except KGError:
            return False
        return bool(rows) and int(rows[0]["n"]) > cap

    def known_types(self) -> set[str]:
        if self._types is None:
            try:
                self._types = {str(r["label"]) for r in self.ask("CALL db.labels() YIELD label RETURN label")}
            except KGError:
                self._types = set()
        return self._types

    def local_schema(self, entity_id: str, limit: int = 20, wanted: str = "",
                     first: list[str] | None = None) -> LocalSchema:  # noqa: ARG002
        """`wanted` and `first` are accepted and ignored: one Cypher describes every relation."""
        entries: list[SchemaEntry] = []
        sampled = self.is_hub(entity_id)
        cut = f"WITH r, m ORDER BY m.{self.id_property} LIMIT {HUB_SAMPLE_EDGES}\n" if sampled else ""
        for direction, pattern in (("out", f"{self._me}-[r]->(m)"), ("in", f"(m)-[r]->{self._me}")):
            cypher = (
                f"MATCH {pattern}\n"
                f"{cut}"
                f"WITH type(r) AS predicate, m ORDER BY m.{self.id_property}\n"
                f"WITH predicate, count(*) AS n,\n"
                f"     head(collect(m)) AS one, collect(m)[..{TYPE_SAMPLE}] AS few\n"
                f"CALL {{ WITH few UNWIND few AS f UNWIND labels(f) AS l\n"
                f"        WITH l WHERE NOT l IN $generic\n"
                f"          AND NOT l STARTS WITH 'base/' AND NOT l STARTS WITH 'user/'\n"
                f"        WITH l, count(*) AS seen ORDER BY seen DESC\n"
                f"        RETURN collect(l)[..14] AS kinds }}\n"
                f"RETURN predicate, n,\n"
                f"       one.{self.id_property} AS sample,\n"
                f"       one.{self.name_property} AS sample_name,\n"
                f"       ('CVT' IN labels(one) OR one.{self.name_property} IS NULL)\n"
                f"         AS mediator, kinds,\n"
                f"       [f IN few[..{EXAMPLES}] WHERE f.{self.name_property} IS NOT NULL | "
                f"         [f.{self.id_property}, f.{self.name_property}, "
                f"          [l IN labels(f) WHERE NOT l IN $generic AND NOT l STARTS WITH 'base/' "
                f"           AND NOT l STARTS WITH 'user/'][0]]] AS examples,\n"
                f"{_ONE_TYPE}\n"
                f"ORDER BY n DESC LIMIT {FETCH_CAP}"
            )
            try:
                rows = self.ask(cypher, id=entity_id, generic=GENERIC_LABELS)
            except KGError:
                continue
            entries.extend(
                SchemaEntry(
                    predicate=str(row["predicate"]), direction=direction, count=int(row["n"]),
                    sample_object=None if row["sample"] is None else str(row["sample"]),
                    sample_name=row.get("sample_name"), target_type=row.get("target_type"),
                    target_types=list(row.get("kinds") or []), mediator=bool(row.get("mediator")),
                    examples=[Example(id=str(e[0]), name=e[1], kind=e[2])
                              for e in (row.get("examples") or []) if e and e[0] is not None],
                    sampled=sampled,
                )
                for row in rows
            )
        entries.sort(key=lambda entry: -entry.count)
        return LocalSchema(entity_id=entity_id, entries=entries[:FETCH_CAP],
                           paths=self.mediator_paths(entity_id, limit))

    def _probe(self, entity_id: str) -> LocalSchema:
        with self._probes_lock:
            if entity_id in self._probes:
                return self._probes[entity_id]
        found = self.local_schema(entity_id, limit=FETCH_CAP)
        with self._probes_lock:
            self._probes[entity_id] = found
        return found

    def properties(self, entity_id: str) -> list[SchemaEntry]:
        return self._probe(entity_id).entries

    def properties_with_samples(self, ids: list[str]) -> dict[str, dict[str, list[str]]]:
        found: dict[str, dict[str, list[str]]] = {}
        for entity_id in ids:
            paths: dict[str, list[str]] = {}
            direct = (f"MATCH {self._me}-[r]->(m)\n"
                      f"WITH type(r) AS p, collect(DISTINCT toString(m.{self.id_property}))[..2] AS samples\n"
                      f"RETURN p, samples LIMIT 300")
            for row in self._sampled(direct, entity_id):
                values = [v for v in row["samples"] or [] if v and not MID.match(v)]
                if values:
                    paths[str(row["p"])] = values
            carried = (f"MATCH {self._me}-[r]->(m)\n"
                       f"WHERE m:CVT OR m.{self.name_property} IS NULL\n"
                       f"WITH type(r) AS p, m LIMIT 2000\n"
                       f"UNWIND range(0, size(coalesce(m.prop_keys, [])) - 1) AS i\n"
                       f"WITH p, m.prop_keys[i] AS k, m.prop_values[i] AS v WHERE k <> 'type'\n"
                       f"RETURN p, k, collect(DISTINCT toString(v))[..2] AS samples LIMIT 300")
            for row in self._sampled(carried, entity_id):
                if row["samples"]:
                    paths[f"{row['p']} / {row['k']}"] = [str(v) for v in row["samples"]]
            found[entity_id] = paths
        return found

    def _sampled(self, cypher: str, entity_id: str) -> list[dict]:
        try:
            return self.ask(cypher, id=entity_id)
        except KGError:
            return []

    def values_of(self, ids: list[str], path: str) -> dict[str, list[str]]:
        relations = [r.strip() for r in path.split(" / ")]
        value = (f"toString({_property_read('m', relations[1])})" if len(relations) == 2
                 else f"toString(m.{self.id_property})")
        cypher = (f"MATCH ({self._anchor})-[:{_quote_type(relations[0])}]->(m)\n"
                  f"WHERE n.{self.id_property} IN $ids\n"
                  f"RETURN n.{self.id_property} AS e, collect(DISTINCT {value}) AS vs")
        try:
            rows = self.ask(cypher, ids=list(ids))
        except KGError:
            return {}
        return {str(row["e"]): [str(v) for v in row["vs"] or []] for row in rows}

    def opened(self, entity_id: str, relations: list[str],
               failed: set[tuple[str, str]] | None = None) -> LocalSchema:  # noqa: ARG002
        found = self._probe(entity_id)
        return LocalSchema(entity_id=entity_id,
                           entries=[e for e in found.entries if e.predicate in set(relations)],
                           paths=[p for p in found.paths if p.relations[0] in set(relations)])

    def mediator_paths(self, entity_id: str, limit: int = 20) -> list[PathSchema]:
        cypher = (
            f"MATCH {self._me}-[r1]-(m)\n"
            f"WHERE (m:CVT OR m.{self.name_property} IS NULL)\n"
            f"WITH type(r1) AS p1, m ORDER BY m.{self.id_property}\n"
            f"WITH p1, collect(m)[..{MEDIATORS_PER_RELATION}] AS few\n"
            f"UNWIND few AS m\n"
            f"MATCH (m)-[r2]-(x)\n"
            f"WHERE x.{self.id_property} <> $id AND x.{self.name_property} IS NOT NULL\n"
            f"WITH p1, type(r2) AS p2, m, x ORDER BY x.{self.id_property}\n"
            f"WITH p1, p2, {_property_read('m', 'type')} AS mtype, m, x\n"
            f"WITH p1, p2, mtype, count(*) AS n,\n"
            f"     head(collect(x)) AS one, head(collect(m)) AS record\n"
            f"RETURN p1, p2, mtype, n, one.{self.name_property} AS sample_name,\n"
            f"{_ONE_TYPE},\n"
            f"       [k IN coalesce(record.prop_keys, []) WHERE k <> 'type'] AS props\n"
            f"ORDER BY n DESC LIMIT {limit}"
        )
        try:
            rows = self.ask(cypher, id=entity_id, generic=GENERIC_LABELS)
        except KGError:
            return []
        return [PathSchema(relations=[str(row["p1"]), str(row["p2"])], target_type=row.get("target_type"),
                           sample_name=row.get("sample_name"), count=int(row["n"]),
                           mediator_type=(str(row["mtype"]) if row.get("mtype") else None),
                           properties=[str(k) for k in (row.get("props") or [])][:8])
                for row in rows]

    def exists(self, entity_id: str) -> bool:
        return bool(self.ask(f"MATCH {self._me} RETURN 1 AS one LIMIT 1", id=entity_id))

    def label(self, entity_id: str) -> str | None:
        rows = self.ask(f"MATCH {self._me}\nRETURN n.{self.name_property} AS name LIMIT 1", id=entity_id)
        return None if not rows or rows[0]["name"] is None else str(rows[0]["name"])

    def labels(self, ids: list[str]) -> dict[str, str | None]:
        if not ids:
            return {}
        cypher = (f"MATCH ({self._anchor}) WHERE n.{self.id_property} IN $ids\n"
                  f"RETURN n.{self.id_property} AS id, n.{self.name_property} AS name")
        found = {str(row["id"]): str(row["name"]) for row in self.ask(cypher, ids=list(ids))
                 if row["name"] is not None}
        return {i: found.get(i) for i in ids}
