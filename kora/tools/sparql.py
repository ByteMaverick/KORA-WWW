"""Freebase over SPARQL, as the KGQA literature queries it."""

from __future__ import annotations

import functools
import hashlib
import json
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import weakref
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from pathlib import Path
from typing import TypeVar

from kora import ARTIFACT_ROOT
from kora.tools.dialect import (COMPARISONS, comparisons, count_variable, literal_kind, parse_modifiers, parse_query,
                                property_absences, property_reads, set_anchors, type_constraints,
                                without_type_constraints)
from kora.tools.kg import Example, KGError, KGResult, LocalSchema, PathSchema, SchemaEntry

NS = "http://rdf.freebase.com/ns/"
PREFIX = f"PREFIX ns: <{NS}>\nPREFIX xsd: <http://www.w3.org/2001/XMLSchema#>"

NOISE = re.compile(
    r"^(type\.object\.(key|type|name|id)|type\.type\.|rdf-schema|22-rdf|common\.topic\.(article|"
    r"topic_equivalent_webpage|description|image|alias|topical_webpage|social_media_presence|official_website|weblink)|"
    r"common\.document\.|freebase\.|user\.|dataworld\.|base\.schemastaging\.|imdb\.|media_common\.quotation)"
)
KEPT = ("common.topic.notable_types", "common.topic.notable_for", "kg.object_profile.prominent_type")
NOT_NS = re.compile(r"^(https?://|22-rdf-syntax|rdf-schema)")

RETRIES = 3
RETRY_SECONDS = 0.5
QUERY_SECONDS = 45.0
TRANSIENT = frozenset({500, 502, 503, 504})

HUB_DEGREE = 50_000
EXAMPLES = 6
DETAILED = 24
ROUTES = 24
FIRST_HOPS = 24
ROUTE_SECONDS = 20.0
ROUTE_QUERY_SECONDS = 25
DATED = ("date", "year", "from", "to", "when", "start", "end", "time")
BEGAN = ("from", "start")
VALUES_LIMIT = 5000
SLOW_HOP_SECONDS = 0.9 * ROUTE_QUERY_SECONDS
PROBE_WORKERS = 8
CLAIM_SECONDS = QUERY_SECONDS + 5
VOCABULARY_ROOT = ARTIFACT_ROOT / "vocabulary"
PAGE_ROOT = ARTIFACT_ROOT / "pages"


def probe_workers() -> int:
    """How many probes to run at once: `KORA_PROBE_WORKERS`, or `PROBE_WORKERS`."""
    try:
        return max(1, int(os.environ.get("KORA_PROBE_WORKERS", PROBE_WORKERS)))
    except ValueError:
        return PROBE_WORKERS


T = TypeVar("T")
_Claim = tuple[tuple[str, str, str], threading.Event, SchemaEntry]


def at_once(jobs: list[Callable[[], T]], pool: ThreadPoolExecutor | None) -> list[T]:
    """Run every job, and hand back their results **in the order given**, not in the order they finished."""
    if pool is None or len(jobs) <= 1:
        return [job() for job in jobs]
    futures = [pool.submit(copy_context().run, job) for job in jobs]
    results: list[T] = []
    failure: Exception | None = None
    for future in futures:
        try:
            results.append(future.result())
        except Exception as exc:
            failure = failure or exc
    if failure is not None:
        raise failure
    return results


def timed_out(exc: BaseException) -> bool:
    """Whether this failure is the request's own clock running out, rather than a socket that went away."""
    return isinstance(exc, TimeoutError) or isinstance(getattr(exc, "reason", None), TimeoutError)


def strip_ns(value: str) -> str:
    """`http://rdf.freebase.com/ns/m.0345h` -> `m.0345h`."""
    return value[len(NS):] if value.startswith(NS) else value


def interesting(predicate: str) -> bool:
    return predicate in KEPT or not (NOT_NS.match(predicate) or NOISE.match(predicate))


_MID_SHAPED = re.compile(r"^(m|g)\.[0-9a-z_]+$")
_DATE_LITERAL = re.compile(r"^(?P<year>\d{4})(-(?P<month>\d{2})(-(?P<day>\d{2}))?)?(?P<time>T[\d:]+)?$")


def _term(token: str) -> str:
    if token.startswith("?"):
        return token
    if token.startswith(("http://", "https://")):
        return f"<{token}>"
    return f"ns:{token}"


def _set_member(member: str) -> str:
    return f"ns:{member}" if _MID_SHAPED.match(member) else f'"{_escaped(member)}"'


def _entity_term(token: str) -> str:
    return _term(token) if token.startswith(("?", "http://", "https://")) else _set_member(token)


def _date_cast(literal: str) -> str | None:
    match = _DATE_LITERAL.match(literal)
    if not match:
        return None
    return ("xsd:dateTime" if match.group("time") else "xsd:date" if match.group("day")
            else "xsd:gYearMonth" if match.group("month") else "xsd:gYear")


_EMPTY_WORDS = frozenset({
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "and", "or", "is", "was", "were", "did", "does", "do",
    "what", "which", "who", "whom", "where", "when", "how", "that", "this", "with", "from", "by", "all", "identify",
    "find", "determine", "common", "topic", "type", "object", "base", "user",
})
_WORD = re.compile(r"[a-z0-9]+")


def words_of(text: str) -> set[str]:
    """The words worth matching on, from a question or a predicate name."""
    return {w for w in _WORD.findall((text or "").lower()) if len(w) >= 3 and w not in _EMPTY_WORDS}


def rank_for(entries: list[SchemaEntry], wanted: str) -> list[SchemaEntry]:
    """The order to spend the describing budget in."""
    asked = words_of(wanted)
    if not asked:
        return list(entries)
    return sorted(entries, key=lambda entry: (-len(words_of(entry.predicate) & asked), -entry.count))


def named_first(entries: list[SchemaEntry], first: list[str] | None, wanted: str) -> list[SchemaEntry]:
    """`first` in the order given, then everything else by `rank_for`."""
    rank = {name: i for i, name in enumerate(first or [])}
    remembered = sorted((e for e in entries if e.predicate in rank), key=lambda e: rank[e.predicate])
    return remembered + rank_for([e for e in entries if e.predicate not in rank], wanted)


def to_sparql(query: str, limit: int = 500, sets: dict[str, list[str]] | None = None) -> str | None:
    """Kora's compact triple dialect as a SPARQL SELECT, or None."""
    parsed = parse_query(query)
    if parsed is None:
        return None
    variables, all_patterns = parsed
    patterns = without_type_constraints(all_patterns)
    if not variables or not patterns:
        return None

    lines = [f"{_entity_term(s)} {_term(p)} {_entity_term(o)} ." for s, p, o in patterns]
    for subject, key, variable in property_reads(all_patterns):
        prop = f"?p_{re.sub(r'[^A-Za-z0-9_]', '_', variable[1:])}"
        lines.append(f"{_entity_term(subject)} {prop} {variable} .")
        lines.append(f'FILTER(STRENDS(STR({prop}), ".{_escaped(key)}"))')
    for subject, key in property_absences(all_patterns):
        safe_key = re.sub(r'[^A-Za-z0-9_]', '_', key)
        safe_subject = re.sub(r'[^A-Za-z0-9_]', '_', subject.lstrip("?"))
        prop = f"?p_{safe_key}_{safe_subject}"
        placeholder = f"?_{safe_key}_{safe_subject}"
        lines.append(f'FILTER NOT EXISTS {{ {_entity_term(subject)} {prop} {placeholder} . '
                     f'FILTER(STRENDS(STR({prop}), ".{_escaped(key)}")) }}')
    for term, kinds in type_constraints(all_patterns).items():
        lines.extend(f"{_entity_term(term)} ns:type.object.type ns:{kind} ." for kind in kinds)
    for variable, op, literal in comparisons(all_patterns):
        symbol = COMPARISONS[op]
        cast = _date_cast(literal)
        if literal_kind(literal) == "number":
            lines.append(f"FILTER(xsd:double({variable}) {symbol} {literal})")
        elif cast:
            lines.append(f'FILTER({cast}({variable}) {symbol} "{_escaped(literal)}"^^{cast})')
        else:
            lines.append(f'FILTER(STR({variable}) {symbol} "{_escaped(literal)}")')
    for variable, name in set_anchors(all_patterns):
        members = (sets or {}).get(name)
        if members is None:
            return None
        lines.insert(0, f"VALUES {variable} {{ {' '.join(_set_member(m) for m in members[:2000])} }}")
    mods = parse_modifiers(query)
    tail = ""
    if mods["order"]:
        tail += f" ORDER BY {mods['order'][1]}({mods['order'][0]})"
    tail += f" LIMIT {mods['limit'] or limit}"
    body = "\n  ".join(lines)
    counted = count_variable(query)
    select = f"(COUNT(DISTINCT {counted}) AS {counted})" if counted else "DISTINCT " + " ".join(variables)
    return f"{PREFIX}\nSELECT {select} WHERE {{\n  {body}\n}}{tail}"


def _escaped(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


def _mid(uri: str) -> str:
    return uri.rsplit("/", 1)[-1] if uri.startswith("http") else uri


class FreebaseSPARQL:
    """A `KGProvider` over a Virtuoso SPARQL endpoint holding Freebase."""

    backend = "freebase-sparql"
    id_property = "mid"
    anchor_label = ""
    hub_relations: frozenset[str] = frozenset()
    name_property = "name"
    LABEL_BATCH = 100

    def __init__(self, endpoint: str, timeout: float = QUERY_SECONDS, limit: int = 500,
                 hub_degree: int = HUB_DEGREE):
        self.endpoint = endpoint
        self.timeout = timeout
        self.limit = limit
        self.hub_degree = hub_degree
        self.queries: list[str] = []
        self.calls = 0
        self._labels: dict[str, str | None] = {}
        self._hubs: dict[str, bool] = {}
        self._entries: dict[str, list[SchemaEntry]] = {}
        self._paths: dict[str, list[PathSchema]] = {}
        self._expanded: set[tuple[str, str]] = set()
        self._expanding: dict[tuple[str, str], threading.Event] = {}
        self._slow_hops: set[tuple[str, str]] = set()
        self._record_valued: dict[str, list[str]] = {}
        self._recording: dict[str, threading.Event] = {}
        self._described: set[tuple[str, str, str]] = set()
        self._describing: dict[tuple[str, str, str], threading.Event] = {}
        self._vocabulary: set[str] | None = None
        self._counting: dict[str, threading.Event] = {}
        self._lock = threading.Lock()
        self._probes: ThreadPoolExecutor | None = None
        self._values: dict[tuple[str, str], list[str]] = {}

    def ask(self, sparql: str, timeout: float | None = None) -> list[dict]:
        data = urllib.parse.urlencode({"query": sparql, "format": "application/sparql-results+json"}).encode()
        request = urllib.request.Request(self.endpoint, data=data, headers={
            "Accept": "application/sparql-results+json", "Content-Type": "application/x-www-form-urlencoded"})
        for attempt in range(RETRIES):
            with self._lock:
                self.calls += 1
            try:
                with urllib.request.urlopen(request, timeout=timeout or self.timeout) as response:
                    payload = json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                if exc.code not in TRANSIENT:
                    return []
                if attempt == RETRIES - 1:
                    raise KGError(f"cannot reach {self.endpoint}: HTTP {exc.code} {exc.reason}") from None
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                if timed_out(exc) or attempt == RETRIES - 1:
                    raise KGError(f"cannot reach {self.endpoint}: {exc}") from None
            else:
                return payload.get("results", {}).get("bindings", [])
            time.sleep(RETRY_SECONDS * (attempt + 1))
        return []

    @staticmethod
    def value(binding: dict, name: str) -> str | None:
        found = binding.get(name)
        return strip_ns(found["value"]) if found else None

    def health(self) -> bool:
        return bool(self.ask("SELECT ?s WHERE { ?s ?p ?o } LIMIT 1", timeout=10))

    def execute(self, query: str, sets: dict[str, list[str]] | None = None) -> KGResult:
        started = time.time()
        self.queries.append(query)
        result = KGResult(query=query)
        sparql = to_sparql(query, self.limit, sets=sets)
        if sparql is None:
            parsed = parse_query(query)
            unknown_set = parsed is not None and any(name not in (sets or {}) for _, name in set_anchors(parsed[1]))
            result.error = ("a set anchor names a set this task does not hold" if unknown_set else
                            "the query selects nothing or matches nothing; a set anchor must name a bound set")
            return result
        try:
            bindings = self.ask(sparql)
        except KGError as exc:
            result.error = str(exc)
        else:
            variables = parse_query(query)[0]
            result.rows = [{v: self.value(b, v[1:]) or "" for v in variables if v[1:] in b} for b in bindings]
        result.latency_seconds = time.time() - started
        return result

    def _pool(self) -> ThreadPoolExecutor | None:
        if probe_workers() <= 1:
            return None
        with self._lock:
            if self._probes is None:
                self._probes = ThreadPoolExecutor(max_workers=probe_workers(), thread_name_prefix="kora-probe")
                weakref.finalize(self, self._probes.shutdown, wait=False)
            return self._probes

    def _release(self, key: object, inflight: dict, event: threading.Event) -> None:
        with self._lock:
            if inflight.get(key) is event:
                del inflight[key]
        event.set()

    @staticmethod
    def _await(events: list[threading.Event], deadline: float) -> None:
        for event in events:
            remaining = deadline - time.time()
            if remaining <= 0:
                return
            event.wait(remaining)

    def _once(self, entity_id: str, cached: dict[str, T], inflight: dict[str, threading.Event],
              probe: Callable[[], T]) -> T:
        while True:
            with self._lock:
                found = cached.get(entity_id)
                if found is not None:
                    return found
                waiting = inflight.get(entity_id)
                if waiting is None:
                    mine = inflight[entity_id] = threading.Event()
                    break
            self._await([waiting], time.time() + CLAIM_SECONDS)
        try:
            value = probe()
            with self._lock:
                cached[entity_id] = value
        finally:
            self._release(entity_id, inflight, mine)
        return value

    def local_schema(self, entity_id: str, limit: int = 20, wanted: str = "", first: list[str] | None = None,
                     failed_hops: set[tuple[str, str]] | None = None) -> LocalSchema:
        entries = self._counted(entity_id)
        self.describe(entity_id, named_first(entries, first, wanted))
        candidates = self.record_valued(entity_id)
        by_name = {e.predicate: e for e in entries}
        paths = self.expand_routes(entity_id, [
            entry.predicate for entry in
            rank_for([by_name.get(p) or SchemaEntry(predicate=p, direction="out") for p in candidates], wanted)
        ], failed_hops=failed_hops)
        return LocalSchema(entity_id=entity_id, entries=entries, paths=paths)

    def properties(self, entity_id: str) -> list[SchemaEntry]:
        return [e.model_copy() for e in self._counted(entity_id)]

    def opened(self, entity_id: str, relations: list[str],
               failed: set[tuple[str, str]] | None = None) -> LocalSchema:
        entries = [e for e in self._counted(entity_id) if e.predicate in set(relations)]
        try:
            self.describe(entity_id, entries, budget=len(entries))
        except Exception:
            pass
        keep = set(self.record_valued(entity_id))
        through = [p for p in dict.fromkeys(relations) if p in keep]
        paths = self.expand_routes(entity_id, through, budget=len(through), failed_hops=failed) if through else []
        return LocalSchema(entity_id=entity_id, entries=entries,
                           paths=[p for hop in through for p in paths if p.relations[0] == hop])

    def _record_predicates(self, entity_id: str, limit: int) -> list[str]:
        """The interesting relations from the entity to unnamed nodes (records), busiest first."""
        rows = self.ask(
            f"{PREFIX}\nSELECT ?p (COUNT(*) AS ?n) WHERE {{\n"
            f"  ns:{entity_id} ?p ?mid . FILTER(isIRI(?mid))\n"
            f"  FILTER NOT EXISTS {{ ?mid ns:type.object.name ?nm }}\n"
            f"}} GROUP BY ?p ORDER BY DESC(?n) LIMIT {limit}"
        )
        return [p for p in (self.value(row, "p") or "" for row in rows) if interesting(p)]

    def record_valued(self, entity_id: str) -> list[str]:
        return self._once(entity_id, self._record_valued, self._recording,
                          lambda: list(dict.fromkeys(self._record_predicates(entity_id, 100))))

    def expand_routes(self, entity_id: str, predicates: list[str], budget: int = FIRST_HOPS,
                      seconds: float = ROUTE_SECONDS,
                      failed_hops: set[tuple[str, str]] | None = None) -> list[PathSchema]:
        with self._lock:
            paths = self._paths.setdefault(entity_id, [])
        workers = probe_workers()
        deadline = time.time() + seconds
        failure: Exception | None = None
        spent, at = 0, 0
        while at < len(predicates) and spent < budget and time.time() < deadline:
            wave: list[tuple[str, threading.Event]] = []
            waits: list[threading.Event] = []
            try:
                while (at < len(predicates) and len(wave) + len(waits) < workers
                       and spent + len(wave) + len(waits) < budget):
                    predicate = predicates[at]
                    at += 1
                    key = (entity_id, predicate)
                    if (failed_hops is not None and key in failed_hops) or key in self._slow_hops:
                        continue
                    with self._lock:
                        if key in self._expanded:
                            continue
                        opening = self._expanding.get(key)
                        if opening is not None:
                            waits.append(opening)
                            continue
                        opening = self._expanding[key] = threading.Event()
                    wave.append((predicate, opening))
                if not wave and not waits:
                    continue
                spent += len(wave) + len(waits)
                opened = at_once([functools.partial(self._open_hop, entity_id, p) for p, _ in wave], self._pool())
                survived = any(error is None for _, error in opened)
                for (predicate, _), (found, error) in zip(wave, opened, strict=True):
                    if not survived:
                        failure = failure or error
                    elif error is not None:
                        if failed_hops is not None:
                            failed_hops.add((entity_id, predicate))
                        if getattr(error, "seconds", 0.0) >= SLOW_HOP_SECONDS:
                            with self._lock:
                                self._slow_hops.add((entity_id, predicate))
                    if error is None:
                        with self._lock:
                            self._expanded.add((entity_id, predicate))
                    if not found:
                        continue
                    with self._lock:
                        known = {tuple(p.relations) for p in paths}
                        paths.extend(p for p in found if tuple(p.relations) not in known)
            finally:
                for predicate, claim in wave:
                    self._release((entity_id, predicate), self._expanding, claim)
            self._await(waits, min(deadline, time.time() + CLAIM_SECONDS))
        if failure is not None:
            raise failure
        with self._lock:
            return list(paths)

    def _open_hop(self, entity_id: str, predicate: str) -> tuple[list[PathSchema], Exception | None]:
        started = time.time()
        try:
            return self._routes_through(entity_id, predicate), None
        except Exception as exc:
            exc.seconds = time.time() - started      # how long it ran: `expand_routes` tells slow from broken by it
            return [], exc

    def describe(self, entity_id: str, entries: list[SchemaEntry], budget: int = DETAILED) -> None:
        mine, theirs = self._claim_descriptions(entity_id, entries, budget)
        failure = self._describe_claimed(entity_id, mine)
        self._await([event for _, event, _ in theirs], time.time() + CLAIM_SECONDS)
        if theirs:
            orphans, opened = self._claim_descriptions(entity_id, [entry for _, _, entry in theirs], len(theirs))
            failure = failure or self._describe_claimed(entity_id, orphans)
            self._await([event for _, event, _ in opened], time.time() + CLAIM_SECONDS)
        if mine:
            self._save_page(entity_id)
        if failure is not None:
            raise failure

    def _claim_descriptions(self, entity_id: str, entries: list[SchemaEntry],
                            budget: int) -> tuple[list[_Claim], list[_Claim]]:
        mine: list[_Claim] = []
        theirs: list[_Claim] = []
        with self._lock:
            for entry in entries:
                if len(mine) + len(theirs) >= budget:
                    break
                key = (entity_id, entry.direction, entry.predicate)
                if key in self._described:
                    continue
                describing = self._describing.get(key)
                if describing is not None:
                    theirs.append((key, describing, entry))
                    continue
                event = self._describing[key] = threading.Event()
                mine.append((key, event, entry))
        return mine, theirs

    def _describe_claimed(self, entity_id: str, claimed: list[_Claim]) -> Exception | None:
        if not claimed:
            return None

        def describe_one(key: tuple[str, str, str], entry: SchemaEntry) -> Exception | None:
            try:
                self._describe(entity_id, entry)
            except Exception as exc:
                return exc
            with self._lock:
                self._described.add(key)
            return None

        try:
            failures = at_once([functools.partial(describe_one, key, entry) for key, _, entry in claimed],
                               self._pool())
        finally:
            for key, event, _ in claimed:
                self._release(key, self._describing, event)
        return next((failed for failed in failures if failed is not None), None)

    def _counted(self, entity_id: str) -> list[SchemaEntry]:
        return self._once(entity_id, self._entries, self._counting,
                          lambda: self._load_page(entity_id) or self._count(entity_id))

    @property
    def _endpoint_key(self) -> str:
        return hashlib.sha1(self.endpoint.encode()).hexdigest()[:12]

    def _page_path(self, entity_id: str) -> Path:
        return PAGE_ROOT / self._endpoint_key / f"{entity_id.replace('/', '_')}.json"

    def _load_page(self, entity_id: str) -> list[SchemaEntry] | None:
        try:
            saved = json.loads(self._page_path(entity_id).read_text())
            entries = [SchemaEntry.model_validate(e) for e in saved["entries"]]
            described = {(str(d), str(p)) for d, p in saved.get("described", [])}
        except (OSError, KeyError, TypeError, ValueError):
            return None
        if not entries:
            return None
        with self._lock:
            for entry in entries:
                if (entry.direction, entry.predicate) in described:
                    self._described.add((entity_id, entry.direction, entry.predicate))
        return entries

    def _save_page(self, entity_id: str) -> None:
        with self._lock:
            entries = list(self._entries.get(entity_id) or [])
            described = [[direction, predicate] for entity, direction, predicate in self._described
                         if entity == entity_id]
        if not entries:
            return
        path = self._page_path(entity_id)
        temporary = path.with_suffix(f".{os.getpid()}.tmp")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_text(json.dumps({"entries": [entry.model_dump() for entry in entries],
                                             "described": described}))
            os.replace(temporary, path)
        except OSError:
            temporary.unlink(missing_ok=True)

    def _count(self, entity_id: str) -> list[SchemaEntry]:
        directions = (("out", f"ns:{entity_id} ?p ?o"), ("in", f"?o ?p ns:{entity_id}"))
        queries = [f"{PREFIX}\nSELECT ?p (COUNT(*) AS ?n) WHERE {{ {pattern} }}\n"
                   f"GROUP BY ?p ORDER BY DESC(?n) LIMIT 300" for _, pattern in directions]
        answers = at_once([functools.partial(self.ask, query) for query in queries], self._pool())
        return [SchemaEntry(predicate=predicate, direction=direction, count=int(row["n"]["value"]))
                for (direction, _), rows in zip(directions, answers, strict=True) for row in rows
                if interesting(predicate := self.value(row, "p") or "")]

    def _describe(self, entity_id: str, entry: SchemaEntry) -> None:
        pattern = (f"ns:{entity_id} ns:{entry.predicate} ?o" if entry.direction == "out"
                   else f"?o ns:{entry.predicate} ns:{entity_id}")
        rows = self.ask(
            f"{PREFIX}\nSELECT ?o (SAMPLE(?nm) AS ?name) (SAMPLE(?ty) AS ?kind) WHERE {{\n"
            f"  {pattern} .\n"
            f'  OPTIONAL {{ ?o ns:type.object.name ?nm FILTER(lang(?nm) = "en") }}\n'
            f"  OPTIONAL {{ ?o ns:kg.object_profile.prominent_type ?ty }}\n"
            f"}} GROUP BY ?o LIMIT {EXAMPLES}", timeout=20,
        )
        named, kinds = [], []
        for row in rows:
            target = self.value(row, "o")
            if target is None:
                continue
            name, kind = self.value(row, "name"), self.value(row, "kind")
            if name:
                named.append(Example(id=target, name=name, kind=kind))
            if kind and kind not in kinds:
                kinds.append(kind)
        entry.examples = named
        entry.target_types = kinds
        entry.target_type = kinds[0] if kinds else None
        entry.sample_object = named[0].id if named else (self.value(rows[0], "o") if rows else None)
        entry.sample_name = named[0].name if named else None
        entry.mediator = bool(rows) and not named

    def mediator_paths(self, entity_id: str, limit: int = 20) -> list[PathSchema]:
        paths: list[PathSchema] = []
        for predicate in self._record_predicates(entity_id, 12):
            paths.extend(self._routes_through(entity_id, predicate))
            if len(paths) >= limit:
                break
        return paths[:limit]

    def _routes_through(self, entity_id: str, predicate: str) -> list[PathSchema]:
        rows = self.ask(
            f"{PREFIX}\nSELECT ?p2 (SAMPLE(?name) AS ?eg) (SAMPLE(?ty) AS ?kind) "
            f"(SAMPLE(?mt) AS ?mtype) (COUNT(*) AS ?n) WHERE {{\n"
            f"  ns:{entity_id} ns:{predicate} ?mid . ?mid ?p2 ?far .\n"
            f'  ?far ns:type.object.name ?name FILTER(lang(?name) = "en")\n'
            f"  OPTIONAL {{ ?far ns:kg.object_profile.prominent_type ?ty }}\n"
            f"  OPTIONAL {{ ?mid ns:type.object.type ?mt }}\n"
            f"}} GROUP BY ?p2 ORDER BY DESC(?n) LIMIT 6", timeout=ROUTE_QUERY_SECONDS,
        )
        properties = self._record_properties(entity_id, predicate) if rows else []
        return [PathSchema(relations=[predicate, second], directions=["out", "out"],
                           target_type=self.value(row, "kind"), sample_name=self.value(row, "eg"),
                           count=int(row["n"]["value"]), mediator_type=self.value(row, "mtype"),
                           properties=properties)
                for row in rows if interesting(second := self.value(row, "p2") or "")]

    def _record_properties(self, entity_id: str, predicate: str) -> list[str]:
        rows = self.ask(
            f"{PREFIX}\nSELECT DISTINCT ?p WHERE {{\n"
            f"  ns:{entity_id} ns:{predicate} ?mid . ?mid ?p ?v . FILTER(isLiteral(?v))\n"
            f"}} LIMIT 12", timeout=20,
        )
        return list(dict.fromkeys(p for p in (self.value(row, "p") or "" for row in rows) if p and interesting(p)))[:8]

    def _memo(self, cache: dict[str, T], key: str, probe: Callable[[], T]) -> T:
        with self._lock:
            if key in cache:
                return cache[key]
        found = probe()
        with self._lock:
            cache[key] = found
        return found

    def label(self, entity_id: str) -> str | None:
        def ask() -> str | None:
            rows = self.ask(f'{PREFIX}\nSELECT ?n WHERE {{ ns:{entity_id} ns:type.object.name ?n . '
                            f'FILTER(lang(?n) = "en") }} LIMIT 1', timeout=15)
            return self.value(rows[0], "n") if rows else None
        return self._memo(self._labels, entity_id, ask)

    def labels(self, ids: list[str]) -> dict[str, str | None]:
        with self._lock:
            names = {i: self._labels[i] for i in ids if self._labels.get(i)}
        wanted = [i for i in dict.fromkeys(ids) if i not in names and re.fullmatch(r"[mg]\.[0-9a-z_]+", i)]
        for start in range(0, len(wanted), self.LABEL_BATCH):
            batch = wanted[start:start + self.LABEL_BATCH]
            try:
                rows = self.ask(
                    f'{PREFIX}\nSELECT ?e ?n WHERE {{ VALUES ?e {{ {" ".join("ns:" + i for i in batch)} }} '
                    f'?e ns:type.object.name ?n . FILTER(lang(?n) = "en") }}', timeout=30,
                )
            except Exception:  # noqa: BLE001 -- a name is a courtesy; the ids still stand
                rows = []
            found: dict[str, str] = {}
            for row in rows:
                entity, name = self.value(row, "e"), self.value(row, "n")
                if entity and name and entity not in found:
                    found[entity] = name
            with self._lock:
                self._labels.update(found)
            names.update(found)
        return {i: names.get(i) for i in ids}

    def resolve_name(self, name: str, limit: int = 5) -> list[tuple[str, str]]:
        wanted = (name or "").strip()
        for variant in dict.fromkeys([wanted, wanted.title(), wanted.capitalize()] if wanted else []):
            rows = self.ask(
                f'{PREFIX}\nSELECT ?x (COUNT(?p) AS ?n) WHERE {{ '
                f'?x ns:type.object.name "{_escaped(variant)}"@en . ?x ?p ?o . }} '
                f'GROUP BY ?x ORDER BY DESC(?n) LIMIT {max(1, limit)}',
                timeout=30,
            )
            found = [(mid, variant) for mid in (_mid(self.value(row, "x") or "") for row in rows) if mid]
            if found:
                return found
        return []

    def exists(self, entity_id: str) -> bool:
        return bool(self.ask(f"{PREFIX}\nSELECT ?p WHERE {{ ns:{entity_id} ?p ?o }} LIMIT 1", timeout=15))

    def is_hub(self, entity_id: str, cap: int | None = None) -> bool:
        def ask() -> bool:
            # A plain count: cheap even on a hub, where a capped subquery is not honoured by the planner.
            rows = self.ask(f"{PREFIX}\nSELECT (COUNT(*) AS ?n) WHERE {{\n"
                            f"  {{ ns:{entity_id} ?p ?o }} UNION {{ ?s ?q ns:{entity_id} }}\n"
                            f"}}", timeout=30)
            return bool(rows) and int(rows[0]["n"]["value"]) > (cap or self.hub_degree)
        return self._memo(self._hubs, entity_id, ask)

    def value_of(self, entity_id: str, prop: str) -> list[str]:
        if (entity_id, prop) not in self._values:
            try:
                self._values[(entity_id, prop)] = self._value_of(entity_id, prop)
            except KGError:
                return []
        return self._values[(entity_id, prop)]

    def _value_of(self, entity_id: str, prop: str) -> list[str]:
        me = f"ns:{entity_id}"
        own = self._read(f"{me} ns:{prop} ?v . FILTER(isLiteral(?v) || EXISTS {{ ?v ns:type.object.name ?n }})")
        if own:
            return own
        carried = self._carried(self._pairs(f"{me} ns:{prop} ?m . ?m ?q ?v . FILTER(isLiteral(?v))"), prop)
        if carried:
            return carried
        for path in self._paths.get(entity_id, []):
            if prop in (path.relations[-1], *path.properties):
                found = self._read(f"{me} ns:{path.relations[0]} ?m . ?m ns:{prop} ?v")
                if found:
                    return found
        tail = prop.rsplit(".", 1)[-1]
        return self._read(f'{me} ?q ?v . FILTER(isLiteral(?v) && STRENDS(STR(?q), "{tail}"))')

    def properties_with_samples(self, ids: list[str]) -> dict[str, dict[str, list[str]]]:
        found: dict[str, dict[str, list[str]]] = {}
        for entity_id in ids:
            me = f"ns:{entity_id}"
            paths: dict[str, list[str]] = {}
            for row in self._sampled(f"{me} ?p ?v . FILTER(isLiteral(?v))", "?p"):
                relation = self.value(row, "p") or ""
                if interesting(relation):
                    paths[relation] = self._samples(row)
            carried = (f"{me} ?p ?m . FILTER(isIRI(?m)) FILTER NOT EXISTS {{ ?m ns:type.object.name ?nm }}\n"
                       f"  ?m ?q ?v . FILTER(isLiteral(?v))")
            for row in self._sampled(carried, "?p ?q"):
                relation, field = self.value(row, "p") or "", self.value(row, "q") or ""
                if interesting(relation) and interesting(field):
                    paths[f"{relation} / {field}"] = self._samples(row)
            found[entity_id] = paths
        return found

    def _sampled(self, pattern: str, keys: str) -> list[dict]:
        try:
            return self.ask(f"{PREFIX}\nSELECT {keys} (MIN(STR(?v)) AS ?a) (MAX(STR(?v)) AS ?b) WHERE {{\n"
                            f"  {pattern}\n}} GROUP BY {keys} LIMIT 300", timeout=ROUTE_QUERY_SECONDS)
        except KGError:
            return []

    def _samples(self, row: dict) -> list[str]:
        return list(dict.fromkeys(v for v in (self.value(row, "a"), self.value(row, "b")) if v))

    def values_of(self, ids: list[str], path: str) -> dict[str, list[str]]:
        relations = [r.strip() for r in path.split(" / ")]
        pattern = (f"?e ns:{relations[0]} ?m . ?m ns:{relations[1]} ?v" if len(relations) == 2
                   else f"?e ns:{relations[0]} ?v")
        members = " ".join(f"ns:{i}" for i in ids)
        try:
            rows = self.ask(f"{PREFIX}\nSELECT ?e ?v WHERE {{ VALUES ?e {{ {members} }} {pattern} .\n"
                            f"  FILTER(isLiteral(?v) && (lang(?v) = '' || langMatches(lang(?v), 'en'))) }} "
                            f"LIMIT {VALUES_LIMIT}")
        except KGError:
            return {}
        found: dict[str, list[str]] = {}
        for row in rows:
            entity, value = self.value(row, "e"), self.value(row, "v")
            if entity and value and value not in found.setdefault(entity, []):
                found[entity].append(value)
        return found

    def _read(self, pattern: str) -> list[str]:
        rows = self.ask(f"{PREFIX}\nSELECT DISTINCT ?v WHERE {{ {pattern} }} LIMIT {self.limit}")
        return list(dict.fromkeys(v for row in rows if (v := self.value(row, "v"))))

    def _pairs(self, pattern: str) -> list[tuple[str, str]]:
        rows = self.ask(f"{PREFIX}\nSELECT ?q ?v WHERE {{ {pattern} }} LIMIT {self.limit}")
        return [(self.value(row, "q") or "", self.value(row, "v") or "") for row in rows]

    @staticmethod
    def _carried(pairs: list[tuple[str, str]], prop: str) -> list[str]:
        def dated(name: str) -> bool:
            return any(word in name.rsplit(".", 1)[-1].lower() for word in DATED)

        dates = [v for q, v in pairs if dated(q)]
        began = [v for q, v in pairs if dated(q) and any(w in q.rsplit(".", 1)[-1] for w in BEGAN)]
        numbers = [] if dated(prop) else [v for q, v in pairs if not dated(q) and re.fullmatch(r"-?\d+(\.\d+)?", v)]
        return list(dict.fromkeys(numbers or began or dates))

    def known_types(self) -> set[str]:
        return set()

    def relation_types(self) -> list[str]:
        return []

    def vocabulary(self) -> set[str]:
        if self._vocabulary is not None:
            return self._vocabulary
        path = VOCABULARY_ROOT / f"{self._endpoint_key}.predicates.json"
        if path.exists():
            try:
                self._vocabulary = set(json.loads(path.read_text()))
                return self._vocabulary
            except (OSError, json.JSONDecodeError):
                pass  # rebuilt below, as if the cache were empty
        rows = self.ask("SELECT DISTINCT ?p WHERE { ?s ?p ?o }", timeout=60)
        found = {p for p in (self.value(r, "p") or "" for r in rows) if p and interesting(p)}
        if not found:
            return found  # a refusal or a genuinely empty graph: retry next time
        self._vocabulary = found
        try:
            VOCABULARY_ROOT.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(sorted(found)))
        except OSError:
            pass
        return found

    def schema(self) -> list[dict]:
        return []

    def schema_notes(self) -> list[str]:
        return []
