"""A graph provider for the BioKG layout served by Virtuoso: BioCQ's graph, and KQA Pro's for the sampler."""

from __future__ import annotations

import json
import re
import threading
import time
import urllib.parse
import urllib.request

from kora import ARTIFACT_ROOT
from kora.tools.kg import KGError, KGResult, LocalSchema, SchemaEntry, load_schema, load_schema_notes

BASE = "http://skill-lab-research.org"
RDFS_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"
RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
#: `OptimusKG:CHEMBL1079604`, `PrimeKG:22243`, `KQAPro:13213` -- the form the benchmark writes.
ENTITY = re.compile(r"^(?:[A-Za-z]+KG|KQAPro):[A-Za-z0-9_.:\-]+$")
#: Predicates that say nothing about what an entity is.
NOISE = ("/schema/properties", "/schema/subject", "/schema/predicate", "/schema/object")
#: The sub-graphs whose relations are listed. AfriMedKG was missing: its 130 val questions could not name a relation
#: (2026-10-03), so every query on it failed to translate.
GRAPHS = ("optimuskg", "primekg", "afrimedkg")
#: AfriMedKG's relations with fewer edges than this are extraction noise, not vocabulary: 53 of them, named `'`,
#: `a`, `d`, `maculopapular`, with 1 to 18 edges each, beside its 10 real relations (88 to 5,342 edges). The curated
#: graphs keep every relation (OptimusKG has a real one with 13 edges).
MIN_EDGES = {"afrimedkg": 20}


def uri_of(entity_id: str) -> str:
    """`OptimusKG:CHEMBL123` -> its URI. A full URI passes through."""
    if entity_id.startswith("http"):
        return entity_id
    graph = entity_id.split(":", 1)[0].lower()
    return f"{BASE}/{graph}/node/{entity_id}"


def id_of(uri: str) -> str:
    """The short id of a node URI, or the value itself when it is not one."""
    return uri.rsplit("/node/", 1)[-1] if "/node/" in uri else uri


def angle(uri: str) -> str:
    """A URI as a SPARQL term, or a pattern that matches it."""
    return "<" + uri + ">"


def spaced(uri: str) -> bool:
    return " " in uri


def kind_of(type_uri: str) -> str:
    """`.../optimuskg/type/gene/protein` -> `gene/protein`."""
    return type_uri.rsplit("/type/", 1)[-1] if "/type/" in type_uri else type_uri.rsplit("/", 1)[-1]


def token_of(predicate_uri: str) -> str:
    """`.../primekg/relation/expression present` -> `primekg.expression_present`."""
    graph, _, name = predicate_uri.partition("/relation/")
    if not name:
        return predicate_uri
    return f"{graph.rsplit('/', 1)[-1]}.{re.sub(r'[^A-Za-z0-9]+', '_', name)}"


class BioKG:
    """A `KGProvider` over a Virtuoso endpoint in the BioKG layout; KQA Pro's graph with `graphs=("kqapro",)`."""

    backend = "biokg"
    id_property = "id"
    anchor_label = ""
    hub_relations: frozenset[str] = frozenset()
    name_property = "name"
    prominence_expr = ""

    #: An entity with more neighbours than this is a sink, not a place to stand.
    HUB = 5000

    def __init__(self, endpoint: str = "http://localhost:8890/sparql", timeout: float = 45.0,
                 graphs: tuple[str, ...] = GRAPHS, schema: str = "biocq"):
        self.endpoint = endpoint
        self.timeout = timeout
        #: The sub-graphs whose relations are the vocabulary, and the curated schema file the page reads.
        self.graphs, self.schema_name = tuple(graphs), schema
        self.calls = 0
        self._lock = threading.Lock()
        self._labels: dict[str, str | None] = {}
        self._predicates: dict[str, str] = {}      # token -> URI
        self._schemas: dict[str, LocalSchema] = {}
        self._kinds: dict[str, str | None] = {}

    # -- the endpoint ---------------------------------------------------------

    def ask(self, sparql: str, timeout: float | None = None) -> list[dict]:
        """One SPARQL query, as bindings. Raises `KGError` on anything else."""
        with self._lock:
            self.calls += 1
        # The `format` parameter alone. Sending it ALSO as an `Accept` header makes this Virtuoso
        # answer 406, which reads as an unreachable endpoint rather than a header it dislikes.
        body = urllib.parse.urlencode({"query": sparql, "format": "application/sparql-results+json"})
        request = urllib.request.Request(f"{self.endpoint}?{body}")
        try:
            with urllib.request.urlopen(request, timeout=timeout or self.timeout) as answer:
                return json.loads(answer.read()).get("results", {}).get("bindings", [])
        except Exception as exc:  # noqa: BLE001
            raise KGError(f"{self.endpoint}: {type(exc).__name__}: {str(exc)[:160]}") from None

    @staticmethod
    def value(binding: dict, name: str) -> str | None:
        found = binding.get(name)
        return found.get("value") if isinstance(found, dict) else None

    def health(self) -> bool:
        try:
            return bool(self.ask("SELECT ?s WHERE { ?s ?p ?o } LIMIT 1", timeout=20))
        except KGError:
            return False

    # -- the vocabulary -------------------------------------------------------

    def predicates(self) -> dict[str, str]:
        """token -> URI for every relation in the graph."""
        if self._predicates:
            return self._predicates
        # BioCQ keeps its original cache name; another graph gets its own file.
        cache = ARTIFACT_ROOT / ("biokg-predicates.json" if self.schema_name == "biocq"
                                 else f"{self.schema_name}-predicates.json")
        if cache.exists():
            with self._lock:
                self._predicates = json.loads(cache.read_text())
            return self._predicates
        found = {}
        for graph in self.graphs:
            rows = self.ask(f"SELECT ?p (COUNT(*) AS ?n) WHERE {{ ?s ?p ?o . "
                            f"FILTER(STRSTARTS(STR(?p),'{BASE}/{graph}/relation/')) }} "
                            f"GROUP BY ?p LIMIT 5000", timeout=600)
            for row in rows:
                uri = self.value(row, "p") or ""
                if uri and not any(n in uri for n in NOISE) and int(self.value(row, "n") or 0) >= MIN_EDGES.get(graph, 0):
                    found[token_of(uri)] = uri
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(found, indent=1))
        with self._lock:
            self._predicates = found
        return self._predicates

    def vocabulary(self) -> set[str]:
        return set(self.predicates())

    def schema(self) -> list[dict]:
        """The whole graph's shape: every relation, what kind it joins to what kind, how often."""
        return load_schema(self.schema_name)

    def schema_notes(self) -> list[str]:
        return load_schema_notes(self.schema_name)

    # -- the dialect ----------------------------------------------------------

    def _predicate_uri(self, token: str) -> str | None:
        """The URI a predicate token names, however it was spelled."""
        if token.startswith(("?", "http")):
            return token if token.startswith("http") else None
        known = self.predicates()
        if token in known:
            return known[token]
        tail = re.sub(r"[^A-Za-z0-9]+", "_", token.split(".")[-1])
        for name, uri in known.items():
            if name.split(".", 1)[-1] == tail:
                return uri
        return None

    def term(self, token: str) -> str:
        """One token of the dialect as SPARQL."""
        if token.startswith("?"):
            return token
        if token.startswith("http"):
            return angle(token)
        if ENTITY.match(token):
            return angle(uri_of(token))
        uri = self.predicates().get(token)
        if uri:
            return angle(uri)
        # A predicate the model wrote without its graph, or with a different spelling of the tail.
        tail = re.sub(r"[^A-Za-z0-9]+", "_", token.split(".")[-1])
        for known, known_uri in self.predicates().items():
            if known.split(".", 1)[-1] == tail:
                return angle(known_uri)
        return f'"{token}"'

    def to_sparql(self, query: str, limit: int = 500) -> str | None:
        """Kora's compact triple dialect as SPARQL, or None when it is not one."""
        match = re.match(r"^\s*SELECT\s+(DISTINCT\s+)?(COUNT\()?\s*(\?\w+)\)?\s*WHERE\s*\{(.*)\}\s*$",
                         query, re.S | re.I)
        if not match:
            return None
        counting, variable, body = bool(match.group(2)), match.group(3), match.group(4)
        patterns = []
        for triple in [t.strip() for t in body.split(" . ") if t.strip()]:
            parts = triple.replace("\n", " ").split()
            if len(parts) != 3:
                return None
            subject, predicate, obj = parts
            if predicate == ":type":
                kind = obj if obj.startswith("?") else angle(f"{BASE}/{obj.split(':')[0].lower()}/type/{obj}") \
                    if ":" in obj else None
                # A kind written plainly ("drug"): match its URI tail instead of guessing a graph.
                if kind is None:
                    patterns.append(f"{self.term(subject)} {angle(RDF_TYPE)} ?t_{subject.lstrip('?')} . "
                                    f"FILTER(STRENDS(STR(?t_{subject.lstrip('?')}), '/{obj}'))")
                else:
                    patterns.append(f"{self.term(subject)} {angle(RDF_TYPE)} {kind}")
                continue
            uri = self._predicate_uri(predicate)
            if uri and spaced(uri):
                # Bound by its string: a space cannot live inside a SPARQL IRI.
                bound = f"?p{len(patterns)}"      # not `variable`: that holds what SELECT returns
                patterns.append(f"{self.term(subject)} {bound} {self.term(obj)} . "
                                f'FILTER(STR({bound}) = "{uri}")')
            else:
                patterns.append(f"{self.term(subject)} {self.term(predicate)} {self.term(obj)}")
        if not patterns:
            return None
        select = f"(COUNT(DISTINCT {variable}) AS {variable})" if counting else f"DISTINCT {variable}"
        return f"SELECT {select} WHERE {{ {' . '.join(patterns)} }} LIMIT {limit}"

    def execute(self, query: str, sets: dict[str, list[str]] | None = None) -> KGResult:  # noqa: ARG002
        started = time.time()
        sparql = self.to_sparql(query)
        if sparql is None:
            return KGResult(query=query, error="not a query this graph understands",
                            latency_seconds=time.time() - started)
        try:
            bindings = self.ask(sparql)
        except KGError as exc:
            return KGResult(query=query, error=str(exc), latency_seconds=time.time() - started)
        rows = [{f"?{name}": id_of(self.value(binding, name) or "") for name in binding}
                for binding in bindings]
        return KGResult(query=query, rows=rows, latency_seconds=time.time() - started)

    # -- what a task needs to know about an entity ----------------------------

    def label(self, entity_id: str) -> str | None:
        key = id_of(entity_id)
        with self._lock:
            if key in self._labels:
                return self._labels[key]
        try:
            rows = self.ask(f"SELECT ?n WHERE {{ {angle(uri_of(key))} {angle(RDFS_LABEL)} ?n }} LIMIT 1",
                            timeout=20)
        except KGError:
            return None
        found = self.value(rows[0], "n") if rows else None
        with self._lock:
            self._labels[key] = found
        return found

    def labels(self, ids: list[str]) -> dict[str, str | None]:
        """Names for many ids, one query per hundred. Only what came back is remembered."""
        wanted = [id_of(i) for i in dict.fromkeys(ids) if i]
        with self._lock:
            known = {i: self._labels[i] for i in wanted if self._labels.get(i)}
        missing = [i for i in wanted if i not in known]
        for start in range(0, len(missing), 100):
            batch = missing[start:start + 100]
            values = " ".join(f"{angle(uri_of(i))}" for i in batch)
            try:
                rows = self.ask(f"SELECT ?e ?n WHERE {{ VALUES ?e {{ {values} }} "
                                f"?e {angle(RDFS_LABEL)} ?n }}", timeout=40)
            except KGError:
                continue
            found = {id_of(self.value(r, "e") or ""): self.value(r, "n") for r in rows}
            with self._lock:
                for i, name in found.items():
                    if name:
                        self._labels[i] = name
            known.update({i: n for i, n in found.items() if n})
        return {i: known.get(i) for i in wanted}

    def exists(self, entity_id: str) -> bool:
        try:
            return bool(self.ask(f"SELECT ?p WHERE {{ {angle(uri_of(id_of(entity_id)))} ?p ?o }} LIMIT 1",
                                 timeout=20))
        except KGError:
            return False

    def is_hub(self, entity_id: str, cap: int | None = None) -> bool:
        try:
            rows = self.ask(f"SELECT (COUNT(*) AS ?n) WHERE {{ {angle(uri_of(id_of(entity_id)))} ?p ?o }}",
                            timeout=25)
        except KGError:
            return False
        return int(self.value(rows[0], "n") or 0) > (cap or self.HUB) if rows else False

    def expand(self, entity_id: str, predicate: str, limit: int = 50) -> list[str]:
        uri = self.predicates().get(predicate) or predicate
        try:
            rows = self.ask(f"SELECT ?o WHERE {{ {angle(uri_of(id_of(entity_id)))} {angle(uri)} ?o }} "
                            f"LIMIT {limit}", timeout=30)
        except KGError:
            return []
        return [id_of(self.value(r, "o") or "") for r in rows]

    def local_schema(self, entity_id: str, limit: int = 20, wanted: str = "",  # noqa: ARG002
                     first: list[str] | None = None,                            # noqa: ARG002
                     failed_hops: set | None = None) -> LocalSchema:            # noqa: ARG002
        """Every relation on an entity, both ways, with a sample of what it reaches."""
        key = id_of(entity_id)
        with self._lock:
            if key in self._schemas:
                return self._schemas[key]
        uri, entries = uri_of(key), []
        for direction, pattern in (("out", f"{angle(uri)} ?p ?o"), ("in", f"?o ?p {angle(uri)}")):
            try:
                rows = self.ask(f"SELECT ?p (COUNT(*) AS ?n) (SAMPLE(?o) AS ?eg) WHERE {{ {pattern} "
                                f"FILTER(CONTAINS(STR(?p),'/relation/')) }} GROUP BY ?p "
                                f"ORDER BY DESC(?n) LIMIT 60", timeout=60)
            except KGError:
                rows = []
            for row in rows:
                predicate = self.value(row, "p") or ""
                sample = id_of(self.value(row, "eg") or "")
                entries.append(SchemaEntry(
                    predicate=token_of(predicate), direction=direction,
                    count=int(self.value(row, "n") or 0), sample_object=sample or None,
                    sample_name=self.label(sample) if sample else None,
                    target_type=self.kind(sample) if sample else None))
        found = LocalSchema(entity_id=key, entries=entries)
        with self._lock:
            self._schemas[key] = found
        return found

    def properties(self, entity_id: str) -> list[SchemaEntry]:
        return self.local_schema(entity_id).entries

    def properties_with_samples(self, ids: list[str]) -> dict[str, dict[str, list[str]]]:
        """Every literal an entity holds, directly or one unnamed record away, two values each."""
        found: dict[str, dict[str, list[str]]] = {}
        for entity_id in ids:
            me = angle(uri_of(id_of(entity_id)))
            paths: dict[str, list[str]] = {}
            for row in self._sampled(f"{me} ?p ?v . FILTER(isLiteral(?v))", "?p"):
                predicate = self.value(row, "p") or ""
                if predicate and not any(n in predicate for n in NOISE):
                    paths[token_of(predicate)] = self._samples(row)
            carried = (f"{me} ?p ?m . FILTER(isIRI(?m)) FILTER NOT EXISTS {{ ?m {angle(RDFS_LABEL)} ?l }} "
                       f"?m ?q ?v . FILTER(isLiteral(?v))")
            for row in self._sampled(carried, "?p ?q"):
                relation, field = self.value(row, "p") or "", self.value(row, "q") or ""
                if relation and field and not any(n in relation + field for n in NOISE):
                    paths[f"{token_of(relation)} / {token_of(field)}"] = self._samples(row)
            found[entity_id] = paths
        return found

    def _sampled(self, pattern: str, keys: str) -> list[dict]:
        try:
            return self.ask(f"SELECT {keys} (MIN(STR(?v)) AS ?a) (MAX(STR(?v)) AS ?b) WHERE {{ {pattern} }} "
                            f"GROUP BY {keys} LIMIT 300", timeout=30)
        except KGError:
            return []

    def _samples(self, row: dict) -> list[str]:
        return list(dict.fromkeys(v for v in (self.value(row, "a"), self.value(row, "b")) if v))

    def values_of(self, ids: list[str], path: str) -> dict[str, list[str]]:
        """A path read for every id in one query (rule 5); a spaced relation is bound by its string."""
        patterns, at = [], "?e"
        for n, token in enumerate(path.split(" / ")):
            uri = self._predicate_uri(token.strip()) or token.strip()
            step = "?v" if n == len(path.split(" / ")) - 1 else f"?m{n}"
            if spaced(uri):
                patterns.append(f'{at} ?p{n} {step} . FILTER(STR(?p{n}) = "{uri}")')
            else:
                patterns.append(f"{at} {angle(uri)} {step}")
            at = step
        members = " ".join(angle(uri_of(id_of(i))) for i in ids)
        try:
            rows = self.ask(f"SELECT ?e ?v WHERE {{ VALUES ?e {{ {members} }} {' . '.join(patterns)} . "
                            f"FILTER(isLiteral(?v)) }} LIMIT 5000", timeout=60)
        except KGError:
            return {}
        found: dict[str, list[str]] = {}
        for row in rows:
            entity, value = id_of(self.value(row, "e") or ""), self.value(row, "v")
            if entity and value and value not in found.setdefault(entity, []):
                found[entity].append(value)
        return found

    def opened(self, entity_id: str, relations: list[str],
               failed: set[tuple[str, str]] | None = None) -> LocalSchema:  # noqa: ARG002
        found = self.local_schema(entity_id, limit=len(relations))
        return LocalSchema(entity_id=entity_id,
                           entries=[e for e in found.entries if e.predicate in set(relations)],
                           paths=[p for p in found.paths if p.relations[0] in set(relations)])

    def known_types(self) -> set[str]:
        kinds = set()
        for row in self.schema():
            kinds |= {row["from"], row["to"]}
            kinds |= {kind for pair in row["also"] for kind in pair.split(" -> ")}
        return kinds - {"?"}

    def kind(self, entity_id: str) -> str | None:
        """What an entity IS, from its `rdf:type`."""
        key = id_of(entity_id)
        with self._lock:
            if key in self._kinds:
                return self._kinds[key]
        try:
            rows = self.ask(f"SELECT ?t WHERE {{ {angle(uri_of(key))} {angle(RDF_TYPE)} ?t }} LIMIT 1",
                            timeout=20)
        except KGError:
            return None
        found = kind_of(self.value(rows[0], "t") or "") if rows else None
        with self._lock:
            self._kinds[key] = found
        return found

    def close(self) -> None:
        return None
