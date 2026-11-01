"""The graphs: read the real Freebase (SELECTs only), write the clones as N-Triples, bulk-load them into a copy."""
from __future__ import annotations

import collections
import fcntl
import json
import os
import random
import re
import statistics
import subprocess
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable, Iterator

import requests

if TYPE_CHECKING:
    from .twin import Clone


# reading the real graph: every query a SELECT
NS = "http://rdf.freebase.com/ns/"
_ID = re.compile(r"(m|g)\.[0-9a-z_]+")                           # a bare Freebase id
PREFIX = f"PREFIX ns: <{NS}>\n"
XSD = "http://www.w3.org/2001/XMLSchema#"
BOILERPLATE = ("type.object.", "type.type.instance", "common.topic.", "freebase.", "kg.", "common.notable_for.",
               "http://www.w3.org/2000/01/rdf-schema#", "http://www.w3.org/2002/07/owl#",
               "http://www.w3.org/1999/02/22-rdf-syntax-ns#type", "http://rdf.freebase.com/key/")
GENERIC_TYPES = ("common.", "base.", "user.", "kg.", "freebase.")   # never a target type
DATE_TYPES = {XSD + t for t in ("date", "gYear", "gYearMonth", "dateTime")}
MAX_ROWS = 9000               # a reply this long may have been cut at the server's ResultSetMaxRows
LABEL_BATCH = 500             # ids per labels / aliases query
TARGET_SAMPLE = 200           # distinct ends of a relation read to find its target type
PAGE = 5000                   # rows per page of ``objects`` (under MAX_ROWS)
BUCKET_OVER = 2               # ``members``' hash bucket holds about this many times the sample asked for
_HAS_PREFIX = re.compile(r"\bPREFIX\s+ns:", re.I)
_DATETIME_CAST = re.compile(r"xsd:(?i:datetime)\(")     # the datasets write xsd:datetime(; Virtuoso knows dateTime
_MID = re.compile(r"[mg]\.\w+")
_INT = re.compile(r"[+-]?\d+")
_FLOAT = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")
_DATE = re.compile(r"(-?\d{4})(?:-(\d{2}))?(?:-(\d{2}))?(T[\d:.]+)?(?:Z|[+-]\d{2}:\d{2})?")
_NOT_BOILERPLATE = " && ".join(f'!STRSTARTS(STR(?p), "{b if "://" in b else NS + b}")' for b in BOILERPLATE)
_ISO_SHAPE = "^[0-9]{4}(-[0-9]{2}){0,2}"                        # a CE date; skips BC dates and bare times
_NUMBER_SHAPE = "^-?[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?$"       # castable to xsd:double
_HTTP = requests.Session()      # one kept-alive connection per endpoint: a GET per query exhausts a node's ports


def _escape(text: str) -> str:
    """A SPARQL string literal's inside: backslashes and quotes escaped."""
    return text.replace("\\", "\\\\").replace('"', '\\"')


TIMEOUT, LONG_TIMEOUT = 180, 600    # seconds; the long one for a scan of a whole type (``members``)
BUCKET_SALT = "v1"                   # the bucket is the same for every build, so shards can share its cache


class RowCap(RuntimeError):
    """A reply of ``MAX_ROWS`` rows or more: it may have been cut at the server's row cap."""


def _sparql_run(endpoint: str, query: str, timeout: int = TIMEOUT) -> list[dict]:
    """The harness's call: one GET, the SPARQL JSON bindings back; a reply Virtuoso marks as partial is refused."""
    r = _HTTP.get(endpoint, params={"query": query, "format": "application/sparql-results+json"}, timeout=timeout)
    r.raise_for_status()
    state, message = r.headers.get("X-SQL-State", ""), r.headers.get("X-SQL-Message", "")
    if state or "partial" in message.lower() or "timeout" in message.lower():
        raise RuntimeError(f"partial result from {endpoint}: {state} {message}")
    return r.json()["results"]["bindings"]


@dataclass
class TypeProfile:                # the profile ``pad`` (twin.py) pads from
    type: str
    degree_q1: int
    degree_median: int
    degree_q3: int
    edges: collections.Counter    # (relation, "out" | "in", target_type or None) -> edges over the sample
    sample_size: int              # nodes sampled


@dataclass
class LiteralRange:
    kind: str                     # "date" | "int" | "float", decided on the sample
    lo: str | int | float         # over all of the property's values; dates as ISO strings
    hi: str | int | float
    values: list[str]             # the sampled values of that kind, raw (the rest are ignored)
    datatypes: collections.Counter  # datatype IRI (None if unbound) -> sampled values, all of them


def boilerplate(relation: str) -> bool:
    return relation.startswith(BOILERPLATE)


def _bare(term: dict) -> str:
    v = term["value"]
    return v[len(NS):] if term.get("type") == "uri" and v.startswith(NS) else v


def _term(relation: str) -> str:
    return f"<{relation}>" if "://" in relation else f"ns:{relation}"


def _chunks(xs: list[str], n: int) -> Iterator[list[str]]:
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


def _values(ids: Iterable[str], var: str = "?s") -> str:
    return f"VALUES {var} {{ " + " ".join(f"ns:{i}" for i in ids) + " }"


def share(counts: dict[str, int], cap: int) -> dict[str, int]:
    """``cap`` split over ``counts`` in proportion, by largest remainder (``counts`` when they fit)."""
    total = sum(counts.values())
    if total <= cap:
        return dict(counts)
    exact = {k: n * cap / total for k, n in counts.items()}
    out = {k: int(x) for k, x in exact.items()}
    for k in sorted(counts, key=lambda k: (out[k] - exact[k], k))[:cap - sum(out.values())]:
        out[k] += 1
    return out


def _quartiles(xs: list[int]) -> tuple[int, int, int]:
    """Inclusive (linear) quartiles, rounded to the nearest int."""
    if len(xs) == 1:
        return xs[0], xs[0], xs[0]
    q1, q2, q3 = statistics.quantiles(xs, n=4, method="inclusive")
    return round(q1), round(q2), round(q3)


def _kind(value: str, datatype: str | None) -> str | None:
    """A date if its datatype says so; otherwise read from the text (the dump types numbers as strings)."""
    if datatype in DATE_TYPES:
        return "date" if _DATE.fullmatch(value) else None
    return next((k for k, p in (("int", _INT), ("float", _FLOAT), ("date", _DATE)) if p.fullmatch(value)), None)


class Freebase:
    def __init__(self, endpoint: str, run: Callable[[str, str], list[dict]] = _sparql_run,
                 cache_dir: Path | None = None):
        self.endpoint, self.cache_dir = endpoint, cache_dir             # cache_dir: ``members`` buckets, shared
        self.timeout = TIMEOUT                                            # raised only around a whole-type scan
        self.run = run if run is not _sparql_run else (lambda e, q: _sparql_run(e, q, self.timeout))
        self._targets: dict[tuple[str, str], str | None] = {}

    def select(self, query: str) -> list[dict[str, str]]:
        """Each binding as ``{"?var": value}``; the dataset's ``xsd:datetime(`` sent as ``xsd:dateTime(``."""
        query = _DATETIME_CAST.sub("xsd:dateTime(", query)
        if not _HAS_PREFIX.search(query):
            query = PREFIX + query
        rows = self.run(self.endpoint, query)
        if len(rows) >= MAX_ROWS:
            raise RowCap(f"{len(rows)} rows: the reply may be cut at the server's row cap")
        return [{f"?{k}": _bare(t) for k, t in row.items()} for row in rows]

    def _english(self, ids: list[str], relation: str, var: str) -> Iterator[tuple[str, str]]:
        """``(id, value)`` for each English ``relation`` value of each id, ``LABEL_BATCH`` ids per query."""
        for batch in _chunks(list(ids), LABEL_BATCH):
            q = f"SELECT ?s ?{var} WHERE {{ {_values(batch)} ?s ns:{relation} ?{var} . FILTER(lang(?{var}) = 'en') }}"
            for row in self.select(q):
                yield row["?s"], row[f"?{var}"]

    def labels(self, ids: list[str]) -> dict[str, str]:
        """The English ``type.object.name`` of every id that has one (the first, if several)."""
        names: dict[str, str] = {}
        for i, name in self._english(ids, "type.object.name", "name"):
            names.setdefault(i, name)
        return names

    def aliases(self, ids: list[str]) -> dict[str, set[str]]:
        """The English ``common.topic.alias`` values of every id that has any."""
        out: dict[str, set[str]] = collections.defaultdict(set)
        for i, alias in self._english(ids, "common.topic.alias", "alias"):
            out[i].add(alias)
        return dict(out)

    def out_edges(self, id: str, cap: int) -> dict[str, tuple[int, list[str]]]:
        """relation -> (count, objects) of the node's out-edges, a sample of ``cap`` edges for a hub."""
        rows = self.select(f"SELECT ?p (COUNT(*) AS ?n) WHERE {{ ns:{id} ?p ?o FILTER({_NOT_BOILERPLATE}) }} "
                           "GROUP BY ?p")
        counts = {r["?p"]: int(r["?n"]) for r in rows if not boilerplate(r["?p"])}
        objects: dict[str, list[str]] = collections.defaultdict(list)
        if sum(counts.values()) <= cap:
            for r in self.select(f"SELECT ?p ?o WHERE {{ ns:{id} ?p ?o FILTER({_NOT_BOILERPLATE}) }}"):
                if r["?p"] in counts:
                    objects[r["?p"]].append(r["?o"])
        else:
            for relation, n in share(counts, cap).items():
                if n:
                    rows = self.select(f"SELECT ?o WHERE {{ ns:{id} {_term(relation)} ?o }} LIMIT {n}")
                    objects[relation] = [r["?o"] for r in rows]
        return {relation: (n, objects[relation]) for relation, n in sorted(counts.items())}

    def objects(self, id: str, relation: str) -> list[str]:
        """Every object of the node's ``relation`` edges, read in ordered pages."""
        out: list[str] = []
        while True:                      # keyset pages: Virtuoso refuses a deep OFFSET over a sorted result
            last = out[-1] if out else None               # a bare id came from an IRI: compare on the full IRI
            after = f' FILTER(STR(?o) > "{_escape(NS + last if _ID.fullmatch(last) else last)}")' if out else ""
            rows = self.select(f"SELECT ?o WHERE {{ ns:{id} {_term(relation)} ?o .{after} }} ORDER BY STR(?o) "
                               f"LIMIT {PAGE}")
            out += [r["?o"] for r in rows]
            if len(rows) < PAGE:
                return out

    def members(self, type: str, n: int, salt: str) -> list[str]:
        """Up to ``n`` nodes of ``type``, an even sample from a hash bucket shared by every build."""
        found = self._bucket(type, n)
        return sorted(random.Random(salt).sample(found, n)) if len(found) > n else found

    def _bucket(self, type: str, n: int) -> list[str]:
        if not self.cache_dir:
            return self._scan(type, n)
        path = self.cache_dir / f"{type}.{n}.json"
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.with_suffix(".lock").open("w") as lock:     # shards wait for the first scan, never repeat it
            fcntl.flock(lock, fcntl.LOCK_EX)
            if path.is_file():
                return json.loads(path.read_text(encoding="utf-8"))
            found = self._scan(type, n)                       # a failure raises: no cache file, the lock freed
            tmp = path.with_suffix(f".{os.getpid()}.tmp")
            tmp.write_text(json.dumps(found), encoding="utf-8")
            tmp.replace(path)
            return found

    def _scan(self, type: str, n: int) -> list[str]:
        """The type's members in its bucket (``members``), read with the long timeout."""
        self.timeout = LONG_TIMEOUT
        try:
            rows = self.select(f"SELECT (COUNT(*) AS ?n) WHERE {{ ?s ns:type.object.type ns:{type} }}")
            count = int(rows[0]["?n"]) if rows else 0
            bucket = ""
            if count > BUCKET_OVER * n:
                below = format(int(BUCKET_OVER * n / count * 16 ** 8), "08x")
                bucket = f' FILTER(SUBSTR(MD5(CONCAT(STR(?s), "{BUCKET_SALT}")), 1, 8) < "{below}")'
            rows = self.select(f"SELECT ?s WHERE {{ ?s ns:type.object.type ns:{type} .{bucket} }}") if count else []
        finally:
            self.timeout = TIMEOUT
        return sorted({r["?s"] for r in rows})

    def types(self, id: str) -> list[str]:
        return [r["?t"] for r in self.select(f"SELECT ?t WHERE {{ ns:{id} ns:type.object.type ?t }}")]

    def notable(self, id: str) -> str | None:
        """The node's notable type, as Freebase stores it (the type's mid, ``m.01mp``), or None."""
        rows = self.select(f"SELECT ?t WHERE {{ ns:{id} ns:common.topic.notable_types ?t }} LIMIT 1")
        return rows[0]["?t"] if rows else None

    def is_cvt(self, id: str) -> bool:
        return not self.select(f"SELECT ?n WHERE {{ ns:{id} ns:type.object.name ?n }} LIMIT 1")

    def target_type(self, relation: str, direction: str = "out") -> str | None:
        """The commonest specific type of up to ``TARGET_SAMPLE`` ends of the relation; cached."""
        key = (relation, direction)
        if key not in self._targets:
            var = "?o" if direction == "out" else "?s"
            ends = self.select(f"SELECT DISTINCT {var} WHERE {{ ?s {_term(relation)} ?o }} LIMIT {TARGET_SAMPLE}")
            nodes = [r[var] for r in ends if _MID.fullmatch(r[var])]
            rows = self.select(f"SELECT ?t (COUNT(*) AS ?n) WHERE {{ {_values(nodes, '?o')} "
                               f"?o ns:type.object.type ?t }} GROUP BY ?t") if nodes else []
            best = min(((-int(r["?n"]), r["?t"]) for r in rows
                        if not r["?t"].startswith(BOILERPLATE + GENERIC_TYPES)), default=None)
            self._targets[key] = best[1] if best else None
        return self._targets[key]

    def literal_range(self, property: str, sample: int = 2000) -> LiteralRange:
        """The property's kind (dates or numbers) from a sample, and its range over all its values."""
        rows = self.select(f"SELECT ?v (DATATYPE(?v) AS ?dt) WHERE {{ ?s ns:{property} ?v }} LIMIT {sample}")
        kinds = [(r["?v"], _kind(r["?v"], r.get("?dt"))) for r in rows]
        dates = [v for v, k in kinds if k == "date"]
        numbers = [v for v, k in kinds if k in ("int", "float")]
        datatypes = collections.Counter(r.get("?dt") for r in rows)
        if not rows or 2 * (len(dates) + len(numbers)) < len(rows):
            raise ValueError(f"{property}: {len(dates) + len(numbers)} of {len(rows)} sampled values are dates or "
                             f"numbers ({dict(datatypes)})")
        if len(dates) >= len(numbers):
            kind, values, shape, cast = "date", dates, _ISO_SHAPE, "STR(?v)"
        else:
            kind = "float" if any(k == "float" for _, k in kinds) else "int"
            values, shape, cast = numbers, _NUMBER_SHAPE, f"<{XSD}double>(?v)"
        agg = self.select(f'SELECT (MIN({cast}) AS ?lo) (MAX({cast}) AS ?hi) WHERE {{ ?s ns:{property} ?v '
                          f'FILTER(REGEX(STR(?v), "{shape}")) }}')
        if not agg or "?lo" not in agg[0]:
            raise ValueError(f"{property}: no {kind} values to bound")
        lo, hi = agg[0]["?lo"], agg[0]["?hi"]
        if kind != "date":
            num = int if kind == "int" else float
            lo, hi = num(float(lo)), num(float(hi))
        return LiteralRange(kind, lo, hi, values, datatypes)


def collides_remote(*graphs: Freebase) -> Callable[[str], bool]:
    """Whether a name is an English name or alias on any of ``graphs``: exact, lower- and title-cased."""
    def collides(name: str) -> bool:
        for variant in dict.fromkeys((name, name.lower(), name.title())):
            text = variant.replace("\\", "\\\\").replace('"', '\\"')
            query = (f'SELECT ?s WHERE {{ ?s ?r "{text}"@en .\n'
                     "FILTER(?r = ns:type.object.name || ?r = ns:common.topic.alias) } LIMIT 1")
            if any(graph.select(query) for graph in graphs):
                return True
        return False
    return collides


def _cached(cache: dict, key: str, compute: Callable, errors: Path | None = None):
    """``compute(key)`` once per run; None when it fails, any failure but ``ValueError`` logged to ``errors``."""
    if key not in cache:
        try:
            cache[key] = compute(key)
        except ValueError:
            cache[key] = None
        except Exception:
            cache[key] = None
            if errors:
                with errors.open("a", encoding="utf-8") as f:
                    f.write(f"== {getattr(compute, '__name__', compute)} {key}\n{traceback.format_exc()}\n")
    return cache[key]


def _capped_degree(fb: Freebase, id: str, cap: int) -> int:
    """The node's non-boilerplate edges in and out, each direction counted up to ``cap`` (a hub is never scanned)."""
    total = 0
    for pattern in (f"ns:{id} ?p ?o", f"?o ?p ns:{id}"):
        rows = fb.select(f"SELECT (COUNT(*) AS ?n) WHERE {{ SELECT ?p WHERE {{ {pattern} FILTER({_NOT_BOILERPLATE}) }}"
                         f" LIMIT {cap} }}")
        total += int(rows[0]["?n"]) if rows else 0
    return total


# the build's limits
MAX_PATH = 200                    # path nodes (topics and whole candidate sets included) beyond which a question drops
PATH_TOO_LARGE = "path too large"
BENCHMARKS = ("webqsp", "cwq")  # their real questions' topics and answers are never padding targets


def benchmark_nodes(splits: Path) -> set[str]:
    """Every topic and answer id of the real WebQSP and CWQ questions, train, val and test."""
    nodes: set[str] = set()
    for name in BENCHMARKS:
        for split in ("train", "val", "test"):
            for q in json.loads((splits / name / f"{split}.json").read_text(encoding="utf-8"))["questions"]:
                nodes.update(q.get("topic_entity_ids") or (), q.get("topic_entity") or {}, q.get("answer_ids") or ())
    return nodes


# writing the clones, and loading them into a copy (never the real database)
GRAPH = "http://freebase.com"
ISQL_TIMEOUT = 7200             # seconds for one isql call: a bulk load and its checkpoint
NAME_LITERALS = ("type.object.", "common.topic.")    # a clone's names are ``names``' alone
NEXT_ID = "next_id.txt"        # in <replica-root>/load/: the clone number the next build on the replica starts at
LOCK = ".lock"                  # in <root>/load/: one load at a time per replica
_ESCAPES = str.maketrans({**{c: f"\\u{c:04X}" for c in [*range(0x20), 0x7F]},
                          "\\": "\\\\", '"': '\\"', "\n": "\\n", "\r": "\\r", "\t": "\\t"})
_INI = re.compile(r"^\s*(/\S.*?\.ini)\s*$", re.M)     # the absolute path in virtuoso_ini_path()'s reply
_IRI = re.compile(r'[^\x00-\x20<>"{}|^`\\]+')
_NO_ERRORS = re.compile(r"(?<!\d)0 Rows\.")       # "10 Rows." holds "0 Rows." too


def _iri(id: str) -> str:
    iri = id if "://" in id else NS + id
    if not _IRI.fullmatch(iri):
        raise ValueError(f"not writable as an IRI: {iri!r}")
    return f"<{iri}>"


def _string(value: str) -> str:
    return f'"{value.translate(_ESCAPES)}"'


def _typed(value: str, datatype: str | None) -> str:
    if datatype is None:
        return _string(value)
    if not datatype.startswith(XSD):
        raise ValueError(f"not an XSD datatype: {datatype!r}")
    return f"{_string(value)}^^{_iri(datatype)}"


def _lines(clone: Clone, names: dict[str, str], literals: dict[tuple[str, str, str], str],
           notable: dict[str, str | None], datatypes: dict[str, str | None]) -> Iterator[str]:
    for cid in clone.mapping.values():
        if cid in names:
            yield f"{_iri(cid)} {_iri('type.object.name')} {_string(names[cid])}@en ."
        for type in clone.source_types[cid]:
            yield f"{_iri(cid)} {_iri('type.object.type')} {_iri(type)} ."
        if notable.get(cid):
            yield f"{_iri(cid)} {_iri('common.topic.notable_types')} {_iri(notable[cid])} ."
    for s, p, o in clone.edges + clone.kept_edges:
        yield f"{_iri(s)} {_iri(p)} {_iri(o)} ."
    for s, p, old in clone.literals:
        if not p.startswith(NAME_LITERALS):
            yield f"{_iri(s)} {_iri(p)} {_typed(literals.get((s, p, old), old), datatypes[p])} ."


def _datatype(bounds: LiteralRange | None) -> str | None:
    return bounds.datatypes.most_common(1)[0][0] if bounds and bounds.datatypes else None


def ntriples(clones: list[Clone], names: dict[str, str], literals: dict[tuple[str, str, str], str],
             notable: dict[str, str | None], datatypes: dict[str, str | None], out: Path) -> int:
    """Write the clones to ``out`` (see the module docstring); the number of triples written."""
    ids = {cid for clone in clones for cid in clone.mapping.values()}
    for what, keys, known in (("names", names, ids), ("notable", notable, ids),
                              ("literals", literals, {literal for clone in clones for literal in clone.literals})):
        if stray := set(keys) - known:
            raise ValueError(f"{len(stray)} keys of {what} match no clone, e.g. {min(stray)}")
    ntriples.skipped = sum(p.startswith(NAME_LITERALS) for clone in clones for _, p, _ in clone.literals)
    lines = dict.fromkeys(line for clone in clones for line in _lines(clone, names, literals, notable, datatypes))
    Path(out).write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    return len(lines)


ntriples.skipped = 0


def refuse_source(root: Path) -> None:
    here = Path(root).resolve()
    for env in ("FREEBASE_DB_SRC", "FREEBASE_LIVE_REPLICA"):
        if os.environ.get(env) and here == Path(os.environ[env]).resolve():
            raise RuntimeError(f"refusing to write {here}: it is ${env}")
    if not (here / "virtuoso.db").is_file():
        raise RuntimeError(f"refusing to write {here}: no virtuoso.db, not a replica")


def run_isql(argv: list[str]) -> str:
    r = subprocess.run(argv, capture_output=True, text=True, timeout=ISQL_TIMEOUT)
    if r.returncode != 0 or "*** Error" in r.stdout + r.stderr:
        raise RuntimeError(f"isql failed: {argv[-1][:200]}\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    return r.stdout


def bulk_load(root: Path, host: str, isql_port: int, nt: Path, graph: str = GRAPH,
              isql_bin: str = os.environ.get("ISQL_BIN", "isql"), run: Callable[[list[str]], str] = run_isql) -> None:
    """Load ``nt`` (inside ``root``) into ``graph`` on the replica at ``host:isql_port``; raise on any loader error."""
    refuse_source(root)
    nt = Path(nt).resolve()
    if not nt.is_relative_to(Path(root).resolve()):
        raise ValueError(f"{nt} is not inside the replica {root}: its DirsAllowed would refuse it")
    if not nt.is_file():
        raise FileNotFoundError(nt)
    folder = nt.parent

    def isql(sql: str) -> str:
        return run([isql_bin, f"{host}:{isql_port}", "dba", "dba", f"exec={sql}"])
    reply = isql("SELECT virtuoso_ini_path();")
    ini = _INI.search(reply)
    if not ini or not Path(ini.group(1)).resolve().is_relative_to(Path(root).resolve()):
        raise RuntimeError(f"refusing to load into {host}:{isql_port}: its ini is not inside {root}:\n{reply}")
    lock = Path(root) / "load" / LOCK
    lock.parent.mkdir(exist_ok=True)
    with lock.open("a") as held:                      # closing the file releases the lock, on any exit
        fcntl.flock(held, fcntl.LOCK_EX)
        isql(f"DELETE FROM DB.DBA.load_list WHERE ll_file = '{nt}'; "
             f"ld_dir('{folder}', '{nt.name}', '{graph}'); rdf_loader_run(); checkpoint;")
        errors = isql(f"SELECT ll_file, ll_error FROM DB.DBA.load_list WHERE ll_file = '{nt}' "
                      "AND ll_error IS NOT NULL;")
        if not _NO_ERRORS.search(errors):
            raise RuntimeError(f"bulk load reported errors:\n{errors}")
