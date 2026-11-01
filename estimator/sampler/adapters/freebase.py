"""The Freebase adapter: WebQSP, CWQ and GrailQA over KORA's own `FreebaseSPARQL` provider.

    make_graph("webqsp" | "cwq" | "grailqa") -> Graph

The graph is read through the provider's raw `ask` (bounded SPARQL, `VALUES` and `LIMIT`), its `labels` (names) and
`resolve_name` (same-name nodes). Nothing here writes.

The spine. The gold is turned into a *plan*: a set of triple patterns (a tree rooted at `?x`), the *chain* of them
from one topic entity to `?x` (the hops, each with its direction) and the *sides*, the other groups of patterns (a
class of the answer, a filter, the second branch of a conjunction, a superlative). Three sources build the plan, in
this order:

- WebQSP and CWQ carry the gold SPARQL. Its top-level triples are the tree; what the triples do not say (a date
  filter, `ORDER BY ... LIMIT`, `UNION`) stays in the gold text, which is what the end set is read from.
- GrailQA carries an s-expression; `(AND class (JOIN r e) ...)`, `le/lt/ge/gt`, `ARGMIN/ARGMAX` become triples,
  filters and an order. `COUNT` and `TC` are not walkable: None.
- Otherwise `gold["hops"]` and `gold["constraints"]` are the chain and the sides (a `class` keeps the nodes of that
  type, a `filter` the nodes with the value, a `superlative` the extreme by the property). A constraint goes on the
  answer set if that leaves something, else on the earlier set whose `to_kind` is the property's type.

The end set (the candidate) is one query over the whole tree (so a hub in the middle of the chain cannot truncate
it); the hops before it are walked one step at a time, each step restricted by the sides that hang on it, to show the
route (a mediator is a node with no English name: `Node.name` None, `?x<k>` in the evidence). Dropping a side gives
the `drop_constraint` variant. A literal object (a date, a number) is a node with id `lit:<value>` and the value as name.
"""
from __future__ import annotations

import ast
import itertools
import os
import re
from dataclasses import dataclass, field

from ..graph import Rel
from ..rows import Constraint, Hop, Node, Path

ENDPOINT = "http://localhost:8890/sparql"
NS = "http://rdf.freebase.com/ns/"
PREFIX = f"PREFIX ns: <{NS}>\nPREFIX xsd: <http://www.w3.org/2001/XMLSchema#>"
#: Relations that are bookkeeping, not facts about the world.
SKIP = ("type.object.", "common.", "kg.", "freebase.", "dataworld.")
MID = re.compile(r"^[mg]\.[0-9a-z_]+$")
REL = re.compile(r"^[A-Za-z0-9_.]+$")
LIT = "lit:"
MAX_MEDIATORS = 5           # unnamed nodes sampled when listing relations
MAX_NAMED = 10              # named nodes sampled likewise
RELATION_LIMIT = 300
SPINE_LIMIT = 2000          # objects kept at a spine hop
END_LIMIT = 5000            # rows read for the end set
CHUNK = 400                 # ids per VALUES block
VARIANTS = 4
WRONG_VALUES = 3          # wrong-value variants per question
TERM = r'(?:\?\w+|ns:[\w.]+|"[^"]*"(?:@\w+|\^\^[\w:]+|\^\^<[^>]+>)?|-?[\d.]+)'
TRIPLE = re.compile(rf"(\?\w+|ns:[\w.]+)\s+(ns:[\w.]+)\s+({TERM})\s*(?:\.|$)")
DIFFER = re.compile(r"^FILTER\s*\(\s*\?x\s*!=\s*(ns:[\w.]+|\?\w+)\s*\)$")
LOCALE = re.compile(r"^FILTER\s*\(\s*!isLiteral\(\?x\)")


class Unsupported(Exception):
    """The gold uses something the walk cannot express (COUNT, TC, ...): the question has no spine."""


def _ns(i: str) -> str:
    return f"ns:{i}"


def _lit_node(value: str) -> Node:
    return Node(LIT + value, value)


def parse_gold(gold) -> dict:
    if isinstance(gold, str):
        try:
            gold = ast.literal_eval(gold)
        except (ValueError, SyntaxError):
            return {}
    return gold if isinstance(gold, dict) else {}


# ----------------------------------------------------------------------------------------------------------------
# The plan
# ----------------------------------------------------------------------------------------------------------------

@dataclass
class Side:
    """A group of patterns hanging on the chain variable at position `at` (1-based step; 0: not placeable, so it
    shows in no hop but still counts in the end set). `real` sides (a constant, a filter or an order in them) are
    constraints a `drop_constraint` variant may leave out."""
    at: int
    idxs: list[int]
    filters: list[str] = field(default_factory=list)
    order: tuple[str, str] | None = None        # (variable, "asc" | "desc")
    real: bool = True


@dataclass
class Plan:
    triples: list[tuple[str, str, str]]
    chain: list[tuple[int, str, str]]           # (triple index, relation, direction) from the anchor to ?x
    anchor: str
    chain_vars: list[str]                       # [anchor term, ?v1, ..., ?x]
    sides: list[Side] = field(default_factory=list)
    differ: list[str] = field(default_factory=list)
    gold_query: str | None = None               # the gold text, run as is for the end set
    complex: bool = False                       # the gold says more than its triples (a date filter, ORDER BY, UNION)


def _strip_comments(text: str) -> str:
    return re.sub(r"(?m)(^|\s)#[^\n]*", " ", text)


def _balanced(text: str, start: int, open_: str, close: str) -> int:
    depth, i, quote = 0, start, False
    while i < len(text):
        c = text[i]
        if c == '"':
            quote = not quote
        elif not quote:
            depth += c == open_
            depth -= c == close
            if depth == 0:
                return i
        i += 1
    return len(text) - 1


def sparql_triples(query: str) -> tuple[list[tuple[str, str, str]], bool, list[str]] | None:
    """The top-level triple patterns of a gold SPARQL, whether it says more than they do (a FILTER other than the
    boilerplate, a block, ORDER BY, LIMIT), and the terms `?x` must differ from. None when there is no WHERE block."""
    if "WHERE {" not in query:
        return None
    head, body = query.split("WHERE {", 1)
    end = body.rfind("}")
    tail = body[end + 1:].strip() if end >= 0 else ""
    body = _strip_comments(body[:end] if end >= 0 else body)
    complex_, differ, flat, i = bool(tail), [], [], 0
    while i < len(body):
        if body.startswith("FILTER", i):
            j = _balanced(body, body.index("(", i), "(", ")")
            text = re.sub(r"\s+", " ", body[i:j + 1]).strip()
            m = DIFFER.match(text)
            if m:
                differ.append(m.group(1))
            elif not LOCALE.match(text):
                complex_ = True
            i = j + 1
        elif body[i] == "{":
            i = _balanced(body, i, "{", "}") + 1
            complex_ = True
        else:
            flat.append(body[i])
            i += 1
    text = "".join(flat)
    if re.search(r"\b(UNION|OPTIONAL|MINUS|BIND)\b", text):
        complex_ = True
    triples = [(m.group(1), m.group(2), m.group(3)) for m in TRIPLE.finditer(re.sub(r"\s+", " ", text))]
    return triples, complex_, differ


def _is_var(t: str) -> bool:
    return t.startswith("?")


def _plan_from_triples(triples: list[tuple[str, str, str]], anchor_ids: list[str], hub, differ=(), filters=(),
                       orders=(), gold_query: str | None = None, complex_: bool = False) -> Plan | None:
    """The chain is the longest route from an anchor to `?x`; every other group of patterns is a side."""
    adj: dict[str, list[tuple[int, str, str]]] = {}
    for k, (s, _, o) in enumerate(triples):
        adj.setdefault(s, []).append((k, o, "out"))
        adj.setdefault(o, []).append((k, s, "in"))
    consts = {t for tr in triples for t in (tr[0], tr[2]) if t.startswith("ns:")}
    routes: dict[str, list[tuple[int, str, str]]] = {}
    for a in anchor_ids:
        start = _ns(a)
        if start not in adj:
            continue
        queue, seen = [(start, [])], {start}      # a tree: the first route found is the only one
        while queue:
            node, route = queue.pop(0)
            if node == "?x":
                routes[a] = route
                break
            for k, other, d in adj.get(node, []):
                if other in seen or (other in consts and other != "?x"):
                    continue
                seen.add(other)
                queue.append((other, route + [(k, other, d)]))
    if not routes:
        return None
    longest = max(len(r) for r in routes.values())
    tied = [a for a in routes if len(routes[a]) == longest]
    best = sorted(tied, key=lambda a: bool(hub(a)))[0]      # a hub (a time zone, a gender) is never the start if avoidable
    route = routes[best]
    chain_vars = [_ns(best)] + [o for _, o, _ in route]
    chain = [(k, triples[k][1][3:], d) for k, _, d in route]
    used = {k for k, _, _ in route}
    rest = [k for k in range(len(triples)) if k not in used]
    parent = {k: k for k in rest}

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k
    by_var: dict[str, list[int]] = {}
    for k in rest:
        for t in (triples[k][0], triples[k][2]):
            if _is_var(t) and t not in chain_vars:
                by_var.setdefault(t, []).append(k)
    for ks in by_var.values():
        for k in ks[1:]:
            parent[find(k)] = find(ks[0])
    groups: dict[int, list[int]] = {}
    for k in rest:
        groups.setdefault(find(k), []).append(k)
    sides = []
    for ks in groups.values():
        terms = {t for k in ks for t in (triples[k][0], triples[k][2])}
        gfilters = [text for text, var in filters if var in terms]
        gorder = next(((v, d) for v, d in orders if v in terms), None)
        real = any(not _is_var(t) for t in terms) or bool(gfilters) or gorder is not None
        positions = {chain_vars.index(t) for t in terms if t in chain_vars}
        at = positions.pop() if len(positions) == 1 else 0
        sides.append(Side(at, ks, gfilters, gorder, real))
    return Plan(triples, chain, best, chain_vars, sides, list(differ), gold_query, complex_)


def _plan_from_sparql(query: str, anchor_ids: list[str], hub) -> Plan | None:
    parsed = sparql_triples(query)
    if parsed is None or not parsed[0]:
        return None
    triples, complex_, differ = parsed
    text = re.sub(r"(?m)^\s*#[^\n]*\n", "", query).lstrip()
    return _plan_from_triples(triples, anchor_ids, hub, differ=differ, gold_query=text, complex_=complex_)


# ----------------------------------------------------------------------------------------------------------------
# GrailQA: the s-expression
# ----------------------------------------------------------------------------------------------------------------

def _read_sexpr(text: str):
    tokens = re.findall(r"\(|\)|[^\s()]+", text)
    pos = 0

    def read():
        nonlocal pos
        tok = tokens[pos]
        pos += 1
        if tok == "(":
            out = []
            while tokens[pos] != ")":
                out.append(read())
            pos += 1
            return out
        return tok
    return read()


_OPS = {"le": "<=", "lt": "<", "ge": ">=", "gt": ">"}


def _literal(tok: str) -> str:
    value, _, dtype = tok.partition("^^")
    return f'"{value}"^^<{dtype}>' if dtype else f'"{value}"'


class _Sexpr:
    """GrailQA's s-expression as triples over `?x`: a symbol with dots is a class, `m.`/`g.` an entity,
    `(R r)` the reverse of r; `(JOIN r e)` is `?x r e`, `(JOIN (R r) e)` is `e r ?x`."""

    def __init__(self):
        self.triples: list[tuple[str, str, str]] = []
        self.filters: list[tuple[str, str]] = []
        self.orders: list[tuple[str, str]] = []
        self.n = 0

    def fresh(self) -> str:
        self.n += 1
        return f"?y{self.n}"

    def rel(self, r) -> tuple[str, bool]:
        if isinstance(r, list) and len(r) == 2 and r[0] == "R" and isinstance(r[1], str):
            return r[1], True
        if isinstance(r, str) and REL.match(r):
            return r, False
        raise Unsupported(f"relation {r!r}")

    def binary(self, b, a: str, c: str) -> None:
        if isinstance(b, list) and b and b[0] == "JOIN":
            m = self.fresh()
            self.binary(b[1], a, m)
            self.binary(b[2], m, c)
            return
        r, rev = self.rel(b)
        self.triples.append((c, _ns(r), a) if rev else (a, _ns(r), c))

    def sets(self, e, var: str) -> None:
        if isinstance(e, str):
            if not REL.match(e) or MID.match(e):
                raise Unsupported(f"set {e!r}")
            self.triples.append((var, "ns:type.object.type", _ns(e)))
            return
        op = e[0]
        if op == "AND":
            for part in e[1:]:
                self.sets(part, var)
        elif op == "JOIN":
            r, rev = self.rel(e[1])
            x = e[2]
            if isinstance(x, str):
                obj = _ns(x) if MID.match(x) else _literal(x) if ("^^" in x or re.fullmatch(r"-?[\d.]+", x)) else None
                if obj is None:
                    raise Unsupported(f"join object {x!r}")
            else:
                obj = self.fresh()
                self.sets(x, obj)
            self.triples.append((obj, _ns(r), var) if rev else (var, _ns(r), obj))
        elif op in _OPS:
            num = self.fresh()
            self.triples.append((var, _ns(self.rel(e[1])[0]), num))
            self.filters.append((f"FILTER({num} {_OPS[op]} {_literal(e[2])})", num))
        elif op in ("ARGMAX", "ARGMIN"):
            self.sets(e[1], var)
            num = self.fresh()
            self.binary(e[2], var, num)
            self.orders.append((num, "desc" if op == "ARGMAX" else "asc"))
        else:
            raise Unsupported(op)


def _plan_from_sexpr(text: str, hub) -> Plan | None:
    try:
        conv = _Sexpr()
        conv.sets(_read_sexpr(text), "?x")
    except (Unsupported, IndexError, ValueError):
        return None
    anchors = list(dict.fromkeys(t[3:] for tr in conv.triples for t in (tr[0], tr[2])
                                 if t.startswith("ns:") and MID.match(t[3:])))
    return _plan_from_triples(conv.triples, anchors, hub, filters=conv.filters, orders=conv.orders)


# ----------------------------------------------------------------------------------------------------------------
# gold["hops"] and gold["constraints"]
# ----------------------------------------------------------------------------------------------------------------

_CMP = re.compile(r"^(<=|>=|<|>|=)\s*(\S+?)(?:\^\^(\S+))?$")


def _prefers(hops: list[dict], domain: str) -> list[int]:
    """Positions (1-based, the set after step i) a constraint on `domain` may sit: the sets of that kind first, the
    answer set next, then the rest, latest first."""
    n = len(hops)
    exact = [i for i in range(n, 0, -1) if hops[i - 1].get("to_kind") == domain]
    rest = [i for i in range(n, 0, -1) if i not in exact]
    return list(dict.fromkeys(exact + [n] + rest))


def _plans_from_hops(gold: dict, anchor_ids: list[str]) -> list[Plan]:
    """Candidate plans in preference order (the placements of the constraints); [] when the hops cannot be walked."""
    hops = gold.get("hops") or []
    if not hops or len(anchor_ids) != 1:
        return []
    n = len(hops)
    names = ["?x" if i == n else f"?v{i}" for i in range(1, n + 1)]
    chain_vars = [_ns(anchor_ids[0])] + names
    triples, chain = [], []
    for i, h in enumerate(hops, 1):
        a, b, rel = chain_vars[i - 1], chain_vars[i], _ns(h["relation"])
        triples.append((a, rel, b) if h.get("direction", "out") == "out" else (b, rel, a))
        chain.append((i - 1, h["relation"], h.get("direction", "out")))
    per_constraint: list[list[tuple[int, list[tuple[str, str, str]], list[str], tuple | None]]] = []
    for c in gold.get("constraints") or []:
        kind, prop, value = c.get("kind"), c.get("property") or "", c.get("value") or ""
        options = []
        for at in _prefers(hops, value if kind == "class" else prop.rsplit(".", 1)[0])[:3]:
            var = names[at - 1]
            if kind == "class":
                options.append((at, [(var, "ns:type.object.type", _ns(value))], [], None))
            elif kind == "filter":
                if MID.match(value) or re.fullmatch(r"[a-z_]+(\.[a-z_0-9]+)+", value):
                    options.append((at, [(var, _ns(prop), _ns(value))], [], None))
                elif _CMP.match(value):
                    op, lit, dtype = _CMP.match(value).groups()
                    typed = f'"{lit}"^^<{dtype}>' if dtype else lit
                    options.append((at, [(var, _ns(prop), "?num")], [f"FILTER(?num {op} {typed})"], None))
                else:
                    return []
            elif kind == "superlative" and gold.get("projection") in ("argmax", "argmin"):
                options.append((at, [(var, _ns(prop), "?num")], [], ("?num", "desc" if gold["projection"] == "argmax" else "asc")))
            else:
                return []           # count, or a kind the walk cannot express
        if not options:
            return []
        per_constraint.append(options)
    plans = []
    for combo in itertools.islice(itertools.product(*per_constraint), 6):
        all_triples, sides = list(triples), []
        for k, (at, extra, flt, order) in enumerate(combo):
            idxs = list(range(len(all_triples), len(all_triples) + len(extra)))
            # one constraint per `?num`: rename so two constraints never share it
            all_triples += [tuple(t.replace("?num", f"?num{k}") for t in tr) for tr in extra]
            sides.append(Side(at, idxs, [f.replace("?num", f"?num{k}") for f in flt],
                              (order[0].replace("?num", f"?num{k}"), order[1]) if order else None))
        plans.append(Plan(all_triples, chain, anchor_ids[0], chain_vars, sides))
    return plans or [Plan(triples, chain, anchor_ids[0], chain_vars)]


# ----------------------------------------------------------------------------------------------------------------
# The graph
# ----------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------
# Constraint lines: every restriction a path applied, as text
# ----------------------------------------------------------------------------------------------------------------

_SIDE_FILTER = re.compile(r"FILTER\s*\(\s*(\?\w+)\s*(<=|>=|<|>|=)\s*(.+?)\s*\)\s*$")
_TEXT_FILTER = re.compile(r"FILTER\s*\(\s*(?:xsd:\w+\s*\(\s*)?\?(\w+)\s*\)?\s*(<=|>=|<|>|=)\s*(\"[^\"]*\"(?:\^\^\S+?)?|-?[\d.]+)\s*\)")
_ORDER = re.compile(r"ORDER\s+BY\s+(DESC|ASC)?\s*\(?\s*(?:xsd:\w+\s*\(\s*)?\?(\w+)", re.I)
def _flip_filter(f: str) -> str:
    m = _SIDE_FILTER.match(f.strip())
    return f if not m else f"FILTER({m.group(1)} {_FLIP.get(m.group(2), m.group(2))} {m.group(3)})"


_FLIP = {"<": ">", ">": "<", "<=": ">=", ">=": "<="}


def _literal_text(term: str) -> str:
    """A SPARQL literal as display text: `"1969-09-13"^^xsd:dateTime` -> `1969-09-13`."""
    t = term.strip()
    if t.startswith('"'):
        t = t[1:t.index('"', 1)] if '"' in t[1:] else t[1:]
    return t[:10] if re.match(r"^\d{4}-\d{2}-\d{2}", t) else t


def _order_word(direction: str) -> str:
    return "(largest)" if direction == "desc" else "(smallest)"


def _binding(text: str, var: str) -> tuple[str, str] | None:
    """The (subject term, relation) of the pattern that binds `?var` as an object in `text`."""
    m = re.search(rf"(\?\w+|ns:[\w.]+)\s+ns:([\w.]+)\s+\?{re.escape(var)}\b", text)
    return (m.group(1), m.group(2)) if m else None


def text_constraints(gold: str, chain_vars: list[str]) -> list[tuple[str, str, str]]:
    """The filters and the `ORDER BY ... LIMIT` a gold SPARQL applies beyond its top-level triples, as
    (subject term, relation, value) with value `< 92`, `<= 2009-12-31` or `(largest)`. A superlative counts only with
    a LIMIT; a comparison only when its variable is bound by a pattern (the relation it compares on)."""
    out = []
    for m in _TEXT_FILTER.finditer(gold):
        var, op, lit = m.group(1), m.group(2), m.group(3)
        bound = _binding(gold, var)
        if bound and var != "x":
            out.append((bound[0], bound[1], f"{op} {_literal_text(lit)}"))
    om = _ORDER.search(gold)
    if om and re.search(r"\bLIMIT\s+\d+", gold[om.end():]):
        bound = _binding(gold, om.group(2))
        if bound:
            out.append((bound[0], bound[1], _order_word("desc" if (om.group(1) or "").upper() == "DESC" else "asc")))
    return out


def flip_text(gold: str) -> list[str]:
    """Wrong-value versions of a gold SPARQL: its order reversed, and each comparison filter flipped, one at a time."""
    out = []
    om = _ORDER.search(gold)
    if om and re.search(r"\bLIMIT\s+\d+", gold[om.end():]):
        seg = gold[om.start():om.end()]
        if (om.group(1) or "").upper() == "DESC":
            new = re.sub(r"DESC", "ASC", seg, count=1, flags=re.I)
        elif (om.group(1) or "").upper() == "ASC":
            new = re.sub(r"ASC", "DESC", seg, count=1, flags=re.I)
        else:
            new = re.sub(r"ORDER\s+BY\s+", "ORDER BY DESC", seg, count=1, flags=re.I)
        out.append(gold[:om.start()] + new + gold[om.end():])
    for m in _TEXT_FILTER.finditer(gold):
        op = m.group(2)
        if op in _FLIP:
            out.append(gold[:m.start(2)] + _FLIP[op] + gold[m.end(2):])
    return out


class FreebaseGraph:
    """`Graph` (see ../graph.py) over a `FreebaseSPARQL`-shaped provider: `ask(sparql)`, `labels(ids)`,
    `resolve_name(name, limit)` and, when it has it, `is_hub(id)`."""

    def __init__(self, bench: str, provider):
        self.bench = bench
        self.p = provider
        self.errors = 0
        self.last_failure: dict[str, str] = {}      # qid -> why the spine is None, for the report

    # -- queries ---------------------------------------------------------------------------------------------

    def _ask(self, sparql: str, timeout: float | None = None) -> list[dict]:
        text = f"{PREFIX}\n{sparql}"
        return self.p.ask(text, timeout=timeout) if timeout else self.p.ask(text)

    @staticmethod
    def _values(var: str, ids: list[str]) -> str:
        return f"VALUES {var} {{ {' '.join(_ns(i) for i in ids)} }}"

    @staticmethod
    def _term_id(binding: dict | None) -> str | None:
        """An id for a result binding: the mid, or `lit:<value>` for an English or untagged literal; None otherwise."""
        if not binding:
            return None
        value = binding.get("value", "")
        if binding.get("type") == "uri":
            return value[len(NS):] if value.startswith(NS) else None
        lang = binding.get("xml:lang")
        return None if lang and not lang.startswith("en") else LIT + value

    def _nodes(self, ids: list[str]) -> list[Node]:
        mids = [i for i in ids if not i.startswith(LIT)]
        names = self.p.labels(mids) if mids else {}
        return [(_lit_node(i[len(LIT):]) if i.startswith(LIT) else Node(i, names.get(i))) for i in ids]

    def _hub(self, mid: str) -> bool:
        try:
            return bool(self.p.is_hub(mid)) if hasattr(self.p, "is_hub") else False
        except Exception:  # noqa: BLE001
            return False

    # -- the Graph protocol ----------------------------------------------------------------------------------

    def relations(self, nodes: list[Node]) -> list[Rel]:
        mids = [n for n in nodes if MID.match(n.id)]
        unnamed = [n for n in mids if not n.named]
        sample = unnamed[:MAX_MEDIATORS] if unnamed else mids[:MAX_NAMED]
        if not sample:
            return []
        ids = [n.id for n in sample]
        counts: dict[tuple[str, str], int] = {}
        for direction, pattern in (("out", "?s ?p ?o"), ("in", "?o ?p ?s")):
            try:
                rows = self._ask(f"SELECT ?p (COUNT(*) AS ?n) WHERE {{ {self._values('?s', ids)} {pattern} }} "
                                 f"GROUP BY ?p ORDER BY DESC(?n) LIMIT {RELATION_LIMIT}")
            except Exception:  # noqa: BLE001 -- an endpoint refusal on a hub: list what we have
                self.errors += 1
                continue
            for row in rows:
                uri = row.get("p", {}).get("value", "")
                if not uri.startswith(NS):
                    continue
                rel = uri[len(NS):]
                if rel.startswith(SKIP) or not REL.match(rel):
                    continue
                counts[(rel, direction)] = counts.get((rel, direction), 0) + int(row["n"]["value"])
        return [Rel(r, d, c) for (r, d), c in sorted(counts.items(), key=lambda kv: -kv[1])]

    def follow(self, nodes: list[Node], rel: Rel, limit: int = 200) -> list[Node]:
        try:
            return self._nodes(self._follow_ids([n.id for n in nodes], rel.relation, rel.direction, limit))
        except Exception:  # noqa: BLE001 -- a refused or timed-out hop reaches nothing
            self.errors += 1
            return []

    def _follow_ids(self, ids: list[str], relation: str, direction: str, limit: int, where: str = "") -> list[str]:
        """The objects (subjects, for "in") of `relation` from `ids`; `where` adds patterns on `?o`."""
        ids = [i for i in dict.fromkeys(ids) if MID.match(i)]
        if not ids or not REL.match(relation):
            return []
        pattern = f"?s {_ns(relation)} ?o" if direction == "out" else f"?o {_ns(relation)} ?s"
        found: list[str] = []
        for start in range(0, len(ids), CHUNK):
            rows = self._ask(f"SELECT DISTINCT ?o WHERE {{ {self._values('?s', ids[start:start + CHUNK])} {pattern} . "
                             f"{where} }} LIMIT {limit}")
            found += [t for t in (self._term_id(r.get("o")) for r in rows) if t]
            if len(found) >= limit:
                break
        return list(dict.fromkeys(found))[:limit]

    def kinds(self, ids: list[str]) -> dict[str, str]:
        """What kind of thing each node is, by its notable type's name (`City/Town`, `Film`); a literal is
        a `value`. Read from the graph, so any system can supply it. Cached; {} on a refusal."""
        cache = self.__dict__.setdefault("_kinds", {})
        mids = [i for i in dict.fromkeys(ids) if MID.match(i) and i not in cache]
        for start in range(0, len(mids), CHUNK):
            chunk = mids[start:start + CHUNK]
            for i in chunk:
                cache[i] = ""
            try:
                rows = self._ask(f"SELECT ?s ?t WHERE {{ {self._values('?s', chunk)} ?s ns:common.topic.notable_types ?t }}")
            except Exception:  # noqa: BLE001
                self.errors += 1
                continue
            types = {}
            for r in rows:
                s_, t_ = self._term_id(r.get("s")), self._term_id(r.get("t"))
                if s_ and t_ and MID.match(t_):
                    types.setdefault(s_, t_)
            names = self.p.labels(list(set(types.values()))) if types else {}
            for s_, t_ in types.items():
                cache[s_] = names.get(t_) or ""
        return {i: ("value" if i.startswith(LIT) else cache.get(i, "")) for i in ids}

    def reverse_relations(self, relations: list[str], path: Path | None = None) -> set[str]:
        """The relations that state the gold path's facts the other way (`people.person.children` for
        `people.person.parents`): for each hop, the relations linking its objects back to its subjects on the hop's own
        edges. The Freebase copy holds no schema (`type.property.reverse_property` is absent), so the data decides.
        [] without a path or on a refusal."""
        if path is None:
            return set()
        found: set[str] = set()
        for hop in path.hops:
            subs = [n.id for n in hop.subjects if MID.match(n.id)][:5]
            objs = [n.id for n in hop.objects if MID.match(n.id)][:20]
            if not subs or not objs or not REL.match(hop.relation):
                continue
            # the hop's own triple, and the same pair linked the other way round: out is `s rel o` (back: `o ?r s`),
            # in is `o rel s` (back: `s ?r o`)
            fact, back = ("?s {r} ?o", "?o ?r ?s") if hop.direction == "out" else ("?o {r} ?s", "?s ?r ?o")
            try:
                rows = self._ask(f"SELECT ?r (COUNT(*) AS ?n) WHERE {{ {self._values('?s', subs)} {self._values('?o', objs)} "
                                 f"{fact.format(r=_ns(hop.relation))} . {back} }} GROUP BY ?r LIMIT 50")
            except Exception:  # noqa: BLE001
                self.errors += 1
                continue
            for row in rows:
                uri = row.get("r", {}).get("value", "")
                if uri.startswith(NS) and not uri[len(NS):].startswith(SKIP):
                    found.add(uri[len(NS):])
        return found

    def same_name(self, node: Node, limit: int = 3) -> list[Node]:
        if not node.named or node.id.startswith(LIT):
            return []
        try:
            found = self.p.resolve_name(node.name, limit + 1)
        except Exception:  # noqa: BLE001
            self.errors += 1
            return []
        return [Node(i, name) for i, name in found if i != node.id][:limit]

    # -- the spine -------------------------------------------------------------------------------------------

    def _anchor_ids(self, question: dict) -> list[str]:
        ids = question.get("topic_entity_ids") or list((question.get("topic_entity") or {}).keys())
        names = question.get("topic_entity") or {}
        ids = [i for i in ids if MID.match(i)]
        # A class entity (CWQ lists the answer's type among the topics, named by its own id) is never the start.
        return sorted(ids, key=lambda i: names.get(i) in (None, i))

    def _anchor_node(self, question: dict, mid: str) -> Node:
        name = (question.get("topic_entity") or {}).get(mid)
        if not name or name == mid:
            name = self.p.labels([mid]).get(mid)
        return Node(mid, name)

    def _plans(self, question: dict) -> list[Plan]:
        gold = parse_gold(question.get("gold"))
        anchors = self._anchor_ids(question)
        query = gold.get("query") or ""
        plan = None
        if "WHERE {" in query:
            plan = _plan_from_sparql(query, anchors, self._hub)
        elif query.lstrip().startswith("("):
            plan = _plan_from_sexpr(query, self._hub)
        if plan:
            return [plan]
        return _plans_from_hops(gold, anchors)

    @staticmethod
    def _body(plan: Plan, skip: Side | None, flip: Side | None = None,
              replace: tuple[str, str] | None = None) -> tuple[str, tuple[str, str] | None]:
        """The patterns and filters of the whole tree (minus `skip`), and the order if one is left. `flip` reverses
        that side's order or comparison; `replace` swaps one constant for another (the wrong-value variants)."""
        skipped = set(skip.idxs) if skip else set()
        lines = []
        for k, (s_, p_, o_) in enumerate(plan.triples):
            if k in skipped:
                continue
            if replace and replace[0] in (s_, o_):
                s_, o_ = (replace[1] if s_ == replace[0] else s_), (replace[1] if o_ == replace[0] else o_)
            lines.append(f"{s_} {p_} {o_} .")
        order = None
        for side in plan.sides:
            if side is skip:
                continue
            if side is flip:
                lines += [_flip_filter(f) for f in side.filters]
                if side.order:
                    order = order or (side.order[0], "asc" if side.order[1] == "desc" else "desc")
                continue
            lines += side.filters
            order = order or side.order
        lines += [f"FILTER(?x != {t})" for t in plan.differ]
        return "\n".join(lines), order

    def _end_ids(self, plan: Plan, skip: Side | None = None, simplified: bool = False, flip: Side | None = None,
                 replace: tuple[str, str] | None = None, text: str | None = None) -> list[str]:
        """The end set: the gold text as written (or `text`, a flipped copy of it), or the tree's patterns (minus a
        dropped side, with a flipped side or a swapped constant)."""
        use_text = text or (plan.gold_query if (skip is None and not simplified and flip is None and replace is None) else None)
        if use_text:
            q = use_text
            q = q if re.search(r"\bLIMIT\s+\d+\s*$", q.strip()) else q.rstrip() + f"\nLIMIT {END_LIMIT}"
            # The gold writes `xsd:datetime(`; Virtuoso spells the cast `dateTime`, and the gold declares no xsd prefix.
            q = q.replace("xsd:datetime(", "xsd:dateTime(")
            if "PREFIX xsd:" not in q:
                q = "PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>\n" + q
            rows = self.p.ask(q)
            return list(dict.fromkeys(t for t in (self._term_id(r.get("x")) for r in rows) if t))
        body, order = self._body(plan, skip, flip, replace)
        if order:
            var, direction = order
            rows = self._ask(f"SELECT DISTINCT ?x {var} WHERE {{ {body} }} ORDER BY "
                             f"{'DESC' if direction == 'desc' else 'ASC'}({var}) LIMIT 50")
            if not rows:
                return []
            top = rows[0].get(var[1:], {}).get("value")
            rows = [r for r in rows if r.get(var[1:], {}).get("value") == top]
        else:
            rows = self._ask(f"SELECT DISTINCT ?x WHERE {{ {body} }} LIMIT {END_LIMIT}")
        return list(dict.fromkeys(t for t in (self._term_id(r.get("x")) for r in rows) if t))

    # -- constraint lines --------------------------------------------------------------------------------------

    def _at(self, plan: Plan, term: str) -> int:
        """The chain position a term sits on (0: the anchor), else the answer set's."""
        return plan.chain_vars.index(term) if term in plan.chain_vars else len(plan.chain)

    def _term_text(self, term: str) -> str:
        if term.startswith("ns:"):
            i = term[3:]
            return (self.p.labels([i]).get(i) or i) if MID.match(i) else i
        return _literal_text(term) if not term.startswith("?") else term

    def _side_lines(self, plan: Plan, side: Side, flip: bool = False,
                    replace: tuple[str, str] | None = None) -> list[Constraint]:
        """One side as lines: a class as `type`; the rest as its relations (joined by /) and what they reach: a
        constant's name, a comparison (`< 92`) or an order (`(largest)`)."""
        at = side.at or len(plan.chain)
        triples = [plan.triples[k] for k in side.idxs]
        if replace:
            triples = [tuple(replace[1] if t == replace[0] else t for t in tr) for tr in triples]
        out = [Constraint(at, "type", tr[2][3:]) for tr in triples if tr[1] == "ns:type.object.type" and tr[2].startswith("ns:")]
        rest = [tr for tr in triples if tr[1] != "ns:type.object.type"]
        if not rest:
            return out
        relation = " / ".join(tr[1][3:] for tr in rest)
        if side.order:
            direction = side.order[1] if not flip else ("asc" if side.order[1] == "desc" else "desc")
            value = _order_word(direction)
        elif side.filters:
            parts = []
            for f in side.filters:
                m = _SIDE_FILTER.match(f.strip())
                if m:
                    op = _FLIP.get(m.group(2), m.group(2)) if flip else m.group(2)
                    parts.append(f"{op} {_literal_text(m.group(3))}")
            value = "; ".join(parts) or "(filtered)"
        else:
            consts = [t for tr in rest for t in (tr[0], tr[2]) if not t.startswith("?") and t not in plan.chain_vars]
            value = ", ".join(self._term_text(t) for t in dict.fromkeys(consts)) or "(present)"
        return out + [Constraint(at, relation, value)]

    def _lines(self, plan: Plan, skip: Side | None = None, flip: Side | None = None,
               replace: tuple[str, str, Side] | None = None, text: str | None = None) -> list[Constraint]:
        """The constraints a path built this way applied: every real side (but `skip`; `flip` and `replace` shown as
        changed) and, when its end set came from a gold text, that text's own filters and order."""
        out = []
        for side in plan.sides:
            if not side.real or side is skip:
                continue
            rep = (replace[0], replace[1]) if replace and replace[2] is side else None
            out += self._side_lines(plan, side, flip=side is flip, replace=rep)
        if text:
            for subj, rel, value in text_constraints(text, plan.chain_vars):
                out.append(Constraint(self._at(plan, subj), rel, value))
        return out

    def _walk(self, question: dict, plan: Plan, end: list[str], source: str, constraints: list[Constraint] | None = None,
              skip: Side | None = None) -> Path | None:
        """The chain, a step at a time, each restricted by the sides that hang on it (but `skip`), ending on `end`."""
        start = self._anchor_node(question, plan.anchor)
        cur, hops = [start], []
        for i, (_, relation, direction) in enumerate(plan.chain, 1):
            var = plan.chain_vars[i]
            where = ""
            for side in plan.sides:
                if side.at == i and side.real and not side.order and side is not skip:
                    where += " ".join(f"{' '.join(plan.triples[k])} ." for k in side.idxs) + " " + " ".join(side.filters) + " "
            where = re.sub(rf"(?<![\w?]){re.escape(var)}\b", "?o", where) if where else ""
            ids = self._follow_ids([n.id for n in cur], relation, direction, SPINE_LIMIT, where)
            if not ids and where:
                ids = self._follow_ids([n.id for n in cur], relation, direction, SPINE_LIMIT)
            if i == len(plan.chain):
                ids = end[:SPINE_LIMIT]
            if not ids:
                return None
            objs = self._nodes(ids)
            hops.append(Hop(cur, relation, direction, objs))
            cur = objs
        return Path([start], hops, source, constraints=list(constraints or []))

    def spine(self, question: dict) -> Path | None:
        qid = str(question.get("id"))
        try:
            plans = self._plans(question)
        except Exception as exc:  # noqa: BLE001
            self.errors += 1
            self.last_failure[qid] = f"plan {type(exc).__name__}"
            return None
        if not plans:
            self.last_failure[qid] = "gold not walkable"
            return None
        for plan in plans:
            try:
                end = self._end_ids(plan)
                if not end:
                    continue
                path = self._walk(question, plan, end, "spine", self._lines(plan, text=plan.gold_query))
            except Exception as exc:  # noqa: BLE001 -- the endpoint refused or timed out: no spine, counted
                self.errors += 1
                self.last_failure[qid] = f"query {type(exc).__name__}: {str(exc)[:80]}"
                continue
            if path is not None:
                path.extra["plan"] = plan
                return path
        self.last_failure.setdefault(qid, "walk empty")
        return None

    def _alternatives(self, plan: Plan, side: Side, const: str, keep: int = 3) -> list[str]:
        """Other constants the side's slot takes in the graph, most answers first (`another series`, `another year`)."""
        body, _ = self._body(plan, None, replace=(const, "?alt"))
        rows = self._ask(f"SELECT ?alt (COUNT(DISTINCT ?x) AS ?n) WHERE {{ {body} FILTER(?alt != {const}) }} "
                         f"GROUP BY ?alt ORDER BY DESC(?n) LIMIT {keep}")
        out = []
        for r in rows:
            b = r.get("alt") or {}
            if b.get("type") == "uri" and b.get("value", "").startswith(NS):
                out.append(_ns(b["value"][len(NS):]))
        return out

    def spine_variants(self, question: dict, spine: Path) -> list[Path]:
        """The superset (one constraint dropped, source drop_constraint) and the wrong value (an order or a comparison
        reversed, a constant swapped for another the slot takes: source wrong_value). Each path carries the constraint
        lines of exactly what built it."""
        plan: Plan | None = spine.extra.get("plan")
        if plan is None:
            return []
        out: list[Path] = []
        seen = {spine.key()}

        def add(end, source, lines, skip=None):
            if not end:
                return
            p = self._walk(question, plan, end, source, lines, skip=skip)
            if p is not None and p.key() not in seen:
                seen.add(p.key()); out.append(p)

        drops = [{"skip": s} for s in plan.sides if s.real]
        if plan.complex and plan.gold_query:
            drops.append({"simplified": True})         # the gold's date filter / ORDER BY / UNION left out
        for kw in drops[:VARIANTS]:
            try:
                add(self._end_ids(plan, **kw), "drop_constraint", self._lines(plan, skip=kw.get("skip")), kw.get("skip"))
            except Exception:  # noqa: BLE001
                self.errors += 1
        wrong = 0
        for side in plan.sides:
            if wrong >= WRONG_VALUES or not side.real:
                continue
            try:
                if side.order or side.filters:
                    add(self._end_ids(plan, flip=side), "wrong_value", self._lines(plan, flip=side), side); wrong += 1
                    continue
                consts = [t for k in side.idxs for t in (plan.triples[k][0], plan.triples[k][2])
                          if t.startswith("ns:") and MID.match(t[3:]) and plan.triples[k][1] != "ns:type.object.type"]
                for const in consts[:1]:
                    for alt in self._alternatives(plan, side, const)[:1]:
                        add(self._end_ids(plan, replace=(const, alt)), "wrong_value",
                            self._lines(plan, replace=(const, alt, side)), side); wrong += 1
            except Exception:  # noqa: BLE001
                self.errors += 1
        if plan.complex and plan.gold_query and wrong < WRONG_VALUES:
            for text in flip_text(plan.gold_query)[:WRONG_VALUES - wrong]:
                try:
                    add(self._end_ids(plan, text=text), "wrong_value", self._lines(plan, text=text))
                except Exception:  # noqa: BLE001
                    self.errors += 1
        return out


def make_graph(bench: str) -> FreebaseGraph:
    from kora.tools.sparql import FreebaseSPARQL
    return FreebaseGraph(bench, FreebaseSPARQL(os.environ.get("FREEBASE_SPARQL_URL", ENDPOINT), timeout=30))
