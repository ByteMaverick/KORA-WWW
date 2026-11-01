"""One question into one twin: expose its gold, clone the path, move its literals, pad, rewrite the question."""
from __future__ import annotations

import calendar
import collections
import copy
import itertools
import math
import random
import re
import weakref
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

from .graph import GENERIC_TYPES, LiteralRange, TypeProfile, share


# step 1: the gold query, every variable exposed
_COMMENT = re.compile(r"#[^\n]*")
_QUOTED = re.compile(r"""(["'])(.*?)\1(?:\^\^([\w:]+)|@[\w-]+)?""")
_LANG_FILTER = re.compile(r"FILTER\s*\(\s*!isLiteral\(\?\w+\) OR lang\(\?\w+\) = '' "
                          r"OR langMatches\(lang\(\?\w+\), 'en'\)\s*\)", re.I)
_FILTER = re.compile(r"\bFILTER\s*\(", re.I)
_BARE = re.compile(r"(?<![\w.:?])-?\d+(?:\.\d+)?(?![\w.])")
_CONSTANT = re.compile(r"ns:([mg]\.\w+)")
_VARIABLE = re.compile(r"\?\w+")
_TOKEN = re.compile(r"""(?P<term>"[^"]*"(?:\^\^[\w:]+|@[\w-]+)?|\?\w+|ns:\w+(?:\.\w+)*|-?\d+(?:\.\d+)?)"""
                    r"|(?P<sep>[.;{}])|[^\s.;{}]+")
_SELECT = re.compile(r"\bSELECT\b.*?(?=\s*\bWHERE\b)", re.I | re.S)
_LANG_TEST = re.compile(r"!?\s*isLiteral\s*\(\s*\?\w+\s*\)|lang\s*\(\s*\?\w+\s*\)\s*=\s*'[^']*'"
                        r"|langMatches\s*\(\s*lang\s*\(\s*\?\w+\s*\)\s*,\s*'[^']*'\s*\)", re.I)
_IDENTITY_TEST = re.compile(r"(?:\?\w+|ns:[\w.]+)\s*!?=\s*(?:\?\w+|ns:[\w.]+)")
_VALUE_TEST = re.compile(r"""["']|(?<![\w.:?])-?\d|[<>]|(?<![\w:])(?!(?i:or|and|not|exists)\b)[A-Za-z_][\w:]*\s*\(""")
_UNION = re.compile(r"UNION\b", re.I)
BRANCH, EXISTS = "?_branch", "?_exists"      # the markers expose binds: a row's UNION branch, a matched EXISTS group


@dataclass(frozen=True)
class Literal:
    value: str                 # the text inside the quotes, or the bare number
    datatype: str | None       # ``xsd:dateTime`` for ``"2015-08-10"^^xsd:dateTime``; None otherwise
    position: int              # order of appearance in the query, from 0


@dataclass
class Gold:
    select: str                                  # the answer variable
    variables: list[str]                         # every ?var, in order of first appearance
    constants: list[str]                         # every mid (``m.…`` / ``g.…``), in order of first appearance
    literals: list[Literal]
    order: tuple[str, str] | None                # (variable, "ASC" | "DESC") of the outer ORDER BY
    limit: int | None
    filters: list[str]                           # each FILTER clause, raw
    triples: list[tuple[str, str, str]]          # the pattern's triples, outside FILTER clauses
    value_filters: list[int] = field(default_factory=list)   # indexes into ``filters`` of the ones expose removes
    # every triple a row of expose(query) may instantiate (``triples`` in order, then those of the value filters'
    # EXISTS groups), each with the markers (``?_branchK``, ``?_existsK``) the row must bind for it to hold
    scoped: list[tuple[tuple[str, str, str], tuple[str, ...]]] = field(default_factory=list)


def _blank(text: str, spans: Iterable[tuple[int, int]]) -> str:
    """``text`` with every span overwritten by spaces, so positions in it still index the original."""
    chars = list(text)
    for start, end in spans:
        chars[start:end] = " " * (end - start)
    return "".join(chars)


def _masks(query: str) -> tuple[str, str]:
    """The query with comments blanked, and the same with strings blanked too (its structure alone)."""
    code = _blank(query, (m.span() for m in _COMMENT.finditer(query)))
    return code, _blank(code, (m.span() for m in _QUOTED.finditer(code)))


def _filter_spans(bare: str) -> list[tuple[int, int]]:
    """Each FILTER clause as (start, end), its parentheses balanced; a FILTER nested inside another is part of it."""
    spans: list[tuple[int, int]] = []
    for m in _FILTER.finditer(bare):
        if spans and m.start() < spans[-1][1]:
            continue
        depth, end = 1, m.end()
        while depth:
            depth += {"(": 1, ")": -1}.get(bare[end], 0)
            end += 1
        spans.append((m.start(), end))
    return spans


def _literal_spans(code: str, bare: str) -> list[tuple[int, int, Literal]]:
    """Every literal with the span of its value (inside the quotes for a string), in order of appearance."""
    text = _blank(code, (m.span() for m in _LANG_FILTER.finditer(code)))
    found = [(m.start(2), m.end(2), m.group(2), m.group(3)) for m in _QUOTED.finditer(text)]
    for start, end in _filter_spans(bare):
        found += [(m.start(), m.end(), m.group(), None) for m in _BARE.finditer(bare, start, end)]
    return [(s, e, Literal(v, t, i)) for i, (s, e, v, t) in enumerate(sorted(found))]


def _triples(text: str) -> list[tuple[str, str, str]]:
    """Subject-predicate-object runs ended by ``.``, ``;`` (the subject carries on) or a brace."""
    return [triple for triple, _ in _triples_at(text)]


def _triples_at(text: str) -> list[tuple[tuple[str, str, str], int]]:
    """``_triples`` with the position in ``text`` of each triple's last term."""
    triples: list[tuple[tuple[str, str, str], int]] = []
    terms: list[str] = []
    for m in _TOKEN.finditer(text):
        if m.group("term"):
            terms.append(m.group())
            at = m.start()
            continue
        if m.group("sep") and len(terms) == 3:
            triples.append(((terms[0], terms[1], terms[2]), at))
        terms = terms[:1] if m.group() == ";" and len(terms) == 3 else []
    return triples


def is_value_filter(clause: str) -> bool:
    """A FILTER clause that tests values (see the module docstring), not only languages or identities."""
    inner = clause[clause.index("(") + 1:clause.rindex(")")]
    return bool(_VALUE_TEST.search(_IDENTITY_TEST.sub(" ", _LANG_TEST.sub(" ", inner))))


def _groups(text: str, start: int = 0, end: int | None = None) -> list[tuple[int, int]]:
    """Every ``{…}`` group in ``text[start:end]`` as (position of ``{``, position of ``}``), in order of opening."""
    groups, stack = [], []
    for i in range(start, len(text) if end is None else end):
        if text[i] == "{":
            stack.append(len(groups))
            groups.append([i, None])
        elif text[i] == "}" and stack:
            groups[stack.pop()][1] = i
    return [(a, b) for a, b in groups if b is not None]


def _innermost(groups: list[tuple[int, int]], at: int) -> tuple[int, int]:
    return max((g for g in groups if g[0] < at < g[1]), key=lambda g: g[0])


def _structure(query: str):
    """What parse and expose read off the query: masks, FILTER spans, groups, UNION branches, EXISTS groups."""
    code, bare = _masks(query)
    spans = _filter_spans(bare)
    pattern = _blank(bare, spans)
    groups = _groups(pattern)
    branches = [(a, b) for a, b in groups
                if _UNION.match(pattern[b + 1:].lstrip()) or pattern[:a].rstrip().upper().endswith("UNION")]
    values = [i for i, (s, e) in enumerate(spans) if is_value_filter(code[s:e])]
    exists = []
    for i in values:
        s, e = spans[i]
        inside = bare.index("(", s) + 1
        nested = [(a + inside, b + inside) for a, b in _filter_spans(bare[inside:e])]
        text = _blank(code, nested)
        top: list[tuple[int, int]] = []
        for a, b in _groups(_blank(bare, nested), s, e):
            if not any(a0 < a < b0 for a0, b0 in top):
                top.append((a, b))
        exists += [(i, a, b, text[a + 1:b]) for a, b in top]
    return code, bare, spans, pattern, groups, branches, values, exists


def _markers(at: int, branches: list[tuple[int, int]]) -> tuple[str, ...]:
    return tuple(f"{BRANCH}{k}" for k, (a, b) in enumerate(branches) if a < at < b)


def _unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def parse(query: str) -> Gold:
    code, bare, spans, _, _, branches, values, exists = _structure(query)
    tail = bare[bare.rindex("}") + 1:]
    order = re.search(r"ORDER\s+BY\s+(ASC|DESC)?[^?]*(\?\w+)", tail, re.I)
    limit = re.search(r"LIMIT\s+(\d+)", tail, re.I)
    start = bare.index("{")
    body = _blank(code, spans)[start:bare.rindex("}") + 1]
    main = [(triple, _markers(start + at, branches)) for triple, at in _triples_at(body)]
    inner = [(triple, _markers(a, branches) + (f"{EXISTS}{k}",))
             for k, (_, a, _, text) in enumerate(exists) for triple, _ in _triples_at(text + " .")]
    return Gold(select=re.search(r"\bSELECT\b[^?]*(\?\w+)", bare, re.I).group(1),
                variables=_unique(_VARIABLE.findall(bare)),
                constants=_unique(_CONSTANT.findall(bare)),
                literals=[lit for _, _, lit in _literal_spans(code, bare)],
                order=(order.group(2), (order.group(1) or "ASC").upper()) if order else None,
                limit=int(limit.group(1)) if limit else None,
                filters=[query[s:e] for s, e in spans],
                triples=[triple for triple, _ in main],
                value_filters=values,
                scoped=main + inner)


def expose(query: str) -> str:
    """The query selecting every path variable and marker, its value filters and ORDER BY / LIMIT removed."""
    gold = parse(query)
    code, bare, spans, pattern, groups, branches, values, exists = _structure(query)
    removed: list[tuple[int, int]] = []
    working = pattern
    for i in values:
        s, e = spans[i]
        before = working[:s].rstrip()
        after = len(working) - len(working[e:].lstrip())
        if after < len(working) and working[after] == "." and (not before or before[-1] in ".{"):
            e = after + 1
        removed.append((s, e))
        working = _blank(working, [(s, e)])
    inserted: dict[int, list[str]] = {}             # at a group's closing brace, in order
    for k, (i, _, _, text) in enumerate(exists):
        close = _innermost(groups, spans[i][0])[1]
        inserted.setdefault(close, []).append(f"OPTIONAL {{ {text.strip()} BIND(1 AS {EXISTS}{k}) }}")
    for k, (_, close) in enumerate(branches):
        inserted.setdefault(close, []).append(f"BIND(1 AS {BRANCH}{k})")
    edits = [(s, e, "") for s, e in removed] + [(at, at, " " + " ".join(texts) + " ")
                                                  for at, texts in inserted.items()]
    on_path = {term for triple, _ in gold.scoped for term in triple}
    path = [v for v in gold.variables if v == gold.select or v in on_path]
    path += [f"{BRANCH}{k}" for k in range(len(branches))] + [f"{EXISTS}{k}" for k in range(len(exists))]
    out = query[:bare.rindex("}") + 1]                # every edit lies in the body, after the SELECT clause
    for start, end, new in sorted(edits, reverse=True):
        out = out[:start] + new + out[end:]
    select = _SELECT.search(bare)
    return out[:select.start()] + "SELECT DISTINCT " + " ".join(path) + out[select.end():] + "\n"


def substitute(query: str, ids: dict[str, str], literals: dict[int, str]) -> str:
    """Replace each mid in ``ids`` (``m.…`` → ``m.…``) and each literal value by position; nothing else changes."""
    code, bare = _masks(query)
    edits = [(m.start(1), m.end(1), ids[m.group(1)]) for m in _CONSTANT.finditer(bare) if m.group(1) in ids]
    edits += [(s, e, literals[lit.position]) for s, e, lit in _literal_spans(code, bare) if lit.position in literals]
    for start, end, new in sorted(edits, reverse=True):
        query = query[:start] + new + query[end:]
    return query


def _template(query: str) -> str:
    """The query with every constant and literal replaced by a placeholder: what "mod entities" counts."""
    gold = parse(query)
    return substitute(query, dict.fromkeys(gold.constants, "E"), {lit.position: "L" for lit in gold.literals})


# step 2: the path the exposed rows instantiate, and its clone
CLOSED_CLASS_FREEBASE = {"type.type"}      # classes only; with CLASS_RELATIONS' objects, the one not-list
CLASS_RELATIONS = {"type.object.type", "common.topic.notable_types"}   # their object is a class: always real
MAX_MISSES = 50                 # draws pad wastes (a duplicate, a self-loop, no target) before it stops a clone
CVT_CAP = 30                    # clone CVTs per relation per clone: more CVT records get a draw of this many
CLASS_TO_CLONE = "a class points at a clone"
_ID = re.compile(r"(m|g)\.[0-9a-z_]+")
_QUOTED_TERM = re.compile(r'"(.*)"(?:\^\^\S+|@\S+)?')
_BASE36 = "0123456789abcdefghijklmnopqrstuvwxyz"
Triple = tuple[str, str, str]


@dataclass
class Node:
    id: str
    role: str                      # "topic" | "answer" | "constant" | "mediator" | "intermediate"
    types: list[str]
    is_cvt: bool


@dataclass
class Path:
    nodes: dict[str, Node]
    edges: list[Triple]            # the triples the rows instantiate, each once
    literals: list[Triple]         # (subject, property, value) whose object is a literal
    answers: list[str]             # the answer variable's bindings, each once: mids, or literal values


@dataclass
class Clone:
    mapping: dict[str, str]        # source id -> clone id, for every cloned node
    edges: list[Triple]            # between clones; pad appends its clone <-> real edges here
    kept_edges: list[Triple]       # between a clone and a real node it kept
    literals: list[Triple]         # (clone id, property, value): the path's literals on the clones
    answers: list[str]             # the answers mapped; a kept node or a literal stays as it is
    types: dict[str, str]          # clone id -> its one type
    source_types: dict[str, list[str]]   # clone id -> every type of its source
    unpadded: collections.Counter = field(default_factory=collections.Counter)   # (clone, relation, reason) -> edges
    flags: list[str] = field(default_factory=list)   # the twin's: hub_kept_real, hub_cloned, topic_hub (capped)
    hubs: list[str] = field(default_factory=list)    # clone ids of the non-topic hubs cloned (their cap: "hub cap")


class Rejected(Exception):
    """The twin cannot be planted under the rules above; the message is the drop reason (``CLASS_TO_CLONE``)."""


_RUNS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()   # graph -> (id -> CVT?, relation -> CVT-valued?)


def _known(is_cvt: Callable[[str], bool]) -> tuple[dict[str, bool], dict[str, bool]]:
    """What the run learnt of ``is_cvt``, by id and by relation, kept while its graph lives."""
    try:
        return _RUNS.setdefault(getattr(is_cvt, "__self__", is_cvt), ({}, {}))
    except TypeError:
        return {}, {}


def _asked(is_cvt: Callable[[str], bool], ids: dict[str, bool], id: str) -> bool:
    if id not in ids:
        ids[id] = is_cvt(id)
    return ids[id]


def is_id(value: str) -> bool:
    return bool(_ID.fullmatch(value))


def is_closed(types: list[str], closed_class: set[str]) -> bool:
    return any(t in closed_class for t in types)


def fresh_ids(start: int = 1) -> Callable[[], str]:
    """``m.x`` and a base-36 counter: ``m.x1`` … ``m.x9``, ``m.xa`` … ``m.xz``, ``m.x10`` …"""
    counter = itertools.count(start)
    return lambda: "m.x" + _base36(next(counter))


def _base36(n: int) -> str:
    digits = _BASE36[n % 36]
    while n >= 36:
        n //= 36
        digits = _BASE36[n % 36] + digits
    return digits


def _value(term: str, row: dict[str, str]) -> str | None:
    """A pattern term in one row: the variable's binding (None if unbound), a bare ``ns:`` id, or a literal's text."""
    if term.startswith("?"):
        return row.get(term)
    if term.startswith("ns:"):
        return term[3:]
    quoted = _QUOTED_TERM.fullmatch(term)
    return quoted.group(1) if quoted else term


def _is_literal(term: str, value: str, relation: str) -> bool:
    """Whether a pattern term's value is a literal: quoted, or not a mid and not a schema id."""
    if _QUOTED_TERM.fullmatch(term):
        return True
    return not is_id(value) and not term.startswith("ns:") and relation not in CLASS_RELATIONS


def _instantiate(gold: Gold, rows: list[dict[str, str]]) -> tuple[list[Triple], list[Triple]]:
    """Every triple the rows instantiate, split into edges and literals, each once."""
    edges, literals = {}, {}
    for row in rows:
        for (s, p, o), markers in gold.scoped:
            triple = (_value(s, row), _value(p, row), _value(o, row))
            if None in triple or not all(m in row for m in markers):
                continue
            (literals if _is_literal(o, triple[2], triple[1]) else edges)[triple] = None
    return list(edges), list(literals)


def path_from(gold: Gold, rows: list[dict[str, str]], topic_ids: set[str], is_cvt: Callable[[str], bool],
              types_of: Callable[[str], list[str]]) -> Path:
    edges, literals = _instantiate(gold, rows)
    constants = [t[3:] for triple, _ in gold.scoped for t in triple if t.startswith("ns:") and is_id(t[3:])]
    bound = [v for row in rows for v in row.values() if is_id(v)]
    answers = list(dict.fromkeys(row[gold.select] for row in rows if gold.select in row))
    nodes, known = {}, _known(is_cvt)[0]
    for id in dict.fromkeys(constants + bound):
        cvt = _asked(is_cvt, known, id)
        role = ("topic" if id in topic_ids else "answer" if id in answers else "constant" if id in constants
                else "mediator" if cvt else "intermediate")
        nodes[id] = Node(id, role, types_of(id), cvt)
    return Path(nodes, edges, literals, answers)


def _topics(q: dict) -> list[str]:
    """The question's topic ids: ``topic_entity_ids``, else the keys of ``topic_entity``."""
    return list(q.get("topic_entity_ids") or q.get("topic_entity") or {})


def _path_size(gold: Gold, rows: list[dict[str, str]]) -> int:
    """The path's node count before its per-node queries: the query's ids and every id a row binds."""
    constants = {t[3:] for triple, _ in gold.scoped for t in triple if t.startswith("ns:") and is_id(t[3:])}
    return len(constants | {v for row in rows for v in row.values() if is_id(v)})


def _one_type(types: list[str], out_relations: list[str], in_relations: list[str] = (),
              target_type: Callable[[str, str], str | None] | None = None) -> str | None:
    """A clone's one type: the commonest domain of its out-edges, else of its in-edges' targets."""
    specific = {t for t in types if not t.startswith(GENERIC_TYPES)}
    votes = collections.Counter(r.rsplit(".", 1)[0] for r in out_relations)
    if not votes.keys() & specific and target_type:
        votes = collections.Counter(target_type(r, "out") for r in in_relations)
    owned = [(-n, t) for t, n in votes.items() if t in specific]
    return min(owned)[1] if owned else min(specific, default=None)


def clone_path(path: Path, closed_class: set[str], fresh: Callable[[], str],
               target_type: Callable[[str, str], str | None] | None = None,
               keep: Callable[[str], bool] | None = None, *, hub: Callable[[str], bool] | None = None,
               stay_real: Callable[[str], bool] | None = None) -> Clone:
    """Clone the path's nodes under fresh ids, by the hub and direction rules; ``Rejected`` when they fail."""
    classes = {o for _, r, o in path.edges if r in CLASS_RELATIONS}
    classes |= {n.id for n in path.nodes.values() if is_closed(n.types, closed_class)}
    always = {n.id for n in path.nodes.values() if n.role == "topic" or n.is_cvt}
    candidates = [id for id in path.nodes if id not in always and id not in classes]
    kept_v0 = {id for id in candidates if keep is not None and keep(id)}
    stay = {id for id in candidates if id not in kept_v0 and stay_real is not None and stay_real(id)}
    cloned = set(path.nodes) - (classes - always) - kept_v0 - stay
    hubs = {id for id in path.nodes if id in cloned and id not in always and hub is not None and hub(id)}
    cloned -= hubs                                              # real while the path only points at them
    while pointing := {s for s, r, o in path.edges
                       if r not in CLASS_RELATIONS and o in cloned and s not in cloned and s not in kept_v0}:
        if pointing - stay - hubs:                              # a class, or a schema id the query writes
            raise Rejected(CLASS_TO_CLONE)
        cloned |= pointing                                      # a kept node or a hub would point at a clone
    mapping = {id: fresh() for id in path.nodes if id in cloned}
    flags = ["hub_kept_real"] if hubs - cloned else []
    flags += ["hub_cloned"] if hubs & cloned else []
    edges, kept = [], []
    for s, r, o in path.edges:
        cloned_o = r not in CLASS_RELATIONS and o in mapping
        if s in mapping or cloned_o:
            triple = (mapping.get(s, s), r, mapping[o] if cloned_o else o)
            (edges if s in mapping and cloned_o else kept).append(triple)
    out, inward = collections.defaultdict(list), collections.defaultdict(list)
    for s, r, o in path.edges:
        out[s].append(r)
        if r not in CLASS_RELATIONS:
            inward[o].append(r)
    return Clone(mapping, edges, kept,
                 literals=[(mapping[s], p, v) for s, p, v in path.literals if s in mapping],
                 answers=[mapping.get(a, a) for a in path.answers],
                 types={c: t for id, c in mapping.items()
                        if (t := _one_type(path.nodes[id].types, out[id], inward[id], target_type))},
                 source_types={c: list(path.nodes[id].types) for id, c in mapping.items()},
                 flags=flags, hubs=[mapping[id] for id in path.nodes if id in hubs & cloned])


# step 3: dates and numbers moved by one order-preserving transform
DRAWS = 200
SHIFTS = [d for d in range(-40, 41) if d]          # whole years
SCALE = (0.5, 2.0)
KINDS = {"date": "date", "int": "number", "float": "number"}
DATE_DATATYPES = {"xsd:date", "xsd:datetime", "xsd:gyear", "xsd:gyearmonth"}
_DATE = re.compile(r"(-?\d{4})(?:-(\d{2}))?(?:-(\d{2}))?((?:T[\d:.]+)?(?:Z|[+-]\d{2}:\d{2})?)")
_NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)")
_BINDING = re.compile(r"ns:([\w.]+)\s+(\?\w+)")             # a pattern inside a FILTER binding a variable
_CLAUSE = re.compile(r"&&|\|\|")
_DIFFERENCE = re.compile(r"(?:\)|\?\w+)\s*[-+]\s*[\w?(]")      # A - B, A + B


@dataclass
class Resampled:
    values: dict[tuple[str, str, str], str] = field(default_factory=dict)   # (subject, property, old) -> new
    query_literals: dict[int, str] = field(default_factory=dict)        # Literal.position -> new text
    question_literals: dict[str, str] = field(default_factory=dict)     # surface in the question -> new surface
    ok: bool = True
    reason: str | None = None
    shift: int | None = None        # the transform drawn (None when nothing moved): dates move ``shift`` whole years,
    scale: float | None = None      # numbers scale by ``scale``; ``pad_source`` moves padded values with it


def _parts(text: str) -> tuple[int, ...]:
    """A date's year, month and day, as far as it has them."""
    return tuple(int(g) for g in _DATE.match(text).groups()[:3] if g)


def _key(group: str) -> Callable[[str], tuple | float]:
    """The sort key of a group's texts: a date's (year, month, day, time), missing parts 0; a number's value."""
    if group == "number":
        return float
    return lambda text: (_parts(text) + (0, 0))[:3] + (_DATE.fullmatch(text)[4],)


def _parses(text: str, group: str) -> bool:
    return bool((_DATE if group == "date" else _NUMBER).fullmatch(text))


def _readable(value: str, datatype: str | None, group: str) -> bool:
    """A constant reads as its group: the text in the group's shape, a date datatype on a date only."""
    return _parses(value, group) and (datatype is None or (datatype.lower() in DATE_DATATYPES) == (group == "date"))


def _shift(text: str, delta: int) -> str | None:
    """The date ``delta`` years on, None if that puts February 29 in a common year."""
    m = _DATE.fullmatch(text)
    year = int(m[1]) + delta
    if (m[2], m[3]) == ("02", "29") and not calendar.isleap(year):
        return None
    return ("-" if year < 0 else "") + f"{abs(year):04d}" + text[len(m[1]):]


def _scale(text: str, s: float) -> str:
    decimals, x = len(text.partition(".")[2]), float(text) * s
    return f"{x:.{decimals}f}" if decimals else str(math.floor(x + 0.5))


def _inside(text: str, bounds: LiteralRange) -> bool:
    if bounds.kind != "date":
        return bounds.lo <= float(text) <= bounds.hi
    v, lo, hi = _parts(text), _parts(str(bounds.lo)), _parts(str(bounds.hi))
    return lo[:len(v)] <= v[:len(lo)] and v[:len(hi)] <= hi[:len(v)]


def _order_kept(pairs: list[tuple[str, str]], key: Callable) -> bool:
    """Neighbours in the old order stay strictly ordered, or equal, as they were: so every pairwise order holds."""
    keyed = sorted((key(old), key(new)) for old, new in pairs)
    return all(a[1] < b[1] if a[0] < b[0] else a[1] == b[1] for a, b in zip(keyed, keyed[1:]))


def _holds(items: list[tuple[str, str | None, str, LiteralRange | None]]) -> bool:
    """Whether every new value exists, lies in its range and keeps every order within its group."""
    if any(new is None or (bounds is not None and not _inside(new, bounds)) for _, new, _, bounds in items):
        return False
    return all(_order_kept([(old, new) for old, new, g, _ in items if g == group], _key(group))
               for group in ("date", "number"))


def _constants(gold: Gold, groups: dict[str, str]) -> dict[tuple[str, str | None], str] | None:
    """The query constants that move, ``(value, datatype) -> group``; None if one cannot be read as its kind."""
    bound = {o: groups[p.removeprefix("ns:")] for _, p, o in gold.triples if p.removeprefix("ns:") in groups}
    literals = {(lit.value, lit.datatype) for lit in gold.literals}     # the lang filter's '' and 'en' are not
    found = [((q[2], q[3]), bound[o]) for _, _, o in gold.triples if o in bound and (q := _QUOTED.fullmatch(o))]
    for text in gold.filters:
        local = {**bound, **{v: groups[p] for p, v in _BINDING.findall(text) if p in groups}}
        bare = _QUOTED.sub(lambda m: " " * len(m[0]), text)
        cuts = [0] + [m.end() for m in _CLAUSE.finditer(bare)] + [len(text)]
        for a, b in zip(cuts, cuts[1:]):
            if named := {local[v] for v in _VARIABLE.findall(bare, a, b) if v in local}:
                group = named.pop() if len(named) == 1 else None      # None: a date compared with a number
                found += [((m[2], m[3]), group) for m in _QUOTED.finditer(text, a, b)]
                if not _DIFFERENCE.search(bare, a, b):
                    found += [((m[0], None), group) for m in _BARE.finditer(bare, a, b)]
    out: dict[tuple[str, str | None], str] = {}
    for constant, group in found:
        if constant in literals and (group is None or not _readable(*constant, group)
                                     or out.setdefault(constant, group) != group):
            return None
    return out


def _inferred(texts: list[str]) -> Callable:
    """The key of values no moving range describes: dates if all read as dates, else numbers, else their text."""
    group = next((g for g in ("date", "number") if all(_parses(t, g) for t in texts)), None)
    return _key(group) if group else str


def _tied(gold: Gold, path: Path, groups: dict[str, str]) -> bool:
    """The ordered property's values tie at the LIMIT's cut: the graph does not decide the superlative."""
    if not (gold.order and gold.limit):
        return False
    prop = next((p.removeprefix("ns:") for _, p, o in gold.triples if o == gold.order[0]), None)
    texts = [v for _, p, v in path.literals if p == prop]
    key = _key(groups[prop]) if prop in groups else _inferred(texts)
    ranked = sorted(map(key, texts), reverse=gold.order[1] == "DESC")
    return len(ranked) > gold.limit and ranked[gold.limit - 1] == ranked[gold.limit]


def _surface_pairs(value: str, new: str, group: str) -> list[tuple[str, str]]:
    if group == "date":
        return [(_DATE.fullmatch(value)[1], _DATE.fullmatch(new)[1])]
    grouped = [(f"{int(value):,}", f"{int(new):,}")] if re.fullmatch(r"-?[1-9]\d{3,}", value) else []
    return [(value, new)] + grouped


def _mentions(question: str, surface: str) -> bool:
    return bool(re.search(rf"(?<!\d)(?<!\d[.,]){re.escape(surface)}(?!\d|[.,]\d)", question))


def _reject(reason: str) -> Resampled:
    return Resampled(ok=False, reason=reason)


def resample(gold: Gold, path: Path, ranges: dict[str, LiteralRange], rng: random.Random, question: str,
             cloned: set[str] | None = None) -> Resampled:
    real = {p for s, p, _ in path.literals if cloned is not None and s not in cloned}
    groups = {p: KINDS[ranges[p].kind] for _, p, _ in path.literals
              if p in ranges and ranges[p].kind in KINDS and p not in real}
    moved = [(s, p, v) for s, p, v in path.literals if p in groups]
    if not all(_parses(v, groups[p]) for _, p, v in moved):
        return _reject("a literal not of its property's kind")
    if _tied(gold, path, groups):
        return _reject("tie at the superlative")
    if not moved:
        return Resampled()
    constants = _constants(gold, groups)
    if constants is None:
        return _reject("constant kind mismatch")
    for _ in range(DRAWS):
        delta, s = rng.choice(SHIFTS), rng.uniform(*SCALE)
        move = {"date": lambda t: _shift(t, delta), "number": lambda t: _scale(t, s)}
        values = {(subject, p, v): move[groups[p]](v) for subject, p, v in moved}
        new = {c: move[group](c[0]) for c, group in constants.items()}
        items = [(v, values[subject, p, v], groups[p], ranges[p]) for subject, p, v in moved]
        if s != 1.0 and _holds(items + [(c[0], new[c], group, None) for c, group in constants.items()]):
            break
    else:
        return _reject("no transform in range")
    question_literals: dict[str, str] = {}
    for c, group in constants.items():
        for old, text in _surface_pairs(c[0], new[c], group):
            if _mentions(question, old) and question_literals.setdefault(old, text) != text:
                return _reject("one surface, two new values")
    return Resampled(values, {lit.position: new[lit.value, lit.datatype] for lit in gold.literals
                              if (lit.value, lit.datatype) in new}, question_literals, shift=delta, scale=s)


# step 4: every clone padded in its source's shape
def pad_source(clone: Clone, out_edges: Callable[[str, int], dict[str, tuple[int, list[str]]]],
               pick_target: Callable[[str], str | None], rng: random.Random, *, fresh: Callable[[], str],
               is_cvt: Callable[[str], bool], types_of: Callable[[str], list[str]],
               target_type: Callable[[str, str], str | None], literal_range: Callable[[str], LiteralRange | None],
               cap: int, objects: Callable[[str, str], list[str]], shift: int | None = None,
               scale: float | None = None) -> list[Triple]:
    """Give every clone its source's out-edges in the same counts; the added edges."""

    def move(literals: list[tuple[str, str, str, LiteralRange]]) -> dict[tuple[str, str], str | None]:
        """(relation, value) -> its new value under the question's transform, else one drawn; None out of range."""
        new: dict[tuple[str, str], str | None] = {}
        fixed = shift is not None and scale is not None
        for _ in range((1 if fixed else DRAWS) if literals else 0):
            delta, factor = (shift, scale) if fixed else (rng.choice(SHIFTS), rng.uniform(*SCALE))
            for r, o, group, bounds in literals:
                v = _shift(o, delta) if group == "date" else _scale(o, factor)
                new[r, o] = v if v is not None and _inside(v, bounds) else None
            if None not in new.values():
                break
        return new

    read: dict[str, dict[str, tuple[int, list[str]]]] = {}
    whole: dict[tuple[str, str], set[str]] = {}            # (capped source, relation) -> every object it has

    def edges_of(id: str) -> dict[str, tuple[int, list[str]]]:
        if id not in read:
            read[id] = out_edges(id, cap)
        return read[id]
    source = {c: s for s, c in clone.mapping.items()}
    on_path = collections.defaultdict(set)             # (clone, relation) -> the source objects the path holds
    for s, r, o in clone.edges + clone.kept_edges:
        on_path[s, r].add(source.get(o, o))
    for s, p, v in clone.literals:
        on_path[s, p].add(v)
    read_real = {o for _, _, o in clone.kept_edges}     # real nodes the path reads: never a fresh target
    have = set(clone.edges) | set(clone.kept_edges)
    ids, valued = _known(is_cvt)                        # the run's answers by id, and by relation
    padded_cvts: set[str] = set()                       # clone CVTs made here: their own CVTs are not replicated
    cvt_relations: set[tuple[str, str]] = set()         # (clone, relation) CVT-valued
    lits = set(clone.literals)
    added: list[Triple] = []
    queue = list(clone.mapping.values())
    while queue:
        cid = queue.pop(0)
        real = edges_of(source[cid])
        quota = share({r: n for r, (n, _) in real.items()}, cap)
        capped = sum(n for n, _ in real.values()) > cap
        if capped and cid not in clone.hubs and "topic_hub" not in clone.flags:
            clone.flags.append("topic_hub")
        plan: list[tuple[str, str]] = []                # (relation, object) to pad, in order
        for r, (count, sample) in sorted(real.items()):
            held = len(on_path[cid, r])
            target = max(held, quota[r]) if capped else count
            if count > target:
                clone.unpadded[cid, r, "hub cap" if cid in clone.hubs else "topic hub cap"] += count - target
            todo = [o for o in sample if o not in on_path[cid, r]][:max(0, target - held)]
            new = [o for o in todo if is_id(o) and o not in clone.mapping and o != source[cid]]
            if new and r not in valued:
                valued[r] = _asked(is_cvt, ids, new[0])
            if new and valued[r]:
                linked = sum(1 for o in todo if o in clone.mapping)       # path CVTs and other clones' count first
                room = 0 if cid in padded_cvts else max(0, CVT_CAP - held - linked)
                drawn = set(new) if len(new) <= room else set(rng.sample(new, room))
                if len(new) > len(drawn):
                    clone.unpadded[cid, r, "cvt depth" if cid in padded_cvts else "cvt cap"] += len(new) - len(drawn)
                todo = [o for o in todo if o in drawn or o not in new]
                cvt_relations.add((cid, r))
            plan += [(r, o) for o in todo]
        movable = []
        for r in sorted({r for r, o in plan if not is_id(o)}):
            bounds = literal_range(r)
            group = {"date": "date", "int": "number", "float": "number"}.get(bounds.kind) if bounds else None
            movable += [(r, o, group, bounds) for r2, o in plan if r2 == r and not is_id(o) and group
                        and _parses(o, group)]
        moved = move(movable)
        for r, o in plan:
            if not is_id(o):
                if (r, o) not in moved:
                    string = not _parses(o, "number") and not _parses(o, "date")
                    clone.unpadded[cid, r, "string" if string else "no range"] += 1
                elif moved[r, o] is None:
                    clone.unpadded[cid, r, "out of range"] += 1
                elif (cid, r, moved[r, o]) in lits:
                    clone.unpadded[cid, r, "duplicate"] += 1
                else:
                    lits.add((cid, r, moved[r, o]))
                    clone.literals.append((cid, r, moved[r, o]))
                continue
            if o == source[cid]:
                clone.unpadded[cid, r, "self-loop"] += 1
                continue
            if o in clone.mapping:                                        # an edge among clones
                target = clone.mapping[o]
            elif (cid, r) in cvt_relations:                               # a CVT substructure: a clone CVT
                target = clone.mapping[o] = fresh()
                source[target] = o
                padded_cvts.add(target)
                clone.source_types[target] = list(types_of(o))
                if t := _one_type(clone.source_types[target], list(edges_of(o))):
                    clone.types[target] = t
                queue.append(target)
            else:
                if capped and (source[cid], r) not in whole:
                    whole[source[cid], r] = set(objects(source[cid], r))
                own = whole[source[cid], r] if capped else real[r][1]
                target = _draw(r, o, own, cid, clone.mapping, read_real, have, pick_target, target_type, types_of)
            triple = (cid, r, target)
            if target is None or triple in have:
                clone.unpadded[cid, r, "no target"] += 1
                continue
            have.add(triple)
            added.append(triple)
    clone.edges.extend(added)
    return added


def _draw(relation: str, object: str, objects: list[str] | set[str], cid: str, mapping: dict[str, str],
          read_real: set[str], have: set[Triple], pick_target: Callable[[str], str | None],
          target_type: Callable[[str, str], str | None], types_of: Callable[[str], list[str]]) -> str | None:
    """A fresh real node of the relation's target type that the clone and its path do not hold; or None."""
    type = target_type(relation, "out") or _one_type(types_of(object), [])
    if type is None:
        return None
    for _ in range(MAX_MISSES):
        target = pick_target(type)
        if target is None:
            return None
        if (target not in objects and target not in mapping and target not in read_real
                and (cid, relation, target) not in have):
            return target
    return None


def pad(clone: Clone, profiles: dict[str, TypeProfile], pick_target: Callable[[str], str],
        rng: random.Random) -> list[Triple]:
    """The older type-profile padding (Freebase pads with ``pad_source``)."""
    have = set(clone.edges) | set(clone.kept_edges)
    degree = collections.Counter(n for s, _, o in clone.edges + clone.kept_edges for n in (s, o))
    own = collections.defaultdict(set)              # each node's relations on the path, before any padding
    for s, r, o in have:
        own[s].add(r)
        own[o].add(r)
    added: list[Triple] = []
    for cid, type in clone.types.items():
        profile = profiles.get(type)
        choices = sorted((k, n) for k, n in profile.edges.items()
                         if k[2] is not None and k[0] not in own[cid]) if profile else []
        if not choices:
            continue
        keys, weights = zip(*choices)
        misses, limit = 0, min(profile.degree_q1, profile.degree_q3)
        while degree[cid] < limit and misses < MAX_MISSES:
            relation, direction, target_type = rng.choices(keys, weights)[0]
            target = pick_target(target_type)
            triple = (cid, relation, target) if direction == "out" else (target, relation, cid)
            if target is None or target == cid or triple in have:
                misses += 1
                continue
            have.add(triple)
            added.append(triple)
            degree.update((triple[0], triple[2]))
    clone.edges.extend(added)
    return added


# step 6: the twin record: the question and its gold rewritten
GOLD_FIELDS = ("type", "projection", "answer_kind", "shape", "hops", "constraints")   # copied; masked text is not
REWRITTEN = ("hops", "constraints")                    # their strings take the query's substitutions
QUESTION_WORDS = ("what", "which", "who", "whom", "whose", "when", "where", "why", "how")
GENERATOR = "data/extension"
Span = tuple[int, int]
_CAP = (rf"(?<![\w'’-])(?!(?i:{'|'.join(QUESTION_WORDS)})(?![\w'’-]))"
        r"[A-ZÀ-ÖØ-Þ](?:\.(?=\s)|[\w'’-]*)")                      # "K." of "James K. Polk" keeps its period
_RUN = re.compile(rf"{_CAP}(?:[ \t]+{_CAP})*")
_OPENERS = " \t\"'“‘(["                                             # what may stand before a sentence's first word
_APOSTROPHES = ("'", "’")
_MID = r"(?<![\w.])[mg]\.[0-9a-z_]+(?!\w)"
_FULL_DATE = re.compile(r"-?\d{4}-\d{2}")                         # a date past its year: only the year is written
TOPIC_OFF_PATH = "topic not on path"
TOPIC_UNMENTIONED = "topic mention not found"
CONSTANT_UNMENTIONED = "constant mention not found"
LITERAL_UNMAPPED = "moved literal surface not mapped"


def _mention(text: str) -> re.Pattern:
    return re.compile(rf"(?<!\w){re.escape(text)}(?!\w)", re.I)


def _surfaces(surfaces) -> re.Pattern:
    """Any of ``surfaces`` (longest first) as a whole word outside a longer number."""
    alternation = "|".join(map(re.escape, sorted(surfaces, key=len, reverse=True)))
    return re.compile(rf"(?<!\w)(?<!\d[.,])(?:{alternation})(?!\w)(?![.,]\d)")


def _free(start: int, end: int, taken) -> bool:
    return all(end <= a or b <= start for a, b, *_ in taken)


def locate_mentions(question: str, labels: dict[str, list[str]]) -> dict[str, list[Span]]:
    """Each source's mentions: every occurrence of its label or an alias, longest label first over all sources."""
    candidates = sorted(((label, source) for source, names in labels.items() for label in names if label.strip()),
                        key=lambda candidate: -len(candidate[0]))
    taken: list[Span] = []
    found: dict[str, list[Span]] = collections.defaultdict(list)
    for label, source in candidates:
        for m in _mention(label).finditer(question):
            if _free(*m.span(), taken):
                taken.append(m.span())
                found[source].append(m.span())
    return {source: sorted(spans) for source, spans in found.items()}


def _possessive(question: str, start: int, end: int, name: str) -> tuple[int, str]:
    """The span and name past a bare possessive apostrophe (``Collins'`` -> ``Brennick's``)."""
    mark, after = question[end:end + 1], question[end + 1:end + 2]
    if (mark in _APOSTROPHES and question[end - 1] in "sS" and question[start - 1:start] not in _APOSTROPHES
            and not (after.isalnum() or after == "_") and not name.endswith(("s", "S"))):
        return end + 1, name + mark + "s"
    return end, name


def rewrite_question(question: str, spans: dict[str, list[Span]], new_names: dict[str, str],
                     literal_map: dict[str, str]) -> str:
    """Each span becomes its source's new name, then each literal surface outside them its new surface."""
    lower = question == question.lower()
    edits: list[tuple[int, int, str]] = []
    for start, end, name in sorted(((s, e, new_names[source]) for source, found in spans.items() for s, e in found),
                                   key=lambda edit: (edit[0], -edit[1])):
        if _free(start, end, edits):
            edits.append((start, *_possessive(question, start, end, name.lower() if lower else name)))
    if literal_map:
        edits += [(m.start(), m.end(), literal_map[m[0]]) for m in _surfaces(literal_map).finditer(question)
                  if _free(*m.span(), edits)]
    for start, end, new in sorted(edits, reverse=True):
        question = question[:start] + new + question[end:]
    return question


def _unresolved(text: str, known: list[str]) -> list[str]:
    """Capitalised runs of ``text`` once every ``known`` name is blanked, each once, in order of appearance."""
    for name in sorted(filter(str.strip, known), key=len, reverse=True):
        text = _mention(name).sub(lambda m: "\0" * len(m[0]), text)
    found = []
    for m in _RUN.finditer(text):
        before = text[:m.start()].rstrip(_OPENERS)
        if len(m[0].split()) > 1 or (before and before[-1] not in ".?!"):
            found.append(m[0])
    return list(dict.fromkeys(found))


def _gold_rewriter(query: str, mapping: dict[str, str], query_literals: dict[int, str]):
    """The query's substitutions on any text, in one pass: a mid whole-token, a moved literal's old text."""
    moved: dict[str, str] = {}
    for lit in parse(query).literals:
        if lit.position in query_literals:
            moved.setdefault(lit.value, query_literals[lit.position])
    pattern = re.compile(rf"(?P<mid>{_MID})" + (f"|{_surfaces(moved).pattern}" if moved else ""))
    return lambda text: pattern.sub(lambda m: mapping.get(m[0], m[0]) if m["mid"] else moved[m[0]], text)


def _deep(value, sub):
    """``value`` with ``sub`` applied to every string in it, lists and dicts copied."""
    if isinstance(value, str):
        return sub(value)
    if isinstance(value, list):
        return [_deep(v, sub) for v in value]
    if isinstance(value, dict):
        return {k: _deep(v, sub) for k, v in value.items()}
    return value


def _answer(source: str, clone: Clone, names: dict[str, str], res: Resampled,
            kept_labels: dict[str, str]) -> tuple[str, str]:
    """One source answer as (name, id): its clone, itself if kept real, else its literal re-sampled (or kept)."""
    if source in clone.mapping:
        return names[clone.mapping[source]], clone.mapping[source]
    if is_id(source):
        return kept_labels[source], source
    new = next((value for (_, _, old), value in res.values.items() if old == source), source)
    return new, new


def rejection(q: dict, clone: Clone, spans: dict[str, list[Span]], labels: dict[str, list[str]]) -> str | None:
    """Why the twin cannot name what its gold reads, or None."""
    if any(t not in clone.mapping for t in q["topic_entity_ids"]):
        return TOPIC_OFF_PATH
    if any(t not in spans for t in q["topic_entity_ids"]):
        return TOPIC_UNMENTIONED
    if set(labels) - set(spans):
        return CONSTANT_UNMENTIONED
    return None


def _shown(value: str) -> tuple[set[str], str]:
    """A query constant's mapped surfaces and the digits any written form of it shows."""
    surfaces = {value}
    if date := _DATE.fullmatch(value):
        surfaces.add(date[1])
    if re.fullmatch(r"-?[1-9]\d{3,}", value):
        surfaces.add(f"{int(value):,}")
    digits = date[1].lstrip("-") if date and _FULL_DATE.match(value) else re.sub(r"\D", "", value).strip("0")
    return surfaces, digits


def unmapped_literals(question: str, query: str, res: Resampled) -> tuple[list[str], list[str]]:
    """The moved query constants no mapped surface covers: shown by the question, and not."""
    digits = re.sub(r"(?<=\d)[.,](?=\d)", "", question)
    unmapped, absent = [], []
    for value in dict.fromkeys(lit.value for lit in parse(query).literals if lit.position in res.query_literals):
        surfaces, shown = _shown(value)
        if not surfaces & set(res.question_literals):
            (unmapped if shown and shown in digits else absent).append(value)
    return unmapped, absent


def twin(q: dict, clone: Clone, names: dict[str, str], res: Resampled, labels: dict[str, list[str]],
         kept_labels: dict[str, str], split: str | None = None, unnamed: Sequence[str] = ()) -> dict | None:
    """The twin record of source ``q``, or None when ``rejection`` or ``unmapped_literals`` refuses it."""
    spans = locate_mentions(q["question"], labels)
    unmapped, absent = unmapped_literals(q["question"], q["gold"]["query"], res)
    if rejection(q, clone, spans, labels) or unmapped:
        return None
    question = rewrite_question(q["question"], spans, {s: names[clone.mapping[s]] for s in spans},
                                res.question_literals)
    answers = [_answer(a, clone, names, res, kept_labels) for a in q["answer_ids"]]
    clones = [names[c] for c in clone.mapping.values() if c in names]
    left = _unresolved(question, clones + list(kept_labels.values()))
    flags = [flag for flag, raised in [
        ("mention_repeated", any(len(found) > 1 for found in spans.values())),
        ("literal_surface_repeated", any(len(_surfaces([s]).findall(q["question"])) > 1
                                         for s in res.question_literals)),
        ("constant_not_in_question", bool(absent or unnamed)),
        ("unresolved_text_entities", bool(left))] if raised]
    record = copy.deepcopy(q)
    if "anchor_kinds" in q:                            # keyed by topic id: the clones' now
        record["anchor_kinds"] = {clone.mapping.get(k, k): v for k, v in q["anchor_kinds"].items()}
    topic_entity = {clone.mapping[t]: names[clone.mapping[t]] for t in q["topic_entity_ids"]}   # every one cloned
    sub = _gold_rewriter(q["gold"]["query"], clone.mapping, res.query_literals)
    return {**record, "id": f"{q['id']}_unseen", "twin_id": q["id"], "question": question,
            "answers": [name for name, _ in answers], "answer_ids": [id for _, id in answers],
            "topic_entity": topic_entity, "topic_entity_ids": [clone.mapping[t] for t in q["topic_entity_ids"]],
            "gold": {"query": substitute(q["gold"]["query"], clone.mapping, res.query_literals),
                     **{k: _deep(q["gold"][k], sub) if k in REWRITTEN else copy.deepcopy(q["gold"][k])
                        for k in GOLD_FIELDS}},
            "clone_ids": sorted(clone.mapping.values()),
            "answer_class": "closed" if q["answer_ids"] and all(a not in clone.mapping and is_id(a)
                                                                for a in q["answer_ids"]) else "open",
            "text_entities_unresolved": left, "flags": flags,
            "provenance": {"source_split": q.get("split") or split, "generator": GENERATOR, "resample_ok": res.ok}}
