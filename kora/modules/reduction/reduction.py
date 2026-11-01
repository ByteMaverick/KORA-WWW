from __future__ import annotations

import operator
import re
from collections.abc import Callable

from kora.provenance.models import EntityRef
from kora.modules.reduction.models import NO, OPERATIONS, YES, Operate
from kora.tools.kg import value_of

Lookup = Callable[..., list[str]]
_DATE, _YEAR = re.compile(r"^-?\d{4}-\d{2}"), re.compile(r"^-?\d{4}")
_CMP = {"eq": operator.eq, "ne": operator.ne, "lt": operator.lt, "le": operator.le, "gt": operator.gt, "ge": operator.ge}

def lookup_on(kg, embedder=None, anchors: tuple[str, ...] = ()) -> Lookup:
    cache: dict[tuple[str, str, bool], list[str]] = {}
    named: dict[tuple[str, str], list[str]] = {}

    def read(entity_id: str, prop: str) -> list[str]:
        return kg.value_of(entity_id, prop) if hasattr(kg, "value_of") else value_of(kg, entity_id, prop)

    def fits(values: list[str], want) -> bool:
        return bool(values) and (want is None or want(values))

    def on(entity_id: str, prop: str, want) -> list[str]:
        found = read(entity_id, prop)
        if fits(found, want) or ("." in prop and " " not in prop):
            return found if fits(found, want) else []
        for candidate in properties_named(entity_id, prop):
            if fits(found := read(entity_id, candidate), want):
                return found
        return []

    def properties_named(entity_id: str, words: str) -> list[str]:
        if (entity_id, words) not in named:
            try:
                entries = list(kg.properties(entity_id))
            except Exception:
                entries = []
            candidates = [e.predicate for e in entries]
            named[(entity_id, words)] = list(dict.fromkeys(_closest(words, candidates, embedder)))
        return named[(entity_id, words)]

    def lookup(entity_id: str, prop: str, want=None) -> list[str]:
        key = (entity_id, prop, want is not None)
        if prop and key not in cache:
            found = on(entity_id, prop, want)
            for mediator in [] if found else between(kg, anchors, entity_id):
                if found := on(mediator, prop, want):
                    break
            cache[key] = found
        return cache.get(key, [])

    lookup.label = getattr(kg, "label", lambda _: None)
    return lookup

RANGE = {"end": "to", "ended": "to", "ends": "to", "ending": "to", "until": "to", "left": "to", "finished": "to",
         "start": "from", "started": "from", "starting": "from", "began": "from", "begin": "from", "since": "from",
         "joined": "from", "appointed": "from"}


def between(kg, anchors: tuple[str, ...], entity_id: str) -> list[str]:
    if not anchors or not hasattr(kg, "ask"):
        return []
    found: list[str] = []
    for anchor in anchors:
        if anchor == entity_id:
            continue
        rows = kg.ask(f"PREFIX ns: <http://rdf.freebase.com/ns/> SELECT DISTINCT ?m WHERE {{ ns:{anchor} ?p ?m . "
                      f"?m ?q ns:{entity_id} . FILTER(?m != ns:{entity_id} && ?m != ns:{anchor}) }} LIMIT 5")
        for row in rows or []:
            value = row.get("m") if isinstance(row, dict) else None
            value = value.get("value") if isinstance(value, dict) else value
            if isinstance(value, str) and value:
                found.append(value.rsplit("/", 1)[-1])
    return list(dict.fromkeys(found))


def _closest(words: str, predicates: list[str], embedder, keep: int = 3, floor: float = 0.45) -> list[str]:
    if not predicates:
        return []
    shared = [(_overlap(words, p), p) for p in predicates]
    named = [p for n, p in sorted(shared, key=lambda pair: (-pair[0][0], -pair[0][1])) if any(n)][:keep]
    if named or embedder is None or not hasattr(embedder, "similarity"):
        return named
    scores = embedder.similarity(words, predicates)
    return [p for score, p in sorted(zip(scores, predicates), key=lambda pair: -pair[0]) if score >= floor][:keep]

def _overlap(words: str, predicate: str) -> tuple[int, int]:
    asked = [w for w in re.findall(r"[a-z0-9]+", words.lower()) if len(w) > 1]
    asked += [RANGE[w] for w in asked if w in RANGE]
    wanted, whole = {w[:5] for w in asked if len(w) > 2}, set(asked)
    words_of = re.findall(r"[a-z0-9]+", predicate.lower())
    return (len(whole & set(words_of)), len(wanted & {w[:5] for w in words_of}))

FALLBACK_CAP = 100

def operate(op: Operate, sets: dict[str, list[EntityRef]], lookup: Lookup, choose=None) -> list[EntityRef]:
    """`choose` picks the property once; None, or no `choose`, reads each member through `lookup`."""
    if op.op not in OPERATIONS:
        raise ValueError(f"unknown operation {op.op!r}; one of {', '.join(OPERATIONS)}")
    if not (op.over or op.named) or any(name not in sets for name in op.over):
        raise KeyError(f"{op.op} runs over {op.over}, and the sets held are {', '.join(sets)}")
    inputs = ([list(op.named)] if op.named else []) + [sets[name] for name in op.over]
    first, rest = inputs[0], [{e.id for e in s} for s in inputs[1:]]
    if op.op == "count":
        return [EntityRef(id=f"value:{len(first)}", label=str(len(first)), entity_type="value")]
    if op.op == "intersect":
        return [e for e in first if all(e.id in ids for ids in rest)]
    if op.op == "difference":
        if not rest:
            raise ValueError("difference needs the set to keep and at least one to subtract")
        return [e for e in first if not any(e.id in ids for ids in rest)]
    chosen = None
    if choose is not None and op.prop and op.cmp != "in_set" and any(e.entity_type != "value" for e in first):
        chosen = choose(op, first)
    members = sum(e.entity_type != "value" for e in first)
    if chosen is None and op.prop and op.cmp != "in_set" and members > FALLBACK_CAP:
        # Read member by member, a set this size is thousands of graph queries the clock cannot interrupt.
        raise ValueError(f"{op.prop!r} was not chosen among the members' properties, and reading it on each of "
                         f"{members} members one by one is past the {FALLBACK_CAP}-member limit")
    if op.op == "filter":
        return [e for e in first if _passes(e, op, sets, lookup, chosen)]
    if op.op == "verify":
        return [_verified(first, op, sets, lookup, chosen)]
    valued = [(max(found, key=_order_key), e) for e in first
              if (found := ([e.label or e.id] if e.entity_type == "value" else _orderable([e.label or e.id]))
                  or _orderable(lookup(e.id, op.prop, _literal) if chosen is None else chosen.get(e.id, [])))]
    if not valued:
        raise ValueError(f"{op.prop!r} orders nothing in {op.over[0] if op.over else 'the named things'}: read on each member, "
                         f"through the record route its page shows, and by the property's name")
    pick = max if op.op == "argmax" else min
    return [pick(valued, key=lambda pair: _order_key(pair[0]))[1]]

def _literal(values: list[str]) -> bool:
    return bool(_orderable(values))


def _verified(members: list[EntityRef], op: Operate, sets: dict[str, list[EntityRef]], lookup: Lookup,
              chosen: dict[str, list[str]] | None) -> EntityRef:
    if any(_passes(e, op, sets, lookup, chosen) for e in members):
        return YES
    if not any(_values(e, op, lookup, chosen) for e in members):
        raise ValueError(f"{op.prop!r} has no value on {', '.join(e.label or e.id for e in members[:5])}: "
                         f"the claim is neither verified nor refuted")
    return NO


def _values(entity: EntityRef, op: Operate, lookup: Lookup, chosen: dict[str, list[str]] | None) -> list[str]:
    if chosen is None:
        return lookup(entity.id, op.prop, _literal if _orderable([op.value]) else None)
    return chosen.get(entity.id, [])


def _passes(entity: EntityRef, op: Operate, sets: dict[str, list[EntityRef]], lookup: Lookup,
            chosen: dict[str, list[str]] | None = None) -> bool:
    values = _values(entity, op, lookup, chosen)
    if op.cmp == "in_set":
        if op.value not in sets:
            raise KeyError(f"no set named {op.value!r} to filter by")
        return bool(set(values) & {e.id for e in sets[op.value]})
    label = getattr(lookup, "label", None)
    if chosen is not None:
        # A chosen path holds literals only: a text value is its own name, never an id to label.
        named = [_plain(v) for v in values if not _orderable([v])] if op.cmp in ("eq", "ne") else []
    else:
        named = [_plain(label(v) or "") for v in values if not _orderable([v])] if label and op.cmp in ("eq", "ne") else []
    if op.cmp == "ne":
        return bool(values) and all(_compare(v, "ne", op.value) for v in values) and _plain(op.value) not in named
    return any(_compare(v, op.cmp, op.value) for v in values) or (op.cmp == "eq" and _plain(op.value) in named)

def _orderable(values: list[str]) -> list[str]:
    return [v for v in values if _number(v) is not None or _DATE.match(v or "")]

def _order_key(value: str) -> tuple[bool, float, str]:
    return (_number(value) is None, _number(value) or 0.0, value)

def _plain(text: str) -> str:
    return re.sub(r"[\u2010-\u2015\u2212]", "-", text or "").strip().casefold()

def _compare(value: str, cmp: str, against: str) -> bool:
    left, right = _number(value), _number(against)
    dated = _DATE.match(against or "") or re.fullmatch(r"-?\d{4}", against or "")
    if left is not None and right is not None and not dated:
        if cmp in ("eq", "ne"):
            return (abs(left - right) <= 1e-5 * max(1.0, abs(left), abs(right))) == (cmp == "eq")
        return _CMP[cmp](left, right)
    if dated and _YEAR.match(value or ""):
        n = min(len(value), len(against))
        return _CMP[cmp](value[:n], against[:n])
    if right is not None or dated:
        return False
    return _CMP[cmp](value, against)

def _number(text: str) -> float | None:
    try:
        return float(text)
    except (TypeError, ValueError):
        return None
