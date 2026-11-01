"""The query dialect every graph adapter executes: parse, modifiers, and the facts a result stands for."""

from __future__ import annotations

import re

_SELECT = re.compile(
    r"^\s*SELECT\s+(?P<vars>.+?)\s+WHERE\s*\{(?P<body>.*)\}"
    r"(?P<tail>(?:\s+ORDER\s+BY\s+\?\w+(?:\s+(?:ASC|DESC))?)?(?:\s+LIMIT\s+\d+)?)\s*$",
    re.IGNORECASE | re.DOTALL,
)
_MODIFIERS = re.compile(
    r"(?:ORDER\s+BY\s+(?P<var>\?\w+)(?:\s+(?P<dir>ASC|DESC))?)?\s*(?:LIMIT\s+(?P<limit>\d+))?",
    re.IGNORECASE,
)


def parse_modifiers(query: str) -> dict:
    """`{"order": ("?var", "DESC") | None, "limit": int | None}`."""
    match = _SELECT.match(query or "")
    empty = {"order": None, "limit": None}
    if not match:
        return empty
    tail = (match.group("tail") or "").strip()
    if not tail:
        return empty
    m = _MODIFIERS.fullmatch(tail)
    if not m:
        return empty
    order = (m.group("var"), (m.group("dir") or "ASC").upper()) if m.group("var") else None
    limit = int(m.group("limit")) if m.group("limit") else None
    return {"order": order, "limit": limit}
_COUNT = re.compile(r"COUNT\(\s*(\?\w+)\s*\)", re.IGNORECASE)


def count_variable(query: str) -> str | None:
    """`SELECT COUNT(?x) ...` -> "?x"; None when the query does not count."""
    match = _SELECT.match(query or "")
    if not match:
        return None
    found = _COUNT.search(match.group("vars"))
    return found.group(1) if found else None


_SEPARATOR = re.compile(r"\s+\.\s+|\n")
_TERM = re.compile(r'"([^"]*)"|(\S+)')


def terms_of(chunk: str) -> list[str]:
    """A pattern's terms, with quoted identifiers kept whole."""
    return [bare or quoted for quoted, bare in _TERM.findall(chunk)]


def quote_if_needed(term: str) -> str:
    """How a term is written in the dialect: quoted when it has a space."""
    if not term or term.startswith(("?", ":")) or " " not in term:
        return term
    return '"' + term.replace('"', "") + '"'

TYPE_PREDICATE = ":type"

COMPARISONS = {":lt": "<", ":le": "<=", ":gt": ">", ":ge": ">="}


def comparisons(patterns: list[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
    """`(variable, operator, literal)` for every comparison pattern."""
    return [(s, p, o) for s, p, o in patterns if p in COMPARISONS]


SET_PREDICATE = ":in"


def set_anchors(patterns: list[tuple[str, str, str]]) -> list[tuple[str, str]]:
    """`(variable, "@name")` for every set-anchor pattern."""
    return [(s, o) for s, p, o in patterns if p == SET_PREDICATE]


def beyond_the_page(
    patterns: list[tuple[str, str, str]], anchored_vars: set[str]
) -> set[tuple[str, str, str]]:
    """The traversal patterns that reach past the anchor's page."""
    traversals = without_type_constraints(patterns)
    anchored = set(anchored_vars)
    for subject, _, obj in traversals:
        if not subject.startswith("?") and obj.startswith("?"):
            anchored.add(obj)
        elif not obj.startswith("?") and subject.startswith("?"):
            anchored.add(subject)

    beyond: set[tuple[str, str, str]] = set()
    changed = True
    while changed:
        changed = False
        for pattern in traversals:
            subject, _, obj = pattern
            if pattern in beyond or not (subject.startswith("?") and obj.startswith("?")):
                continue
            if subject in anchored or obj in anchored:
                beyond.add(pattern)
                anchored.add(subject)
                anchored.add(obj)
                changed = True
    return beyond


def literal_kind(literal: str) -> str:
    """'number' when the literal parses as a float, 'string' otherwise."""
    try:
        float(literal)
    except ValueError:
        return "string"
    return "number"


def type_constraints(patterns: list[tuple[str, str, str]]) -> dict[str, list[str]]:
    """The `:type` patterns, grouped by the term they constrain."""
    found: dict[str, list[str]] = {}
    for subject, predicate, obj in patterns:
        if predicate == TYPE_PREDICATE:
            found.setdefault(subject, []).append(obj)
    return found


def property_reads(patterns: list[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
    """`?m :from ?d` -- a literal property of a record read into a variable."""
    return [(s, p[1:], o) for s, p, o in patterns
            if p.startswith(":") and p != TYPE_PREDICATE and p != SET_PREDICATE
            and p not in COMPARISONS and o.startswith("?")]


def property_absences(patterns: list[tuple[str, str, str]]) -> list[tuple[str, str]]:
    """`?m :to none` -- the record must NOT carry this property."""
    return [(s, p[1:]) for s, p, o in patterns
            if p.startswith(":") and p != TYPE_PREDICATE and p != SET_PREDICATE
            and p not in COMPARISONS and o.lower() == "none"]


def without_type_constraints(
    patterns: list[tuple[str, str, str]],
) -> list[tuple[str, str, str]]:
    """The patterns that actually traverse the graph."""
    return [p for p in patterns if not p[1].startswith(":")]


def parse_query(query: str) -> tuple[list[str], list[tuple[str, str, str]]] | None:
    """Split a query into selected variables and triple patterns."""
    match = _SELECT.match(query or "")
    if not match:
        return None
    counted = count_variable(query)
    variables = ([counted] if counted else
                 [v for v in match.group("vars").split() if v.startswith("?")])
    patterns: list[tuple[str, str, str]] = []
    for chunk in _SEPARATOR.split(match.group("body")):
        chunk = chunk.strip().rstrip(".").strip()
        if not chunk:
            continue
        parts = terms_of(chunk)
        if len(parts) != 3:
            return None
        patterns.append((parts[0], parts[1], parts[2]))
    return variables, patterns


def syntax_problem(query: str) -> str:
    """Why a query did not parse, naming the pattern at fault, so the next candidate can fix that pattern."""
    match = _SELECT.match(query or "")
    if not match:
        return "syntax: expected SELECT ?vars WHERE { subject predicate object . ... }"
    for chunk in _SEPARATOR.split(match.group("body")):
        chunk = chunk.strip().rstrip(".").strip()
        parts = terms_of(chunk)
        if chunk and len(parts) != 3:
            hint = (" -- a type constraint is `?x :type T`, on its own, "
                    "constraining the variable on its left"
                    if TYPE_PREDICATE in parts else
                    ' -- an identifier with spaces must be quoted: `"Black Like Me" HAS_GENRE ?x`'
                    if len(parts) > 3 else "")
            return (f"syntax: the pattern `{chunk}` has {len(parts)} terms; every pattern "
                    f"is exactly `subject predicate object`{hint}")
    return "syntax: expected SELECT ?vars WHERE { subject predicate object . ... }"


ANSWER_VARIABLE = "?v"


def bound_variables(patterns: list[tuple[str, str, str]]) -> set[str]:
    """The variables a query's patterns bind."""
    bound = {token for pattern in without_type_constraints(patterns) for token in pattern
             if token.startswith("?")}
    return bound | {variable for _, _, variable in property_reads(patterns)} \
        | {variable for variable, _ in set_anchors(patterns)}


def _select_clause(variables: list[str], original: str) -> str:
    counted = count_variable(original)
    if counted and counted in variables:
        return f"COUNT({counted})"
    return " ".join(variables)


def rebuild_query(variables: list[str], patterns: list[tuple[str, str, str]],
                  original: str) -> str:
    """The query text for these patterns, keeping the original's modifiers."""
    body = " . ".join(f"{quote_if_needed(a)} {p} {quote_if_needed(b)}" for a, p, b in patterns)
    mods = parse_modifiers(original)
    tail = ""
    if mods["order"]:
        tail += f" ORDER BY {mods['order'][0]} {mods['order'][1]}"
    if mods["limit"] is not None:
        tail += f" LIMIT {mods['limit']}"
    return f"SELECT {_select_clause(variables, original)} WHERE {{ {body} }}{tail}"


def output_variables(query: str) -> list[str]:
    """The variables the query selects, in the order it selects them."""
    parsed = parse_query(query)
    return list(parsed[0]) if parsed else []


def triples_from(query: str, rows: list[dict[str, str]]) -> list[tuple[str, str, str]]:
    """Reconstruct the facts a result stands for, as subject-relation-object."""
    parsed = parse_query(query)
    if parsed is None or count_variable(query):
        return []
    _, patterns = parsed
    patterns = without_type_constraints(patterns)

    seen: dict[tuple[str, str, str], None] = {}
    for row in rows:
        for subject, predicate, obj in patterns:
            fact = (row.get(subject, subject), row.get(predicate, predicate),
                    row.get(obj, obj))
            if any(term.startswith("?") for term in fact):
                continue  # a variable the query did not select; nothing to cite
            seen.setdefault(fact, None)
    return list(seen)
