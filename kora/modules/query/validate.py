from __future__ import annotations

import re
from collections import Counter

from kora.modules.query.models import Candidate, Context, Page
from kora.tools.dialect import (
    ANSWER_VARIABLE,
    TYPE_PREDICATE,
    beyond_the_page,
    bound_variables,
    count_variable,
    parse_modifiers,
    parse_query,
    property_absences,
    property_reads,
    rebuild_query,
    set_anchors,
    syntax_problem,
    type_constraints,
    without_type_constraints,
)

IDENTIFIER = re.compile(r"^[mg]\.[0-9a-z_]+$")


def validate(candidates: list[Candidate], page: Page, ctx: Context) -> list[Candidate]:
    checked = [_checked(c, page, ctx) for c in candidates]
    return checked


def _checked(candidate: Candidate, page: Page, ctx: Context) -> Candidate:
    if parse_query(candidate.query) is None:
        return _unparsed(candidate, candidate.query, [])
    query, answered = _answer_repaired(candidate.query)
    if parse_query(query) is None:
        return _unparsed(candidate, query, answered)
    query, renamed = _repaired(query, page)
    parsed = parse_query(query)
    if parsed is None:
        return _unparsed(candidate, query, answered + renamed)
    variables, patterns = parsed
    problems = (_relations(patterns, page) + _anchors(patterns, ctx) + _answer(variables, patterns)
                + _filters(query, variables, patterns) + _repeat(query, ctx) + _routes(patterns, page, ctx)
                + _types(patterns, page) + _properties(patterns, page))
    return candidate.model_copy(update={
        "query": query, "repairs": answered + renamed, "problems": problems,
        "relations": [p for _, p, _ in without_type_constraints(patterns) if not p.startswith("?")]})


def _unparsed(candidate: Candidate, query: str, repairs: list[str]) -> Candidate:
    return candidate.model_copy(update={"query": query, "repairs": repairs,
                                        "problems": [f"parse: {syntax_problem(query)}"]})


def _answer_repaired(query: str) -> tuple[str, list[str]]:
    variables, patterns = parse_query(query)
    mine = [p for p in patterns if ANSWER_VARIABLE in p]
    bound = bound_variables(patterns)
    order = parse_modifiers(query)["order"]
    if (variables != [ANSWER_VARIABLE] or count_variable(query) or len(bound) != 1
            or (order and order[0] == ANSWER_VARIABLE)
            or any(p[0] != ANSWER_VARIABLE or p[1] != TYPE_PREDICATE or p[2].startswith("?") for p in mine)):
        return query, []
    (answer,) = bound
    if mine and answer in type_constraints(patterns):
        return query, []
    fixed = [tuple(answer if term == ANSWER_VARIABLE else term for term in p) for p in patterns]
    return rebuild_query([answer], fixed, query), [f"answer: {ANSWER_VARIABLE} is bound by no pattern, selecting {answer}"]


def _repaired(query: str, page: Page) -> tuple[str, list[str]]:
    variables, patterns = parse_query(query)
    beyond = beyond_the_page(patterns, {v for v, _ in set_anchors(patterns)})
    fixed, repairs = [], []
    for subject, relation, obj in patterns:
        near = _renamed((subject, relation, obj), beyond, page)
        if near != relation:
            repairs.append(f"relation: {relation} -> {near}")
        turned = _turned(subject, near, obj, page.oriented)
        if turned != (subject, near, obj):
            repairs.append(f"orientation: {near} runs the other way")
        fixed.append(turned)
    return (rebuild_query(variables, fixed, query) if repairs else query), repairs


def _renamed(pattern: tuple[str, str, str], beyond: set[tuple[str, str, str]], page: Page) -> str:
    relation = pattern[1]
    if relation.startswith(("?", ":")) or relation in page.relations:
        return relation
    pool = page.relations | page.vocabulary if pattern in beyond else page.relations
    return _nearest(relation, pool) or relation


def _turned(subject: str, relation: str, obj: str, oriented: set[tuple[str, str, str]]) -> tuple[str, str, str]:
    backwards = ((subject, relation, "in") in oriented and (subject, relation, "out") not in oriented) or \
                ((obj, relation, "out") in oriented and (obj, relation, "in") not in oriented)
    return (obj, relation, subject) if backwards else (subject, relation, obj)


def _nearest(name: str, known: set[str]) -> str | None:
    squashed = _squashed(name)
    if not squashed:
        return None
    for reading in (lambda k: _squashed(k) == squashed,
                    lambda k: _squashed(k).endswith(squashed),
                    lambda k: "." not in k and len(_squashed(k)) >= 8 and squashed.startswith(_squashed(k)),
                    lambda k: bool(_words(name)) and _words(name) <= _words(k)):
        found = [k for k in known if reading(k)]
        if len(found) == 1:
            return found[0]
    return None


def _squashed(name: str) -> str:
    return "".join(c for c in name.lower() if c.isascii() and c.isalnum())


def _words(name: str) -> set[str]:
    return set("".join(c if c.isascii() and c.isalnum() else " " for c in name.lower()).split())


def _relations(patterns, page: Page) -> list[str]:
    beyond = beyond_the_page(patterns, {v for v, _ in set_anchors(patterns)})
    return [f"relation: {r} is not on the page" for s, r, o in without_type_constraints(patterns)
            if not r.startswith("?") and r not in page.relations
            and not ((s, r, o) in beyond and r in page.vocabulary)]


def _anchors(patterns, ctx: Context) -> list[str]:
    anchors = set(ctx.anchor_ids)
    loose = sorted({t for s, _, o in without_type_constraints(patterns) for t in (s, o)
                    if IDENTIFIER.match(t) and t not in anchors})
    unknown = sorted({name for _, name in set_anchors(patterns)} - set(ctx.task.sets))
    return ([f"anchor: {t} is not an anchor of this task" for t in loose]
            + [f"anchor: {name} is not a set this task holds" for name in unknown])


def _answer(variables: list[str], patterns) -> list[str]:
    bound = bound_variables(patterns)
    return [f"answer: {v} is bound by no pattern" for v in variables if v not in bound]


def _filters(query: str, variables: list[str], patterns) -> list[str]:
    if len(patterns) < 2:
        return []
    seen = Counter(t for p in patterns for t in p if t.startswith("?"))
    order = parse_modifiers(query)["order"]
    used = set(variables) | {v for _, _, v in property_reads(patterns)} | ({order[0]} if order else set())
    return [f"filter: {v} joins nothing and is never selected; the pattern only filters"
            for v, n in seen.items() if n == 1 and v not in used]


def _repeat(query: str, ctx: Context) -> list[str]:
    ran = {" ".join(e.query.split()) for a in ctx.attempts for e in a.executions}
    return ["repeat: this query already ran and did not settle the goal"] if " ".join(query.split()) in ran else []


def _routes(patterns, page: Page, ctx: Context) -> list[str]:
    hops = without_type_constraints(patterns)
    routes = {(p.relations[0], p.relations[1]) for _, p in page.routes}
    firsts = {first for first, _ in routes}
    problems = []
    for subject, relation, obj in hops:
        if subject not in ctx.anchor_ids or not obj.startswith("?") or relation not in page.mediators | firsts:
            continue
        onward = [p for s, p, _ in hops if s == obj]
        if not onward and relation in page.mediators:
            problems.append(f"route: {obj} is a record reached by {relation}, not an answer; add the second hop")
        problems += [f"route: {second} is not a route onward from {relation}"
                     for second in onward if not second.startswith("?") and (relation, second) not in routes]
    return problems


def _types(patterns, page: Page) -> list[str]:
    return [f"type: {kind} is not a type the page or the graph knows" for kinds in type_constraints(patterns).values()
            for kind in kinds if page.types and not kind.startswith("?") and kind not in page.types]


def _properties(patterns, page: Page) -> list[str]:
    keys = [key for _, key, _ in property_reads(patterns)] + [key for _, key in property_absences(patterns)]
    return [f"property: :{key} is not a property a route on the page lists" for key in keys if key not in page.properties]
