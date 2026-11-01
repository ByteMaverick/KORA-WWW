from __future__ import annotations

from kora.provenance.models import PathSchema, SchemaEntry
from kora.modules.query.models import Candidate, Context, Page, reaches
from kora.tools.dialect import quote_if_needed

PAGE_SEEDS = 6


def seeds(ctx: Context, page: Page) -> list[Candidate]:
    tried = set(ctx.tried)
    expected = ctx.task.expected_entity_type
    route = next(((a, p) for a, p in page.routes if not set(p.relations) & tried), None)
    seeded = {route[1].relations[0]} if route else set()
    hops = []
    for a, e in page.lines:
        if e.mediator or e.predicate in tried or e.predicate in seeded:
            continue
        seeded.add(e.predicate)
        hops.append((a, e))
    found = ([_route(*route, expected)] if route else []) + [_one_hop(a, e, expected) for a, e in hops]
    return [c.model_copy(update={"id": f"s{n}"}) for n, c in enumerate(found[:PAGE_SEEDS], start=1)]


def _one_hop(anchor: str, entry: SchemaEntry, expected: str) -> Candidate:
    written = quote_if_needed(anchor)
    body = f"{written} {entry.predicate} ?x" if entry.direction == "out" else f"?x {entry.predicate} {written}"
    kind = next((k for k in [*entry.target_types, entry.target_type] if k and reaches([k], expected)), None)
    body += f" . ?x :type {kind}" if kind else ""
    return Candidate(id="s", query=f"SELECT ?x WHERE {{ {body} }}", relations=[entry.predicate])


def _route(anchor: str, path: PathSchema, expected: str) -> Candidate:
    first, second = path.relations[0], path.relations[1]
    kind = path.target_type if path.target_type and reaches([path.target_type], expected) else None
    body = f"{quote_if_needed(anchor)} {first} ?mid . ?mid {second} ?x" + (f" . ?x :type {kind}" if kind else "")
    return Candidate(id="s", query=f"SELECT ?x WHERE {{ {body} }}", relations=[first, second])
