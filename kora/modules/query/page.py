from __future__ import annotations

from kora.provenance.models import LocalSchema, SchemaEntry
from kora.modules.query.models import K_RELATIONS, Context, Page
from kora.tools.dialect import quote_if_needed

DEAD_END = 3
CATALOGUE = 8


def build_page(kg, embedder, ctx: Context) -> Page:
    schema = kg.schema()
    whole = 0 < len(schema) <= K_RELATIONS
    blocks, lines, listed, routes, schemas = [], [], [], [], []
    types = set()
    for anchor in ctx.anchor_ids:
        every = _ordered(embedder, ctx, kg.properties(anchor))
        if ctx.task.page_limit:
            every = every[:ctx.task.page_limit]
        hub = len(every) > K_RELATIONS and not whole
        chosen = list(dict.fromkeys(e.predicate for e in every))[:K_RELATIONS]
        opened = kg.opened(anchor, chosen, failed=ctx.failed_routes)
        described = {(e.predicate, e.direction): e for e in opened.entries}
        shown = [described.get((e.predicate, e.direction), e) for e in every]
        block = _block(kg, anchor, shown, described, ctx.tried, hub)
        blocks.append(block + "\n" + _catalogue(kg, embedder, ctx) if len(every) < DEAD_END and not whole else block)
        lines += [(anchor, e) for p in chosen for e in opened.entries if e.predicate == p]
        listed += [(anchor, e) for e in shown]
        routes += [(anchor, p) for p in opened.paths]
        schemas.append(LocalSchema(entity_id=anchor, entries=shown, paths=opened.paths))
        types |= {t for e in opened.entries for t in [*e.target_types, e.target_type, *(x.kind for x in e.examples)] if t}
        types |= {t for p in opened.paths for t in (p.target_type, p.mediator_type) if t}
    blocks += [_routes(routes)] if routes else []
    blocks += [_schema_block(kg, schema)] if whole else []
    blocks += ["BOUND SETS\n" + "\n".join(f"  {h}" for h in ctx.task.held)] if ctx.task.held else []
    ctx.pages = schemas
    relations = {r["relation"] for r in schema} if whole else set()
    anchored = {e.predicate for _, e in listed} | {r for _, p in routes for r in p.relations}
    page = Page(text="\n\n".join(blocks) or "(no relations)", lines=lines, listed=listed, routes=routes,
                types=types | (_schema_types(schema) if whole else set(kg.known_types())),
                vocabulary=anchored | relations if whole else set(kg.vocabulary()),
                schemas=schemas, schema_relations=relations)
    return page


def _schema_block(kg, schema: list[dict]) -> str:
    notes = kg.schema_notes()
    head = ["HOW THIS GRAPH WORKS:"] + [f"  {n}" for n in notes] if notes else []
    lines = [f"  {r['relation']}: {r['from']} -> {r['to']}" + (f" ({r['edges']})" if r.get("edges") else "")
             for r in schema]
    return "\n".join(head + ["THE WHOLE GRAPH (every relation it has; a query may use any of them past the anchor):"]
                    + lines)


def _schema_types(schema: list[dict]) -> set[str]:
    return ({r["from"] for r in schema} | {r["to"] for r in schema}) - {"?"}


def _ordered(embedder, ctx: Context, every: list[SchemaEntry]) -> list[SchemaEntry]:
    scores = embedder.similarity(ctx.task.goal, [e.predicate for e in every])
    ranked = sorted(zip(scores, every, strict=True), key=lambda p: (-p[0], -p[1].count))
    return [e for _, e in ranked]


def _catalogue(kg, embedder, ctx: Context) -> str:
    names = sorted(kg.vocabulary())
    scores = embedder.similarity(ctx.task.goal, names)
    closest = [n for _, n in sorted(zip(scores, names, strict=True), key=lambda p: (-p[0], p[1]))][:CATALOGUE]
    return ("THIS ENTITY HAS ALMOST NO RELATIONS. Relations that exist elsewhere in the graph and match the goal "
            "(NOT usable from this entity; they say what another entity would need to be found first): "
            + ", ".join(closest))


def _block(kg, anchor: str, shown: list[SchemaEntry], described: dict, tried: dict, hub: bool) -> str:
    label = kg.label(anchor)
    head = f"PATTERNS FOR {quote_if_needed(anchor)}" + (f" = {label}" if label else "")
    head += f" ({len(shown)} relations; the {K_RELATIONS} closest to the goal are described)" if hub else ""
    return "\n".join([head + ":"] + [_line(anchor, e, (e.predicate, e.direction) in described, tried.get(e.predicate))
                                     for e in shown])


def _line(anchor: str, entry: SchemaEntry, described: bool, tried: str | None) -> str:
    written = quote_if_needed(anchor)
    pattern = f"  {written} {entry.predicate} ?x" if entry.direction == "out" else f"  ?x {entry.predicate} {written}"
    about = (_about(entry) if described else []) + ([f"TRIED: {tried}"] if tried else [])
    return pattern + f"   ({entry.count})" + ("   " + "; ".join(about) if about else "")


def _about(entry: SchemaEntry) -> list[str]:
    named = [e for e in entry.examples if e.name]
    kinds = entry.target_types or ([entry.target_type] if entry.target_type else [])
    about = []
    if entry.mediator and not named and not entry.target_types:
        about.append("?x is a MEDIATOR -- go through it, do not return it")
    elif entry.mediator:
        about.append("?x is mostly a MEDIATOR (see ROUTES) but also reaches: ?x :type " + " | ".join(kinds or ["?"]))
    elif kinds:
        about.append("?x :type " + " | ".join(kinds))
    if named:
        about.append("e.g. " + ", ".join(f"{e.name} ({e.kind})" if e.kind else e.name for e in named))
    elif entry.sample_name:
        about.append(f"e.g. {entry.sample_name}")
    return about


def _routes(routes) -> str:
    lines = ["ROUTES THROUGH A MEDIATOR (two steps, one fact -- the answer is at the far end, never the middle):"]
    for anchor, path in routes:
        first, second = path.relations[0], path.relations[1]
        kind = f" -> {path.target_type}" if path.target_type else ""
        example = f"   e.g. {path.sample_name}" if path.sample_name else ""
        keys = list(dict.fromkeys(x.rsplit(".", 1)[-1] for x in path.properties))
        about = ([f"?mid :type {path.mediator_type}"] if path.mediator_type else []) + \
                ([f"?mid has {', '.join(keys)}"] if keys else [])
        tail = f"   [{'; '.join(about)}]" if about else ""
        lines.append(f"  {quote_if_needed(anchor)} {first} ?mid . ?mid {second} ?answer{kind}{example}{tail}")
    return "\n".join(lines)
