"""Paths for one question: the enumeration from its anchors, the edits of the gold path, the pick of rows."""
from __future__ import annotations

import random

from .graph import Graph, Rel, Scorer, random_walk
from .rows import Hop, Node, Path, Row, row_of

PER_HOP = (50, 15, 10)      # relations followed at hop 1, 2, 3
FANOUT = (10, 5)            # hop-1 paths continued to hop 2; hop-2 paths continued to hop 3
PER_MEDIATOR = 20
OBJECT_LIMIT = 200
MIN_SCORE = -1.0            # keep everything the ranker returns; the bounds do the cutting


def ranked(scorer: Scorer, question: str, rels: list[Rel], keep: int) -> list[tuple[Rel, float]]:
    if not rels:
        return []
    scores = scorer(question, [r.relation for r in rels])
    pairs = sorted(zip(rels, scores), key=lambda p: (-p[1], -p[0].count))
    return [(r, s) for r, s in pairs if s >= MIN_SCORE][:keep]


def _through_mediator(graph: Graph, scorer: Scorer, question: str, path: Path, keep: int) -> list[Path]:
    """A path ending on unnamed nodes is not a candidate yet: continue it one step so it ends on names."""
    tail = path.hops[-1].objects
    out = []
    for rel, s in ranked(scorer, question, graph.relations(tail), keep):
        objs = graph.follow(tail, rel, OBJECT_LIMIT)
        if not objs or not any(o.named for o in objs):
            continue
        out.append(Path(path.anchors, path.hops + [Hop(tail, rel.relation, rel.direction, objs)], path.source,
                        score=(path.score + s) / 2, extra=dict(path.extra)))
    return out


def extend(graph: Graph, scorer: Scorer, question: str, path: Path, keep: int) -> list[Path]:
    """Every one-hop continuation of `path` through its `keep` best relations, mediators followed through."""
    tail = path.hops[-1].objects if path.hops else path.anchors
    out = []
    for rel, s in ranked(scorer, question, graph.relations(tail), keep):
        objs = graph.follow(tail, rel, OBJECT_LIMIT)
        if not objs:
            continue
        new = Path(path.anchors, path.hops + [Hop(tail, rel.relation, rel.direction, objs)], path.source,
                   score=(path.score * len(path.hops) + s) / (len(path.hops) + 1) if path.hops else s, extra=dict(path.extra))
        if any(o.named for o in objs):
            out.append(new)
        else:
            out += _through_mediator(graph, scorer, question, new, PER_MEDIATOR)
    return out


def enumerate_paths(graph: Graph, scorer: Scorer, question: str, anchors: list[Node], max_hops: int = 2,
                    per_hop: tuple = PER_HOP, fanout: tuple = FANOUT) -> list[Path]:
    """All executed paths of 1..max_hops named hops from `anchors`, source 'enum'."""
    root = Path(anchors, [], "enum")
    level = extend(graph, scorer, question, root, per_hop[0])
    found = list(level)
    for depth in range(1, max_hops):
        level = sorted(level, key=lambda p: -p.score)[:fanout[min(depth - 1, len(fanout) - 1)]]
        nxt = []
        for p in level:
            nxt += extend(graph, scorer, question, p, per_hop[min(depth, len(per_hop) - 1)])
        found += nxt
        level = nxt
    # Paths that loop back onto an anchor or end where they started teach nothing.
    anchor_ids = {a.id for a in anchors}
    return [p for p in found if p.candidate and not {n.id for n in p.candidate} <= anchor_ids]


SIBLINGS = 3
EXTENSIONS = 2
SAME_NAMES = 2
RANDOM_WALKS = 2


def truncations(spine: Path) -> list[Path]:
    return [Path(spine.anchors, spine.hops[:k], "truncate", extra={"of": "spine"})
            for k in range(1, len(spine.hops)) if spine.hops[:k][-1].objects]


def extensions(graph: Graph, scorer: Scorer, question: str, spine: Path, keep: int = EXTENSIONS) -> list[Path]:
    tail = spine.candidate
    out = []
    for rel, s in ranked(scorer, question, graph.relations(tail), keep):
        objs = graph.follow(tail, rel, OBJECT_LIMIT)
        if objs and any(o.named for o in objs):
            out.append(Path(spine.anchors, spine.hops + [Hop(tail, rel.relation, rel.direction, objs)], "extend", score=s))
    return out


def siblings(graph: Graph, scorer: Scorer, question: str, spine: Path, keep: int = SIBLINGS) -> list[Path]:
    """The last hop re-walked through other relations of the same subjects, best-reading first."""
    if not spine.hops:
        return []
    last = spine.hops[-1]
    rels = [r for r in graph.relations(last.subjects) if (r.relation, r.direction) != (last.relation, last.direction)]
    out = []
    for rel, s in ranked(scorer, question, rels, keep):
        objs = graph.follow(last.subjects, rel, OBJECT_LIMIT)
        if objs:
            out.append(Path(spine.anchors, spine.hops[:-1] + [Hop(last.subjects, rel.relation, rel.direction, objs)],
                            "sibling", score=s))
    return out


def same_name_walks(graph: Graph, spine: Path, keep: int = SAME_NAMES) -> list[Path]:
    """The spine's chain of relations walked from a different node with the anchor's name."""
    out = []
    for anchor in spine.anchors:
        for other in graph.same_name(anchor, keep):
            current, hops = [other], []
            for h in spine.hops:
                objs = graph.follow(current, Rel(h.relation, h.direction), OBJECT_LIMIT)
                if not objs:
                    break
                hops.append(Hop(current, h.relation, h.direction, objs)); current = objs
            if len(hops) == len(spine.hops) and hops[-1].objects:
                out.append(Path([other], hops, "same_name"))
    return out


def random_walks(graph: Graph, spine: Path, rng: random.Random, keep: int = RANDOM_WALKS) -> list[Path]:
    out = []
    for _ in range(keep):
        p = random_walk(graph, spine.anchors, len(spine.hops), rng)
        if p is not None and p.candidate:
            out.append(p)
    return out


def edits_of(graph: Graph, scorer: Scorer, question: dict, spine: Path, rng: random.Random) -> list[Path]:
    q = question["question"]
    found = truncations(spine) + extensions(graph, scorer, q, spine) + siblings(graph, scorer, q, spine)
    found += graph.spine_variants(question, spine) + same_name_walks(graph, spine) + random_walks(graph, spine, rng)
    return found


POSITIVES = 8
EDITS = 20                  # edits of the gold path (truncate, extend, sibling, drop_constraint, wrong_value, same_name)
PLAUSIBLE = 8               # enumerated wrong paths that read like the question
SINGLES = 4                 # of which, one-to-three-entity answers first
OTHER = 4                   # easy negatives (random walks first)
NEAR_MISS_SOURCES = ("truncate", "extend", "sibling", "drop_constraint", "wrong_value", "same_name")
EDIT_ORDER = {"wrong_value": 0, "drop_constraint": 1, "same_name": 2, "sibling": 3, "truncate": 4, "extend": 5}


def never_sent(p: Path) -> str | None:
    """Why no KGQA system would send this state, or None. Applied to every source."""
    if not any(n.named for n in p.candidate):
        return "unnamed candidate"                  # nothing to show: "(N items)" then "(empty)"
    if p.hops and sum(1 for s in p.hops[0].subjects if s.named) > 1:
        return "glued anchors"                      # starts from all the question's entities at once: a false fact
    for a, b in zip(p.hops, p.hops[1:]):
        if {n.id for n in b.objects} == {n.id for n in a.subjects}:
            return "returns to its start"           # the next hop walks straight back to where the last one began
    if p.hops and {n.id for n in p.candidate} <= {n.id for n in p.anchors}:
        return "returns to its start"
    return None


def select(paths: list[Path], question: dict, bench: str) -> list[Row]:
    gold_names = question.get("answers") or []
    gold_ids = question.get("answer_ids") or None
    seen: set[tuple] = set(); rows: list[tuple[Path, Row]] = []
    # An edit of the spine and an enumerated path can walk the same chain; the edit's source is the informative one.
    order = {"spine": 0, **{s: 1 for s in NEAR_MISS_SOURCES}, "enum": 2, "random": 3}
    for p in sorted(paths, key=lambda p: order.get(p.source, 2)):
        k = p.key()
        if k in seen or not p.candidate or never_sent(p):
            continue
        seen.add(k)
        rows.append((p, row_of(p, str(question["id"]), bench, question["question"], gold_names, gold_ids)))
    pos = [r for p, r in rows if r.label == 1]
    pos.sort(key=lambda r: (r.source != "spine", -r.score))
    neg = [r for p, r in rows if r.label == 0]
    edits = sorted([r for r in neg if r.source in NEAR_MISS_SOURCES], key=lambda r: EDIT_ORDER.get(r.source, 9))[:EDITS]
    enum = sorted([r for r in neg if r.source == "enum"], key=lambda r: -r.score)
    singles = [r for r in enum if 0 < r.n_candidate <= 3 and r.candidate][:SINGLES]
    rest = sorted([r for r in enum if r not in singles], key=lambda r: (-(r.recall > 0), -r.score))
    plausible = (singles + rest)[:PLAUSIBLE]
    taken = {id(r) for r in edits + plausible}
    other = sorted([r for r in neg if id(r) not in taken], key=lambda r: (r.source != "random", r.score))[:OTHER]
    return pos[:POSITIVES] + edits + plausible + other
