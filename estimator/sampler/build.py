"""Build the sampled rows for one set (`python -m estimator.sampler.build --help`)."""
from __future__ import annotations

import argparse
import collections
import importlib
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from .graph import Graph, Scorer, lexical_scorer
from .paths import edits_of, enumerate_paths, select
from .rows import Node, Path, Row, add_kinds, relations_of, write_rows


def anchors_of(question: dict, graph: Graph) -> list[Node]:
    ids = question.get("topic_entity_ids") or list((question.get("topic_entity") or {}).keys())
    names = question.get("topic_entity") or {}
    return [Node(i, names.get(i) or (graph.name(i) if hasattr(graph, "name") else i)) for i in ids]


def build_question(graph: Graph, scorer: Scorer, question: dict, max_hops: int, rng: random.Random) -> tuple[list[Row], dict]:
    anchors = anchors_of(question, graph)
    if not anchors:
        return [], {"skipped": "no anchor"}
    spine = graph.spine(question)
    paths: list[Path] = [spine] if spine is not None else []
    # Each path starts from one entity, as KGQA systems walk: starting from all of them at once glues their names
    # into one subject and states false facts.
    for anchor in anchors:
        paths += enumerate_paths(graph, scorer, question["question"], [anchor], max_hops=max_hops)
    if spine is not None:
        paths += edits_of(graph, scorer, question, spine, rng)
    rows = select(paths, question, graph.bench)
    if hasattr(graph, "kinds"):
        add_kinds(rows, graph.kinds)                                # anchor and candidate kinds
    if spine is not None:
        gold = relations_of(spine)
        if hasattr(graph, "reverse_relations"):
            gold = gold + sorted(graph.reverse_relations(gold, spine))   # Freebase names each direction of a relation
        mark_lucky(rows, gold)
    lucky = sum(r.lucky for r in rows)
    rows = [r for r in rows if not r.lucky]         # right by overlap, not by the path: never written
    info = {"spine": spine is not None, "paths": len(paths), "rows": len(rows),
            "pos": sum(r.label for r in rows), "spine_positive": None, "lucky dropped": lucky}
    if spine is not None:
        sp = next((r for r in rows if r.source == "spine"), None)
        info["spine_positive"] = bool(sp and sp.label == 1)
    return rows, info


def _relation_words(relations: list[str]) -> set[str]:
    """Relations as single names: a joined hop is labelled `a & b`, a Freebase side `a / b`."""
    return {w.strip() for r in relations for part in r.split(" & ") for w in part.split(" / ") if w.strip()}


def mark_lucky(rows: list[Row], spine_relations: list[str]) -> None:
    """Mark `lucky` a yes that shares no relation with the gold path: it passes the set rule by overlap alone."""
    gold = _relation_words(spine_relations)
    for r in rows:
        if r.label == 1 and r.source != "spine" and not (_relation_words(r.relations) & gold):
            r.lucky = True


def embed_scorer() -> Scorer:
    from kora.tools.embed import Embedder
    emb = Embedder()
    return lambda question, names: emb.similarity(question, names)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m estimator.sampler.build")
    ap.add_argument("bench"); ap.add_argument("adapter")
    ap.add_argument("--questions", required=True); ap.add_argument("--qids")
    ap.add_argument("--out", required=True); ap.add_argument("--max-hops", type=int, default=2)
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--limit", type=int)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--scorer", default="embed")
    ap.add_argument("--exclude", nargs="*", default=[], help="held-out files (validation, test): their questions, "
                    "by id and by text, are never sampled")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    qs = json.load(open(a.questions))["questions"]
    if a.qids:
        keep = {str(x) for x in json.load(open(a.qids))}
        qs = [q for q in qs if str(q["id"]) in keep]
    if a.exclude:
        from .rows import ids_and_texts
        held_ids: set[str] = set(); held_texts: set[str] = set()
        for h in a.exclude:
            i, t = ids_and_texts(h); held_ids |= i; held_texts |= t
        before = len(qs)
        qs = [q for q in qs if str(q["id"]) not in held_ids and q["question"].strip().casefold() not in held_texts]
        print(f"[sampler] excluded {before - len(qs)} questions found in {len(a.exclude)} held-out files", flush=True)
    if a.limit:
        qs = qs[:a.limit]
    module = importlib.import_module(a.adapter if "." in a.adapter else f"estimator.sampler.adapters.{a.adapter}")
    graph = module.make_graph(a.bench)
    scorer = embed_scorer() if a.scorer == "embed" else lexical_scorer
    rows_file, meta = f"{a.out}/{a.bench}.rows.jsonl", f"{a.out}/{a.bench}.meta.jsonl"
    for f in (rows_file, meta):
        if os.path.exists(f):
            os.remove(f)
    stats = collections.Counter(); sources = collections.Counter(); t0 = time.time(); done = 0
    print(f"[sampler] {a.bench}: {len(qs)} questions, max_hops={a.max_hops}, scorer={a.scorer}", flush=True)
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures = {pool.submit(build_question, graph, scorer, q, a.max_hops, random.Random(a.seed + i)): q
                   for i, q in enumerate(qs)}
        for fut in as_completed(futures):
            q = futures[fut]
            try:
                rows, info = fut.result()
            except Exception as exc:  # one question's failure never stops the set; it is counted
                stats["failed"] += 1; print(f"[sampler] {q['id']}: {type(exc).__name__}: {str(exc)[:160]}", flush=True)
                continue
            write_rows(rows, rows_file, meta)
            stats["questions"] += 1; stats["rows"] += len(rows); stats["pos"] += info.get("pos", 0)
            stats["with_spine"] += bool(info.get("spine")); stats["spine_positive"] += bool(info.get("spine_positive"))
            stats["no_rows"] += not rows; stats["lucky dropped"] += info.get("lucky dropped", 0)
            for r in rows:
                sources[f"{r.source}:{r.label}"] += 1
            done += 1
            if done % 50 == 0:
                print(f"[sampler] {done}/{len(qs)} {time.time() - t0:.0f}s rows={stats['rows']} pos={stats['pos']}", flush=True)
    out = {**stats, "sources": dict(sources), "seconds": round(time.time() - t0)}
    json.dump(out, open(f"{a.out}/{a.bench}.stats.json", "w"), indent=1)
    print(f"[sampler] done {json.dumps(out)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
