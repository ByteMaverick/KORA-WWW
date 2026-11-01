"""Our question sets in KG-R1's input format: the graph server's per-question subgraphs and the parquet."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

from harness.dataset import path_of, records as question_records

ROOT = Path(__file__).resolve().parents[2]
OUR_GRAPHS = ("metaqa", "biocq", "kqapro")
FREEBASE = ("cwq", "webqsp")
#: The Unseen halves on Freebase: GCR's live 2-hop subgraphs of the extended graph (baselines/gcr/export.py).
UNSEEN_FREEBASE = ("webqsp-unseen", "webqsp-unseen-real", "cwq-unseen", "cwq-unseen-real")

#: The authors' KG prompt, verbatim (their scripts/data_process_kg/cwq_search_augmented_initial_entities.py and
#: the WebQSP one).
_PROMPT = (
    "Answer the given question. You must conduct reasoning inside <think> and </think> "
    "first every time you get new information. After reasoning, if you find you lack some "
    "knowledge, you can query the knowledge graph by using <kg-query> function_name(arguments) </kg-query>, and it will "
    "return the top query results between <information> and </information>. You "
    "can query as many times as you want. If you find no further external knowledge "
    "needed, you can directly provide the answer inside <answer> and </answer> without "
    "detailed illustrations. For example, <answer> Beijing </answer>.\n\nQuestion: {question}"
)


def kg_name(set_name: str) -> str:
    """The server's name for the set's knowledge graph: their server and evaluator spell CWQ in capitals."""
    return "CWQ" if set_name == "cwq" else set_name


def dataset_name(set_name: str) -> str:
    """extra_info.dataset_name: their evaluator turns "cwq" into "CWQ" itself."""
    return set_name


def prompt(question: str, initial_entities: list[str]) -> str:
    """The model's prompt as the authors build it: the question, then "? (Initial entities: ...)" when there are any."""
    if initial_entities:
        names = ", ".join(f'"{e}"' for e in initial_entities)
        question = f"{question}? (Initial entities: {names})"
    return _PROMPT.format(question=question)


def clean(name) -> str:
    """A name as one line of the server's files (its loader strips each line and skips blank ones)."""
    return " ".join(str(name).split())


def topic_names(q: dict) -> list[str]:
    return list(dict.fromkeys(n for n in (clean(v) for v in (q.get("topic_entity") or {}).values()) if n))


def parquet_row(set_name: str, q: dict, initial_entities: list[str], initial_ids: list[str]) -> dict:
    return {"data_source": f"kgR1_{set_name}",
            "prompt": [{"role": "user", "content": prompt(q["question"], initial_entities)}],
            "ability": "kg-reasoning",
            "reward_model": {"style": "rule", "ground_truth": {"target_text": [], "target_kb_id": []}},
            "extra_info": {"split": "test", "sample_id": str(q["id"]), "dataset_name": dataset_name(set_name),
                           "initial_entities": list(initial_entities), "initial_entity_ids": [str(i) for i in initial_ids]}}


def write_parquet(rows: list[dict], path: Path) -> None:
    """Their columns, with an explicit schema: an all-empty gold list would otherwise be typed list<null>."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    strings = pa.list_(pa.string())
    schema = pa.schema([
        ("data_source", pa.string()),
        ("prompt", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))),
        ("ability", pa.string()),
        ("reward_model", pa.struct([("style", pa.string()),
                                    ("ground_truth", pa.struct([("target_text", strings), ("target_kb_id", strings)]))])),
        ("extra_info", pa.struct([("split", pa.string()), ("sample_id", pa.string()), ("dataset_name", pa.string()),
                                  ("initial_entities", strings), ("initial_entity_ids", strings)])),
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ── our graphs: GCR's subgraphs as index tuples ──────────────────────────────────────────────────────────────────
def gcr_records_from_file(path: Path, questions: list[dict]):
    """(q, graph) per question from the file GCR's export wrote; graph None for a question it lacks."""
    by_id = {}
    for line in path.open(encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            by_id[str(r["id"])] = r
    for q in questions:
        r = by_id.get(str(q["id"]))
        if r is None:
            yield q, None
            continue
        if [clean(x) for x in r["q_entity"]] != [clean(x) for x in (q.get("topic_entity") or {}).values()]:
            raise SystemExit(f"{path}: {q['id']}'s q_entity {r['q_entity']} is not the split's topic entities; "
                             "export GCR again from this split")
        yield q, [tuple(t) for t in r["graph"]]


def gcr_records_built(set_name: str, questions: list[dict], kb: Path | None, endpoint: str | None):
    """(q, graph) per question, built with GCR's own functions (baselines/gcr/export.py), as GCR's export does."""
    from baselines.gcr import export as gcr
    if set_name.startswith("metaqa"):
        records = gcr.metaqa_records(questions, kb)
    elif set_name in UNSEEN_FREEBASE:   # a question the graph was too slow for has no subgraph (graph None): a miss
        notes = {"missing": {}}
        got = {str(q["id"]): graph for q, _a, graph, _c in gcr.freebase_records(questions, endpoint, notes)}
        for q in questions:
            yield q, (list(got[str(q["id"])]) if str(q["id"]) in got else None)
        return
    else:
        records = gcr.biocq_records(questions, endpoint, literals=set_name.startswith("kqapro"))
    for q, _anchors, graph, _capped in records:
        yield q, list(graph)


def our_graph(set_name: str, records, out: Path):
    """Write <out>/kg/<K>/ from (q, graph) records; returns (parquet rows, missing, stats)."""
    entities: dict[str, int] = {}
    relations: dict[str, int] = {}
    idx = lambda table, name: table.setdefault(name, len(table))
    lines, rows, missing = [], [], []
    stats = {"triples": 0, "nameless_triples": 0, "empty_subgraphs": 0, "capped_at": None}
    for q, graph in records:
        if graph is None:
            missing.append({"id": str(q["id"]), "why": "not in the GCR export"})
            continue
        names = topic_names(q)
        tuples = []
        for h, r, t in graph:
            h, r, t = clean(h), clean(r), clean(t)
            if not (h and r and t):
                stats["nameless_triples"] += 1
                continue
            tuples.append([idx(entities, h), idx(relations, r), idx(entities, t)])
        anchors = [idx(entities, n) for n in names]
        held = sorted({i for h, _, t in tuples for i in (h, t)} | set(anchors))
        stats["triples"] += len(tuples)
        stats["empty_subgraphs"] += not tuples
        lines.append({"id": str(q["id"]), "answers": [], "subgraph": {"tuples": tuples, "entities": held}})
        rows.append(parquet_row(set_name, q, names, list((q.get("topic_entity") or {}).keys())))
    folder = out / "kg" / kg_name(set_name)
    folder.mkdir(parents=True, exist_ok=True)
    names_text = "".join(n + "\n" for n in entities)
    (folder / "entities_text.txt").write_text(names_text, encoding="utf-8")
    (folder / "entities.txt").write_text(names_text, encoding="utf-8")
    (folder / "relations.txt").write_text("".join(r + "\n" for r in relations), encoding="utf-8")
    with (folder / "test_simple.json").open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    stats.update(entities=len(entities), relations=len(relations))
    return rows, missing, stats


# ── Freebase: the authors' subgraphs, our questions and topic entities ───────────────────────────────────────────
def freebase(set_name: str, questions: list[dict], data_kg: Path, out: Path):
    src = data_kg / kg_name(set_name)
    ids = [l.strip() for l in (src / "entities.txt").open(encoding="utf-8")]
    texts = [l.strip() for l in (src / "entities_text.txt").open(encoding="utf-8")]
    if len(ids) != len(texts):
        raise SystemExit(f"{src}: entities.txt and entities_text.txt differ in length ({len(ids)} vs {len(texts)})")
    index = {mid: i for i, mid in enumerate(ids)}
    wanted = {str(q["id"]) for q in questions}
    subgraphs = {}
    for line in (src / "test_simple.json").open(encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            if str(r.get("id")) in wanted:
                subgraphs[str(r["id"])] = r.get("subgraph") or {}
    folder = out / "kg" / kg_name(set_name)
    folder.mkdir(parents=True, exist_ok=True)
    for f in ("entities.txt", "entities_text.txt", "relations.txt"):
        shutil.copyfile(src / f, folder / f)
    rows, missing = [], []
    stats = {"topics_outside_subgraph": 0, "empty_subgraphs": 0}
    with (folder / "test_simple.json").open("w", encoding="utf-8") as f:
        for q in questions:
            sg = subgraphs.get(str(q["id"]))
            if sg is None:
                missing.append({"id": str(q["id"]), "why": f"not in the authors' {kg_name(set_name)}/test_simple.json"})
                continue
            held = set(sg.get("entities") or [])
            names, mids = [], []
            for mid, name in (q.get("topic_entity") or {}).items():
                i = index.get(mid)
                if i is None or i not in held:
                    stats["topics_outside_subgraph"] += 1
                    names.append(clean(name))      # the split's name: the model's lookups will say it is absent
                else:
                    names.append(texts[i])
                mids.append(mid)
            stats["empty_subgraphs"] += not sg.get("tuples")
            f.write(json.dumps({"id": str(q["id"]), "answers": [], "subgraph": sg}, ensure_ascii=False) + "\n")
            rows.append(parquet_row(set_name, q, list(dict.fromkeys(n for n in names if n)), mids))
    return rows, missing, stats


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--set", required=True, help="metaqa, cwq, webqsp, webqsp-unseen(-real), "
                                                "cwq-unseen(-real)")
    p.add_argument("--split", default="test")
    p.add_argument("--gcr-export", type=Path, help="our graphs: the folder GCR's export wrote (<dir>/<set>/test.jsonl)")
    p.add_argument("--kb", type=Path, help="MetaQA: the kb.txt of the graph the set is answered on (GCR's function)")
    p.add_argument("--endpoint", help="BioCQ, KQA Pro, the Unseen halves: the SPARQL endpoint of the set's graph "
                                      "(GCR's function)")
    p.add_argument("--data-kg", type=Path, help="CWQ, WebQSP: the authors' data_kg folder (their download)")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    split = Path(path_of(a.set, a.split))
    questions = question_records(a.set, a.split)
    if a.set.startswith(OUR_GRAPHS) or a.set in UNSEEN_FREEBASE:
        if a.gcr_export:
            source = a.gcr_export / a.set / f"{a.split}.jsonl"
            if not source.exists():
                p.error(f"{source}: no GCR export for {a.set}")
            records, origin = gcr_records_from_file(source, questions), {"gcr_export": str(source),
                                                                         "sha256": sha256(source)}
        elif a.set.startswith("metaqa") and a.kb:
            records, origin = gcr_records_built(a.set, questions, a.kb, None), {"kb": str(a.kb)}
        elif not a.set.startswith("metaqa") and a.endpoint:
            records, origin = gcr_records_built(a.set, questions, None, a.endpoint), {"endpoint": a.endpoint}
        else:
            p.error("our graphs need --gcr-export, or --kb (MetaQA) / --endpoint (BioCQ, KQA Pro, the Unseen halves)")
        rows, missing, stats = our_graph(a.set, records, a.out)
    elif a.set in FREEBASE:
        if not a.data_kg:
            p.error("--data-kg (the authors' data_kg folder) is needed for CWQ and WebQSP")
        rows, missing, stats = freebase(a.set, questions, a.data_kg, a.out)
        origin = {"data_kg": str(a.data_kg / kg_name(a.set))}
    else:
        p.error(f"{a.set}: not exported for KG-R1 (GrailQA: no subgraph source; see the module docstring)")
    dest = a.out / "parquet" / a.set
    write_parquet(rows, dest / "test.parquet")
    (dest / "missing.json").write_text(json.dumps(missing, indent=1), encoding="utf-8")
    meta = {"set": a.set, "split": os.path.relpath(split, ROOT), "split_sha256": sha256(split), "kg": kg_name(a.set),
            "questions": len(questions), "exported": len(rows), "missing": len(missing), "source": origin, **stats}
    (dest / "export_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"[kg_r1 export] {a.set}: {len(rows)} of {len(questions)} questions -> {dest}/test.parquet, graph "
          f"{a.out}/kg/{kg_name(a.set)}; {len(missing)} without a subgraph (missing.json; misses); {stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
