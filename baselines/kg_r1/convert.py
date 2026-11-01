"""KG-R1's per-example records as a run folder (<results>/<run>/kg-r1-cqw/<set>-test/kg_r1: run.json, s0.jsonl)."""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

EVAL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EVAL_ROOT))
from baselines.kg_r1 import patch_upstream, upstream  # noqa: E402
from harness import settings  # noqa: E402
from harness.metrics.reasoning import score_text as _score_text_answer  # noqa: E402
from harness.dataset import path_of, records as question_records  # noqa: E402
from harness.manifest import run_manifest  # noqa: E402

_INFO = re.compile(r"<information>(.*?)</information>", re.S)
_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.S)
_TURN = re.compile(r"<kg-query>(.*?)</kg-query>|<information>(.*?)</information>", re.S)
#: What the server returns with data (kg_r1/search/actions.py, every success branch).
SUCCESS = ('Outgoing relations for entity "', 'Incoming relations for entity "', 'Entities outgoing via relation "',
           'Entities incoming via relation "', 'Relations for entity "')
#: Their client's messages when the server could not be reached or answered with a transport error
#: (kg_r1/llm_agent/generation.py _batch_search, _format_http_error), and the server's own refusal of a knowledge
#: graph it was not started with (server.py execute_action).
OUTAGE = ("KG server timeout (", "KG server connection failed (", "KG server response error (",
          "KG server request timed out", "Cannot connect to KG server", "KG server endpoint not found",
          "KG server internal error", "KG server unavailable", "KG server request failed for ", "not supported (available:")
#: Their server's refusal of a whole POST (FastAPI's 422): one request of a turn named an action its schema does not
#: take, and every query of that turn gets this message. Their code does this in their own runs too: counted, not an error.
REJECTED = "Input should be '"


def final_answer(response: str) -> str:
    """The authors' answer: the first <answer> once the <information> blocks are removed (extract_answer_kg)."""
    found = _ANSWER.findall(_INFO.sub("", response or ""))
    return found[0].strip() if found else ""


def answer_set(answer: str) -> list[str]:
    """The authors' own answer set: their entity F1 splits the answer on commas (kg_format_multiturn.py)."""
    return list(dict.fromkeys(e.strip() for e in answer.split(",") if e.strip()))


def trajectory(response: str) -> dict:
    """Queries, the server's answers to them, and the generation turns, read from one trajectory."""
    queries = results = errors = blocks = rejected = 0
    outage = None
    pending = False
    for m in _TURN.finditer(response or ""):
        if m.group(1) is not None:
            queries += 1
            pending = True
            continue
        blocks += 1
        info = m.group(2).strip()
        hit = next((o for o in OUTAGE if o in info), None)
        if hit and outage is None:
            outage = hit
        if pending:                      # the server's answer to the query just made (else: an invalid-turn note)
            if info.startswith(SUCCESS):
                results += 1
            else:
                errors += 1
                rejected += REJECTED in info
        pending = False
    executed = results + errors
    return {"kgr1_kg_queries": executed, "kgr1_kg_results": results, "kgr1_kg_errors": errors,
            "kgr1_unanswered_queries": queries - executed,      # a query in the final turn is not run
            "kgr1_batch_rejected": rejected,
            "num_llm_calls": blocks + 1, "outage": outage}


def read_records(path: Path) -> dict:
    out: dict = {}
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if "harness_sample_id" not in r:
            raise SystemExit(f"{path}: a record without harness_sample_id: the clone is not patched "
                             f"(python -m baselines.kg_r1.patch_upstream <clone>); rerun, never convert it")
        out.setdefault(str(r["harness_sample_id"]), []).append(r)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--detailed", type=Path, required=True, help="their test_detailed_results.jsonl")
    p.add_argument("--export", type=Path, required=True, help="the export's parquet/<set> folder")
    p.add_argument("--set", required=True)
    p.add_argument("--run", default="test")
    p.add_argument("--results", default="results/eval")
    a = p.parse_args(argv)
    import pyarrow.parquet as pq
    split = Path(path_of(a.set, "test"))
    questions = question_records(a.set, "test")
    exported = {r["extra_info"]["sample_id"]: r for r in pq.read_table(a.export / "test.parquet").to_pylist()}
    never_run = {m["id"] for m in json.loads((a.export / "missing.json").read_text(encoding="utf-8"))}
    records = read_records(a.detailed)
    rollouts = settings.system("kg_r1")["rollouts"]
    many = [q for q, rs in records.items() if len(rs) != rollouts]
    if many:
        raise SystemExit(f"{len(many)} questions with other than {rollouts} rollout(s), e.g. {many[0]}: not this "
                         "setting's run")
    stray = set(records) - set(exported)
    if stray:
        raise SystemExit(f"{len(stray)} records for questions this export does not hold, e.g. {sorted(stray)[0]}")
    batch_size = Counter(rs[0]["batch_idx"] for rs in records.values())
    manifest = run_manifest(argparse.Namespace(framework="kg_r1", engine=upstream.ENGINE, dataset_path=str(split),
                                               question_timeout_seconds=settings.TIMEOUT_SECONDS))
    clone = {"repo": upstream.REPO, "commit": upstream.COMMIT, "patch": patch_upstream.MARK,
             "checkpoint": f"{upstream.CHECKPOINT}@{upstream.CHECKPOINT_REVISION}"}
    out = Path(a.results) / a.run / upstream.ENGINE / f"{a.set}-test" / "kg_r1"
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.json").write_text(json.dumps(        # what `python -m harness score` reads, as the runner writes it
        {"system": "kg_r1", "stop": "", "framework": "kg_r1", "dataset": a.set, "split": "test",
         "model": upstream.ENGINE, "timeout_seconds": settings.TIMEOUT_SECONDS}, indent=1) + "\n")
    counts = Counter()
    with (out / "s0.jsonl").open("w", encoding="utf-8") as f:
        for q in questions:
            qid = str(q["id"])
            row = {"qid": qid, "question": q["question"], "answer": "", "answers": q["answers"], "framework": "kg_r1",
                   "engine": upstream.ENGINE, "benchmark": f"{a.set}-test", "error": None,
                   "wall_time_s": float(settings.TIMEOUT_SECONDS), "fw_metrics": {"kgr1_upstream": clone},
                   "manifest": manifest}
            if qid not in records:
                row["error"] = "no_subgraph" if qid in never_run else "no_output"
            else:
                r = records[qid][0]
                if exported[qid]["prompt"][0]["content"].rsplit("Question: ", 1)[-1] not in r.get("prompt", ""):
                    raise SystemExit(f"{qid}: its record's prompt does not hold its question: records and questions "
                                     "are misaligned; do not convert this run")
                t = trajectory(r.get("response", ""))
                seconds = float(r["harness_batch_seconds"])
                answer = final_answer(r.get("response", ""))
                row["wall_time_s"] = round(seconds, 3)
                row["fw_metrics"].update({k: v for k, v in t.items() if k != "outage"})
                row["fw_metrics"].update(kgr1_batch_seconds=round(seconds, 3), kgr1_batch_size=batch_size[r["batch_idx"]],
                                         kgr1_amortized_seconds=round(seconds / batch_size[r["batch_idx"]], 3),
                                         kgr1_answer_set=answer_set(answer),
                                         kgr1_response_chars=len(r.get("response", "")))
                if seconds > settings.TIMEOUT_SECONDS:
                    row["error"] = "timeout"
                elif t["outage"]:
                    row["error"] = f"graph_outage: {t['outage']}"
                else:
                    row["answer"] = answer
            hit, em, prec, rec, f1 = _score_text_answer(row["answer"], q["answers"], None) if row["answer"] \
                else (0, 0, 0, 0, 0)
            row["eval"] = {"hit@1": hit, "exact_match": em, "precision": prec, "recall": rec, "f1": f1}
            counts["rows"] += 1
            counts["hits"] += hit
            counts["answered" if row["answer"] else row["error"] or "no_answer"] += 1
            counts["with a turn the server refused whole"] += bool(row["fw_metrics"].get("kgr1_batch_rejected"))
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[kg_r1 convert] {a.set}: {counts['rows']} rows -> {out}/s0.jsonl, Hit {counts['hits'] / max(counts['rows'], 1):.3f}; "
          + ", ".join(f"{k} {v}" for k, v in sorted(counts.items()) if k not in ("rows", "hits")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
