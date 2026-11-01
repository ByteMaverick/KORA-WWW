"""GCR's predictions as a run folder (<results>/<run>/<model>/<set>-test/gcr: run.json, s0.jsonl), timed and scored."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EVAL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EVAL_ROOT))
from harness import settings  # noqa: E402
from harness.metrics.reasoning import score_text as _score_text_answer  # noqa: E402
from harness.dataset import path_of, records as question_records  # noqa: E402
from harness.manifest import run_manifest  # noqa: E402
from baselines.gcr.timed import load_times  # noqa: E402


def as_text(prediction) -> str:
    if isinstance(prediction, list):
        return "; ".join(str(p) for p in prediction)
    return str(prediction or "")


def as_items(prediction) -> list | None:
    """GCR's prediction list, scored as its items; None for a text prediction."""
    return [str(p).strip() for p in prediction if str(p).strip()] if isinstance(prediction, list) else None


def step1_paths(path: Path) -> dict:
    """qid -> the number of paths step 1 wrote for it."""
    out = {}
    for line in path.open(encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            out[str(r["id"])] = len(r.get("prediction") or [])
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--paths", type=Path, required=True, help="step 1's predictions.jsonl (the paths step 2 read)")
    p.add_argument("--times", type=Path, nargs="+", required=True,
                   help="the per-question seconds timed.py wrote: step 1's file and this model's step 2 file")
    p.add_argument("--exported", type=Path,
                   help="the export's <set>.<split>.harness.json: every question it exported or found missing gets a row")
    p.add_argument("--set", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--run", default="test")
    p.add_argument("--results", default="results/eval")
    a = p.parse_args(argv)
    split = Path(path_of(a.set, "test"))
    gold = {str(q["id"]): q for q in question_records(a.set, "test")}
    paths = step1_paths(a.paths)
    times = load_times(*a.times)
    manifest = run_manifest(argparse.Namespace(framework="gcr", engine=a.model, dataset_path=str(split),
                                               question_timeout_seconds=settings.TIMEOUT_SECONDS))
    predictions = {}
    for line in a.predictions.open(encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            predictions[str(r["id"])] = r
    # A question that ran out where no prediction was written still gets its row: a timeout, as the harness writes.
    timed_out = [q for q, t in times.items() if t["timeout"] and q not in predictions]
    handled = json.loads(a.exported.read_text(encoding="utf-8")) if a.exported else {}
    if handled and handled.get("set") != a.set:
        p.error(f"{a.exported} is the export of {handled.get('set')!r}, not {a.set!r}")
    missing = dict(handled.get("missing") or {})                 # no subgraph: never ran
    unanswered = [q for q in handled.get("exported") or [] if q not in predictions and q not in timed_out]
    topics = handled.get("topic") or {}
    out = Path(a.results) / a.run / a.model / f"{a.set}-test" / "gcr"
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.json").write_text(json.dumps(        # what `python -m harness score` reads, as the runner writes it
        {"system": "gcr", "stop": "", "framework": "gcr", "dataset": a.set, "split": "test", "model": a.model,
         "timeout_seconds": settings.TIMEOUT_SECONDS}, indent=1) + "\n")
    n = hits = over = untimed_n = lost_n = 0
    with (out / "s0.jsonl").open("w", encoding="utf-8") as f:
        for qid in dict.fromkeys([*predictions, *timed_out, *unanswered, *missing]):   # one row per question
            q = gold.get(qid)
            if q is None:
                continue
            r = predictions.get(qid, {})
            t = times.get(qid, {"timeout": False})
            timeout = t["timeout"] or round(t.get(1, 0.0) + t.get(2, 0.0), 3) > settings.TIMEOUT_SECONDS
            lost = "no_subgraph" if qid in missing else "no_prediction" if qid in unanswered else None
            untimed = not timeout and not lost and not (1 in t and 2 in t)   # a prediction this run did not time
            wall = float(settings.TIMEOUT_SECONDS) if untimed else round(t.get(1, 0.0) + t.get(2, 0.0), 3)
            miss = timeout or untimed or bool(lost)
            topic = topics.get(qid)
            answer = "" if miss else as_text(r.get("prediction"))
            items = None if miss else as_items(r.get("prediction"))
            hit, em, prec, rec, f1 = _score_text_answer(answer, q["answers"], items) if answer else (0, 0, 0, 0, 0)
            n += 1
            hits += hit
            over += timeout
            untimed_n += untimed
            lost_n += bool(lost)
            f.write(json.dumps({"qid": qid, "question": q["question"], "answer": answer,
                                "answers": q["answers"], "framework": "gcr", "engine": a.model,
                                "benchmark": f"{a.set}-test",
                                "error": "timeout" if timeout else "untimed" if untimed else lost,
                                "wall_time_s": wall,
                                "fw_metrics": {"gcr_input_chars": len(str(r.get("input", ""))),
                                               "gcr_paths": paths.get(qid, 0),
                                               "no_paths": paths.get(qid, 0) == 0,
                                               "gcr_step_seconds": [t.get(1), t.get(2)],
                                               "untimed": untimed,
                                               **({"gcr_topic_nodes": topic["nodes"],
                                                   "gcr_topic_absent": topic["absent"],
                                                   "gcr_topic_mapped": topic["mapped"]} if topic else {}),
                                               **({"answer_items": items} if items is not None else {})},
                                "eval": {"hit@1": hit, "exact_match": em, "precision": prec, "recall": rec,
                                         "f1": f1},
                                "manifest": manifest}, ensure_ascii=False) + "\n")
    print(f"[gcr convert] {a.set} {a.model}: {n} rows ({over} over {settings.TIMEOUT_SECONDS} s, {untimed_n} untimed, "
          f"{len(missing)} with no subgraph, {lost_n - len(missing)} with no prediction) "
          f"-> {out}/s0.jsonl, Hits@1 {hits / max(n, 1):.3f}")
    if untimed_n:
        print(f"[gcr convert] WARNING: {untimed_n} predictions have no time from this run (a reused output folder?); "
              f"scored as misses", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
