"""`score`, one CSV row per run, and `compare`, the paired difference of two runs (README: scoring)."""
from __future__ import annotations

import json
import os

import numpy as np

from harness import dataset as ds
from harness.metrics import efficiency as cost
from harness.metrics import reasoning
from harness.metrics import unseen as knowledge_gap
from harness.metrics.common import extract
from harness.metrics.common.answers import FREE, read_cache
from harness.metrics.common.cells import one_source
from harness.metrics.groundedness import grounding
from harness.metrics.reasoning import bootstrap_ci

COLUMNS = ("backbone", "dataset", "split", "system", "stop", "status", "n", "scored", "hit", "hit_lo", "hit_hi",
           "f1", "f1_lo", "f1_hi", "precision", "recall", "answered_rate", "empty_rate", "error_rate",
           "timeout_rate", "unscored", "support", "support_note", "llm_calls", "input_tokens", "output_tokens",
           "graph_calls", "latency_median_s", "latency_p90_s")


class RunError(ValueError):
    """A folder that is not a run, or a run that cannot be scored as asked."""


def run_info(run_dir: str) -> dict:
    path = os.path.join(run_dir, "run.json")
    if not os.path.exists(path):
        raise RunError(f"{run_dir} is not a run folder (no run.json)")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def cell_of(run_dir: str, root: str | None = None):
    info = run_info(run_dir)
    questions = ds.questions(info["dataset"], info["split"], root)
    return info, one_source(questions, run_dir, "run")


def extracts_of(run_dir: str, info: dict, cell, run_extractor: bool = True) -> dict | None:
    """The extractor's cached items for a free-text run ({qid: entry}), extracting the rows it has not read yet."""
    if info["framework"] not in FREE:
        return None
    path = os.path.join(run_dir, extract.CACHE_NAME)
    rows = [row for row, *_ in cell.rows.values() if row is not None]
    if run_extractor:
        model = os.environ.get("EXTRACTOR_MODEL", "").strip()
        urls = [u.strip() for u in os.environ.get("EXTRACTOR_BASE_URL", "").split(",") if u.strip()]
        have = extract.load_cache(path, extract.prompt_hash(), model) if model else {}
        todo = [r for r in rows if str(r.get("qid")) not in have
                and not isinstance((r.get("fw_metrics") or {}).get("answer_items"), list)]
        if todo:
            if not model or not urls:
                raise RunError(f"{info['framework']} answers in free text, read through the extractor: set "
                               "EXTRACTOR_BASE_URL and EXTRACTOR_MODEL (or pass --no-extract: those rows stay "
                               "unscored)")
            extract.run_pass(rows, path, extract.Servers(urls), model)
    return read_cache(path)[0]


def _pct(x) -> str:
    return "" if x is None else f"{100 * x:.1f}"


def _num(x, digits: int) -> str:
    return "" if x is None else f"{x:.{digits}f}"


def score_run(run_dir: str, run_extractor: bool = True, root: str | None = None) -> dict:
    """One CSV row (COLUMNS) for one run."""
    info, cell = cell_of(run_dir, root)
    system, dataset = info["framework"], info["dataset"]
    scores = reasoning.cell_scores(system, cell, extracts_of(run_dir, info, cell, run_extractor))
    per = list(scores["per"].values())
    scored = [p for p in per if p["scored"]]
    n = len(per)
    hit = np.array([p["hit"] for p in scored], dtype=float)
    f1 = np.array([p["f1"] for p in scored], dtype=float)
    hit_ci, f1_ci = bootstrap_ci(hit), bootstrap_ci(f1)
    present = sum(row is not None for row, *_ in cell.rows.values())
    timeouts = sum(p["miss"] in ("timeout", "over budget") for p in per)
    errors = sum(p["status"] == "error" for p in per) - sum(p["miss"] == "timeout" for p in per)
    if scores["complete"]:
        state = "complete"
    elif scores["counts"].get("not run (extractor)"):
        state = f"extractor not run on {scores['counts']['not run (extractor)']} rows"
    else:
        state = f"running: {present} of {n} rows"
    sup = grounding.cell_support(system, dataset, cell, scores, name_anchor=True)
    if sup.get("why"):
        note = sup["why"]
    elif sup.get("support") is None:
        note = "no answered question with graph evidence"
    else:
        note = f"answered with evidence: {sup['counts'].get('answered', 0)}"
    spend = cost.pool(cost.cell_costs(system, cell), n)
    return {
        "backbone": info.get("model", ""), "dataset": dataset, "split": info["split"], "system": info["system"],
        "stop": info["stop"], "status": state, "n": n, "scored": len(scored),
        "hit": _pct(scores["hit"]), "hit_lo": _pct(hit_ci[0] if hit_ci else None),
        "hit_hi": _pct(hit_ci[1] if hit_ci else None),
        "f1": _pct(scores["f1"]), "f1_lo": _pct(f1_ci[0] if f1_ci else None), "f1_hi": _pct(f1_ci[1] if f1_ci else None),
        "precision": _pct(scores["p"]), "recall": _pct(scores["r"]),
        "answered_rate": _pct(sum(p["scored"] and p["status"] == "ok" and p["size"] > 0 for p in per) / n if n else None),
        "empty_rate": _pct(sum(p["scored"] and p["status"] == "ok" and not p["size"] for p in per) / n if n else None),
        "error_rate": _pct(errors / n if n else None), "timeout_rate": _pct(timeouts / n if n else None),
        "unscored": n - len(scored),
        "support": _pct(sup.get("support")), "support_note": note,
        "llm_calls": _num(spend.get("calls"), 2), "input_tokens": _num(spend.get("input"), 0),
        "output_tokens": _num(spend.get("output"), 0), "graph_calls": _num(spend.get("graph"), 1),
        "latency_median_s": _num(spend.get("latency_median"), 1), "latency_p90_s": _num(spend.get("latency_p90"), 1),
    }


def _unseen_halves(a: dict, b: dict) -> bool:
    names = {a["dataset"], b["dataset"]}
    base = a["dataset"].split("-")[0]
    return a["split"] == b["split"] and names == {f"{base}-unseen-real", f"{base}-unseen"}


def compare(run_a: str, run_b: str, run_extractor: bool = True, root: str | None = None) -> list:
    """Rows {metric, a, b, difference, ci95, n[, McNemar]}: A minus B, paired by question or by pair."""
    out = []
    infos, scores = [], []
    for d in (run_a, run_b):
        info, cell = cell_of(d, root)
        infos.append(info)
        scores.append(reasoning.cell_scores(info["framework"], cell, extracts_of(d, info, cell, run_extractor)))
    a, b = infos
    if _unseen_halves(a, b):
        real_first = a["dataset"].endswith("-real")
        real, twin = (scores[0], scores[1]) if real_first else (scores[1], scores[0])
        pair_ids = [(str(t["id"]), str(r["id"])) for t, r in ds.pairs(a["dataset"], a["split"], root)]
        for metric in ("hit", "f1"):
            k = knowledge_gap.ka(real, twin, pair_ids, metric)
            sign = 1 if real_first else -1
            ci = k["ci"] and tuple(sorted((sign * k["ci"][0], sign * k["ci"][1])))
            row = {"metric": metric, "a": k["real"] if real_first else k["twin"],
                   "b": k["twin"] if real_first else k["real"],
                   "difference": None if k["gap"] is None else sign * k["gap"], "ci95": ci, "n": k["pairs_scored"]}
            if metric == "hit":
                row.update(b_only=k["b"], c_only=k["c"], mcnemar_p=k["p"])
            out.append(row)
        return out
    if (a["dataset"], a["split"]) != (b["dataset"], b["split"]):
        raise RunError("the two runs are on different question sets (and not the two halves of one Unseen set)")
    common = [q for q, p in scores[0]["per"].items() if p["scored"] and scores[1]["per"][q]["scored"]]
    for metric in ("hit", "f1"):
        va = np.array([scores[0]["per"][q][metric] for q in common], dtype=float)
        vb = np.array([scores[1]["per"][q][metric] for q in common], dtype=float)
        out.append({"metric": metric, "a": float(va.mean()) if len(common) else None,
                    "b": float(vb.mean()) if len(common) else None,
                    "difference": float((va - vb).mean()) if len(common) else None,
                    "ci95": bootstrap_ci(va - vb), "n": len(common)})
    return out
