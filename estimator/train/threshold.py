"""The estimator's threshold: the cut on validation scores at which the larger of the FN and FP rates is smallest,
searched over 401 quantiles of the scores.

    python -m estimator.train.threshold --scores validation_scores.npy --labels VAL.jsonl [--out threshold.json]

--scores: a .npy or .json list of P(yes); --labels: a .npy or .json list of 0/1, or a rows file (.jsonl/.parquet)
with a `label` column, in the same order.
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def balanced_threshold(y, s):
    best = (2.0, 0.5)
    for t in np.unique(np.quantile(s, np.linspace(0, 1, 401))):
        p = s >= t
        best = min(best, (max(((~p) & (y == 1)).sum() / max((y == 1).sum(), 1), (p & (y == 0)).sum() / max((y == 0).sum(), 1)), float(t)))
    return best[1]


def metrics(y, s, t):
    from sklearn.metrics import roc_auc_score
    p = s >= t; tp = int((p & (y == 1)).sum()); fp = int((p & (y == 0)).sum()); fn = int(((~p) & (y == 1)).sum())
    return {"n": len(y), "FN rate": fn / max((y == 1).sum(), 1), "FP rate": fp / max((y == 0).sum(), 1),
            "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1), "F1": 2 * tp / max(2 * tp + fp + fn, 1),
            "accuracy": float((p == (y == 1)).mean()), "AUROC": roc_auc_score(y, s) if 0 < y.sum() < len(y) else np.nan}


def _array(path: str) -> np.ndarray:
    if path.endswith(".npy"):
        return np.load(path)
    if path.endswith(".json"):
        return np.asarray(json.load(open(path)))
    from estimator.train.rename import read_rows
    return read_rows(path).label.to_numpy()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m estimator.train.threshold")
    ap.add_argument("--scores", required=True); ap.add_argument("--labels", required=True)
    ap.add_argument("--out", help='write {"tau": ...} here (a checkpoint\'s threshold.json)')
    a = ap.parse_args(argv)
    s = _array(a.scores).astype(float); y = _array(a.labels).astype(int)
    if len(s) != len(y):
        ap.error(f"{len(s)} scores but {len(y)} labels")
    t = balanced_threshold(y, s)
    if a.out:
        json.dump({"tau": float(t)}, open(a.out, "w"))
    print(json.dumps({"tau": float(t), **{k: float(v) for k, v in metrics(y, s, t).items()}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
