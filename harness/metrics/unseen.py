"""The Unseen gap: real minus twin on the paired questions, its interval and McNemar's test (README: unseen)."""
from __future__ import annotations

import numpy as np

from harness.metrics.reasoning import REPS, SEED, bootstrap_ci, mcnemar


def ka(real: dict, twin: dict, pair_ids: list, metric: str, mask=None, reps: int = REPS, seed: int = SEED) -> dict:
    """The gap on `metric` over the scored pairs [(twin id, real id)]; McNemar's b, c and p for hit."""
    keep = [(t, r) for k, (t, r) in enumerate(pair_ids) if (mask is None or mask[k])
            and real["per"][r]["scored"] and twin["per"][t]["scored"]]
    rv = np.array([real["per"][r][metric] for _, r in keep], dtype=float)
    tv = np.array([twin["per"][t][metric] for t, _ in keep], dtype=float)
    out = {"pairs_scored": len(keep), "real": float(rv.mean()) if len(keep) else None,
           "twin": float(tv.mean()) if len(keep) else None}
    out["gap"] = None if out["real"] is None else out["real"] - out["twin"]
    out["ci"] = bootstrap_ci(rv - tv, reps, seed)
    if metric == "hit":
        b, c = int(((rv == 1) & (tv == 0)).sum()), int(((rv == 0) & (tv == 1)).sum())
        out.update(b=b, c=c, p=mcnemar(b, c))
    return out


def excludes_zero(ci) -> bool:
    return bool(ci) and (ci[0] > 0 or ci[1] < 0)
