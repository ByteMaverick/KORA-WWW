"""Answer correctness: normalization, matching, Hits@Any and F1, the text scorer, the reading rule, the statistics."""
from __future__ import annotations

import glob
import json
import math
import os
import re
import string
import unicodedata
from collections import Counter
from functools import lru_cache
from typing import List, Tuple

import numpy as np

from harness.settings import TIMEOUT_SECONDS

_PUNCT = set(string.punctuation)
_ARTICLES = re.compile(r"\b(a|an|the)\b")


def normalize(s) -> str:
    s = "".join(c for c in str(s).lower() if c not in _PUNCT and not unicodedata.category(c).startswith("P"))
    return " ".join(_ARTICLES.sub(" ", s).split())


def contains(item_norm: str, gold_norm: str) -> bool:
    """The gold's tokens appear in the item's tokens as one contiguous run (whole words)."""
    g, i = gold_norm.split(), item_norm.split()
    return bool(g) and any(i[k:k + len(g)] == g for k in range(len(i) - len(g) + 1))


def match(item: str, gold: str) -> bool:
    """Whether one answer item matches one gold answer (module docstring)."""
    return contains(normalize(item), normalize(gold))


def max_matching(edges: list) -> int:
    """The size of a maximum bipartite matching; `edges[i]` lists the golds item i matches (Kuhn's algorithm)."""
    owner: dict = {}

    def augment(i, seen):
        for g in edges[i]:
            if g not in seen:
                seen.add(g)
                if g not in owner or augment(owner[g], seen):
                    owner[g] = i
                    return True
        return False
    return sum(augment(i, set()) for i in range(len(edges)))


def golds_of(golds) -> list:
    """The gold answers as scored: non-empty after normalization, one each (equal normalized golds are one gold)."""
    out = {}
    for g in golds or []:
        if g is not None and normalize(g):
            out.setdefault(normalize(g), str(g))
    return list(out.values())


def score(items: list, golds) -> dict:
    """Every metric of one question. Items normalizing to nothing are not items (dropped, counted in `dropped`)."""
    kept = [(str(i), normalize(i)) for i in items]
    kept = [(i, n) for i, n in kept if n]
    gold = [(g, normalize(g)) for g in golds_of(golds)]
    edges = [[k for k, (g, gn) in enumerate(gold) if contains(n, gn)] for i, n in kept]
    m = max_matching(edges)
    p = m / len(kept) if kept else 0.0
    r = m / len(gold) if gold else 0.0
    hit = float(m >= 1)
    return {"hit": hit, "p": p, "r": r, "f1": 2 * p * r / (p + r) if p + r else 0.0, "size": len(kept),
            "hit1": float(bool(kept) and bool(edges[0])), "hit_capped": hit if len(kept) <= 2 * len(gold) else 0.0,
            "dropped": len(items) - len(kept), "no_gold": not gold}


METRICS = ("hit", "f1", "p", "r", "size", "hit1", "hit_capped")


def question_scores(system: str, row, q: dict, extract=None) -> dict:
    """One question: its answer set, its metrics (None when not scored), its status and kind of miss."""
    from .common.answers import answer_set      # imported here to break the import cycle with common/answers.py
    ans = answer_set(system, row, extract)
    out = {"status": ans.status, "miss": miss_kind(row), "how": ans.how, "counts": ans.counts, "items": ans.items,
           "ids": ans.ids, "scored": row is not None and ans.items is not None}
    golds = harness_golds(q)
    if out["scored"]:
        out.update(score(ans.items, golds))
    return out


def cell_scores(system: str, cell, extracts: dict | None = None) -> dict:
    """Every question of a cell, the means over scored questions, size median and p90, counts, `complete`."""
    per = {qid: question_scores(system, cell.rows[qid][0], q, (extracts or {}).get(qid))
           for qid, q in cell.questions.items()}
    scored = [p for p in per.values() if p["scored"]]
    counts = Counter()
    for p in per.values():
        counts[p["status"]] += 1
        counts.update({k: v for k, v in p["counts"].items() if v})
        if p["miss"]:
            counts[f"miss: {p['miss']}"] += 1
        if p["how"] == "not run":
            counts["not run (extractor)"] += 1
        if p["how"] == "fallback":
            counts["extractor fallback split"] += 1
        if p["scored"] and p["status"] == "ok" and not p["size"]:
            counts["empty answer"] += 1
        if p["scored"]:
            counts["no gold"] += p["no_gold"]
            counts["items dropped (empty after normalization)"] += p["dropped"]
    means = {m: (float(np.mean([p[m] for p in scored])) if scored and all(p[m] is not None for p in scored) else None)
             for m in METRICS}
    sizes = [p["size"] for p in scored]
    return {"per": per, "n": len(per), "scored": len(scored), **means,
            "size_median": float(np.median(sizes)) if sizes else None,
            "size_p90": float(np.percentile(sizes, 90)) if sizes else None,
            "complete": len(scored) == len(per) and len(per) > 0, "counts": {k: v for k, v in counts.items() if v}}


# The reading rule (which rows are misses) and the statistics.
REPS, SEED = 10_000, 0
_ANSWER_IS = re.compile(r"answer(?: to (?:the|this) question)? is[:\s]+(.+?)\s*$", re.I | re.S)
#: Connection and server failures, and an LLM client's connection error.
INFRA = re.compile(r"Connection timed out|Errno 110|Errno 111|Connection refused|Connection reset|HTTPConnectionPool|"
                   r"EndPointInternalError|Max retries exceeded|Remote end closed|RemoteDisconnected|502 Bad Gateway|"
                   r"503 Service|504 Gateway|ServiceUnavailable|Name or service not known|ConnectionError|"
                   r"APIConnectionError|Read timed out|LLMError.{0,80}connect", re.I | re.S)
MISS_KINDS = ("infrastructure", "timeout", "other error", "over budget")


def status(row: dict | None) -> str:
    """ok, missing, error or over_budget."""
    if row is None:
        return "missing"
    if row.get("error"):
        return "error"
    # The paper's fixed 200 s, whatever --timeout the run used.
    return "over_budget" if (row.get("wall_time_s") or 0) > TIMEOUT_SECONDS else "ok"


def miss_kind(row: dict | None) -> str | None:
    """What kind of miss a present row is (None: not a miss, or missing)."""
    st = status(row)
    if st == "over_budget":
        return "over budget"
    if st != "error":
        return None
    e = str(row.get("error"))
    if INFRA.search(e):
        return "infrastructure"
    return "timeout" if "timeout" in e.lower() or "timed out" in e.lower() else "other error"


def stated(text: str) -> str:
    """A reply cut to its stated answer: the text after its last "answer is", else records.stated_answer."""
    from .groundedness.records import stated_answer   # imported here to break the import cycle with records.py
    m = None
    for m in _ANSWER_IS.finditer(text):
        pass
    if m is not None:
        return m.group(1).replace("**", "").strip().rstrip(".")
    return stated_answer(text.strip())


@lru_cache(maxsize=8)
def _resamples(n: int, reps: int, seed: int):
    return np.random.default_rng(seed).integers(0, n, size=(reps, n))


def bootstrap_ci(diff, reps: int = REPS, seed: int = SEED) -> tuple | None:
    """The percentile 95% interval of the mean of the paired differences."""
    diff = np.asarray(diff, dtype=float)
    if not len(diff):
        return None
    means = diff[_resamples(len(diff), reps, seed)].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def mcnemar(b: int, c: int) -> float:
    """McNemar's exact two-sided p from the discordant pairs b and c."""
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)


# The text scorer (the rows' own `eval` field, the stop oracle, the authors' Hit) and the run's live shard files.
#: Bumped whenever a rule here changes its output for any input (score_text, hit_first, hit_authors).
SCORER_VERSION = "1"

_LIVE = re.compile(r"s(\d+)\.jsonl")


class DuplicateRowError(ValueError):
    """Two live rows for one question in one cell."""


def label_key(s: str) -> str:
    """The text scorer's normalization (lower case, no articles, punctuation to spaces); also the label key."""
    s = s.lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def mentions(text: str, gold: str) -> bool:
    """Whether a normalised gold appears in normalised text as whole words ("9" is not in "1990")."""
    return bool(gold) and re.search(rf"(?<!\w){re.escape(gold)}(?!\w)", text) is not None


def score_text(answer: str, golds: List[str], listed: List[str] | None = None) -> Tuple[float, float, float, float, float]:
    """Hit, exact match, precision, recall and F1 of a free-text answer (README: reasoning, the text scorer)."""
    text = label_key(answer)
    norm_golds = [g for g in dict.fromkeys(label_key(str(a)) for a in golds) if g]
    if not text or not norm_golds:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    found = [g for g in norm_golds if mentions(text, g)]
    hit = 1.0 if found else 0.0
    em = 1.0 if text in norm_golds else 0.0
    parts = items(answer, golds, listed)
    matched = sum(1 for i in parts if any(mentions(i, g) for g in norm_golds))
    prec = matched / len(parts) if parts else 0.0
    rec = len(found) / len(norm_golds)
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return hit, em, prec, rec, f1


def items(answer: str, golds: List[str] = (), listed: List[str] | None = None) -> list[str]:
    """The answer's items: the system's own list, else the text split on , ; | and lines, golds kept whole."""
    if listed is not None:
        return [i for i in (label_key(str(x)) for x in listed) if i]
    text = answer or ""
    for g in sorted({str(g) for g in golds if re.search(r"[,;|]", str(g))}, key=len, reverse=True):
        text = re.sub(re.escape(g), lambda m: re.sub(r"[,;|]", " ", m.group(0)), text, flags=re.I)
    return [i for i in (label_key(x) for x in re.split(r"[,;\n|]+", text)) if i]


def hit_first(answer: str, golds: List[str], listed: List[str] | None = None) -> float:
    """First-item Hit: the FIRST item the answer lists names a gold, so a list dump is not a hit."""
    first = items(answer, golds, listed)[:1]
    norm_golds = [g for g in (label_key(str(a)) for a in golds) if g]
    return float(bool(first) and any(mentions(first[0], g) for g in norm_golds))


def _exact_match(pred: str, gold: str) -> bool:
    # SoG's exact_match, verbatim (its helper's :417-422); ToG's and PoG's are the same rule.
    pred_clean = pred.strip().lower().replace(" ", "")
    gold_clean = gold.strip().lower().replace(" ", "")
    return (
        pred_clean == gold_clean or pred_clean in gold_clean or gold_clean in pred_clean
    )


def hit_authors(answer: str, golds: List[str]) -> float:
    """The ToG / PoG / SoG authors' Hits@1, as SoG's main.py scores it (README: reasoning, the text scorer)."""
    if not answer:
        return 0.0
    braced = [m.strip() for m in re.findall(r"\{([^}]*)\}", answer)]
    preds = braced or [answer]
    return float(any(_exact_match(p, str(g)) for p in preds for g in golds))


def listed_of(row: dict) -> list | None:
    """The list a system returned for this row, if it returned one (harness/runner._dispatch keeps it)."""
    return ((row or {}).get("fw_metrics") or {}).get("answer_items")


def harness_golds(q: dict) -> list:
    """What a row is scored against, as the harness scores it: the text labels, or the ids when there are none."""
    labels = [a for a in (q.get("answers") or []) if a is not None and str(a).strip()]
    return labels or [a for a in (q.get("answer_ids") or []) if a is not None and str(a).strip()]


def live_files(cell: str) -> list[str]:
    """The live shard files of a cell, in shard order."""
    found = [(int(m.group(1)), f) for f in glob.glob(os.path.join(cell, "s*.jsonl"))
             if (m := _LIVE.fullmatch(os.path.basename(f)))]
    return [f for _, f in sorted(found)]


def read_rows(cell: str) -> dict:
    """{qid: row} of a cell's live shard files; a question with two live rows raises DuplicateRowError."""
    out: dict = {}
    for f in live_files(cell):
        for n, line in enumerate(open(f, encoding="utf-8"), 1):
            if not line.strip():
                continue
            r = json.loads(line)
            qid = str(r.get("qid"))
            if qid in out:
                raise DuplicateRowError(f"{qid} has two live rows in {cell} (second at {os.path.basename(f)}:{n})")
            out[qid] = r
    return out
