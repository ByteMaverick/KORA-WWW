"""Every stop decision of a stop arm, one JSON line per judged candidate (README: the verdicts file)."""
from __future__ import annotations

import json
import os
import threading
import time

#: Names and ids kept per candidate: the gold's recall reads every item, its precision the first 50.
CAP = 1000

_LOCK = threading.Lock()
_SEQ: dict[str, int] = {}


def _safe(qid: str) -> str:
    """The question's file name, as harness/llm._trace_call writes it."""
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in (qid or "no-qid"))


def path_for(qid: str | None = None) -> str | None:
    """<HARNESS_TRACE_DIR>/<qid>.verdicts.jsonl for this question (TRACE_QID when not given), or None when unset."""
    trace_dir = os.getenv("HARNESS_TRACE_DIR", "")
    if not trace_dir:
        return None
    if qid is None:
        from harness import llm
        qid = llm.TRACE_QID
    return os.path.join(trace_dir, f"{_safe(qid)}.verdicts.jsonl")


def server_refused(p) -> bool:
    """The estimator server scores a state it refuses exactly 0.0, which no softmax or sigmoid reaches."""
    return p is not None and float(p) == 0.0


def _capped(candidate):
    if not isinstance(candidate, dict):
        return candidate
    out = dict(candidate)
    for key in ("names", "ids"):
        if isinstance(out.get(key), (list, tuple)):
            out[key] = [str(x) for x in list(out[key])[:CAP]]
    return out


def record(arm: str, stop: str, candidate, accept: bool, *, score=None, judge_url=None, threshold=None,
           refused: bool | None = False, gold_hit=None, depth=None, step=None, error=None, **more) -> None:
    """Append one decision on one candidate to the question's verdicts file; never raises but for the alarm."""
    try:
        from harness import llm
        qid = llm.TRACE_QID
        path = path_for(qid)
        if path is None:
            return
        line = {"qid": qid, "seq": None, "ts": time.time(), "pid": os.getpid(), "arm": arm, "stop": stop,
                "judge_url": judge_url, "threshold": threshold, "depth": depth, "step": step,
                "candidate": _capped(candidate), "score": score, "accept": bool(accept),
                "refused": None if refused is None else bool(refused),
                "gold_hit": gold_hit, "error": error, **more}
        with _LOCK:
            if path not in _SEQ:
                try:
                    with open(path, "rb") as f:
                        _SEQ[path] = sum(1 for _ in f)
                except FileNotFoundError:
                    _SEQ[path] = 0
            line["seq"] = _SEQ[path]
            text = json.dumps(line, ensure_ascii=False, default=str) + "\n"
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(text)
            _SEQ[path] += 1
    except Exception:
        pass


def judged(arm: str, candidates: list, scores: list, chosen, *, judge_url, threshold, error=None, depth=None,
           step=None, **more) -> None:
    """One judge decision over `candidates`: a line each, `accept` on index `chosen`; none: one line."""
    try:
        if not candidates:
            record(arm, "judge", None, False, judge_url=judge_url, threshold=threshold, depth=depth, step=step,
                   index=None, n_candidates=0, error=error, **more)
            return
        for k, (candidate, p) in enumerate(zip(candidates, scores)):
            refused = p is None or server_refused(p)
            why = (error or "the judge gave no score") if p is None else (
                "the server scored it exactly 0.0: its renderer refused the state" if server_refused(p) else None)
            record(arm, "judge", candidate, k == chosen, score=p, judge_url=judge_url, threshold=threshold,
                   refused=refused, depth=depth, step=step, index=k, n_candidates=len(candidates), error=why, **more)
    except Exception:
        pass


def gold_entry(entry: dict | None) -> dict | None:
    """The gold's figures for one decision, from a StopOracle log entry."""
    if not isinstance(entry, dict):
        return None
    return {k: entry.get(k) for k in ("precision", "recall", "f1", "hit", "rule", "items_from")}


def golded(arm: str, candidates: list, verdicts: list, entries: list, chosen, *, error=None, depth=None, step=None,
           **more) -> None:
    """One gold decision over `candidates`: the gold's verdict on each, `accept` on index `chosen`."""
    try:
        if not candidates:
            record(arm, "gold", None, False, refused=None, depth=depth, step=step, index=None, n_candidates=0,
                   error=error, **more)
            return
        for k, candidate in enumerate(candidates):
            verdict = verdicts[k] if k < len(verdicts) else None
            entry = entries[k] if k < len(entries) else None
            record(arm, "gold", candidate, k == chosen, refused=None, gold_hit=None if verdict is None else bool(verdict),
                   gold=gold_entry(entry), depth=depth, step=step, index=k, n_candidates=len(candidates),
                   error=error if verdict is None else None, **more)
    except Exception:
        pass
