"""The Stopping Estimator's client: score claims / path / candidate answer with the learned estimator."""
from __future__ import annotations

import json
import threading
import time
import urllib.request

SHOWN = 15


class EstimatorUnavailable(RuntimeError):
    """The learned judge is the judge and cannot be reached or configured: KORA stops instead of judging otherwise."""


def enabled() -> bool:
    """The learned judge decides (settings.judge_impl, `KORA_JUDGE_IMPL`, estimator when unset)."""
    from kora.app import judge_impl
    return judge_impl() == "estimator"


def url() -> str:
    """The judge server's base URL (settings.estimator_url), or EstimatorUnavailable naming every way to give one."""
    from kora.app import estimator_url
    found = estimator_url()
    if not found:
        raise EstimatorUnavailable(
            "the Stopping Estimator is the judge (KORA_JUDGE_IMPL is unset or estimator) but no estimator server was "
            "found: set ESTIMATOR_URL, or set KORA_JUDGE_IMPL=llm for the LLM path judge")
    return found.rstrip("/")


_SERVED: dict[str, float] = {}


def server_threshold(base: str) -> float:
    """The threshold the server reports on GET /health (its checkpoint's `threshold.json`), asked once per URL."""
    if base in _SERVED:
        return _SERVED[base]
    try:
        with urllib.request.urlopen(base + "/health", timeout=30) as response:
            info = json.load(response)
    except Exception as error:  # noqa: BLE001 - reported, never defaulted
        raise EstimatorUnavailable(f"the judge server at {base} did not answer /health ({error!r}): start it, set "
                              "ESTIMATOR_URL to a live one, or set KORA_JUDGE_IMPL=llm") from error
    value = info.get("threshold") if isinstance(info, dict) else None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise EstimatorUnavailable(f"the judge server at {base} reports no threshold on /health (got {value!r}): serve a "
                              "checkpoint whose threshold.json has one, or set ESTIMATOR_TAU explicitly")
    _SERVED[base] = float(value)
    return _SERVED[base]


def threshold() -> float:
    """The cut p(sufficient) must reach: `ESTIMATOR_TAU` when set explicitly, else the server's."""
    from kora.app import estimator_tau
    explicit = estimator_tau()
    return explicit if explicit is not None else server_threshold(url())


def check() -> None:
    """At startup: with the learned judge as the judge, its URL and threshold resolve, or EstimatorUnavailable."""
    if enabled():
        url()
        threshold()


def __getattr__(name: str):
    if name == "THRESHOLD":
        return threshold()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def state_of(claims: str, path: str, count: int, kind: str, wanted: str, candidates: str) -> str:
    """The text the checkpoint was trained on: the three blocks of sufficiency.verify."""
    return (f"CLAIMS\n{claims}\n\nEVIDENCE (the path that was traversed)\n{path}\n\n"
            f"CANDIDATE ANSWER ({count} entities, kind expected: {kind}; the question wants {wanted})\n{candidates}")


def entity_lines(pairs: list[tuple[str, str]], total: int) -> str:
    """`  label (id)` per entity, first SHOWN, and the same "not missing" note the verify prompt uses."""
    if not pairs:
        return "  (none)"
    lines = [f"  {label} ({ident})" for label, ident in pairs[:SHOWN]]
    if total > SHOWN:
        lines.append(f"  (showing {SHOWN} of {total} entities; the other {total - SHOWN} are part of the same set "
                     f"and are not shown to you. They are not missing.)")
    return "\n".join(lines)


EXTRA_IDS = 30
LABEL_IDS = 1000


def extra(ids: list, query: str | None = None, chain: list[dict] | None = None) -> dict:
    """What the server needs beyond the text: candidate ids, their count, the query and its chain."""
    distinct = list(dict.fromkeys(str(i) for i in ids))
    return {"ids": distinct[:EXTRA_IDS], "n": len(distinct), "query": query, "chain": list(chain or []),
            "all_ids": distinct[:LABEL_IDS]}


def score(states: list[str], questions: list[str] | None = None,
          extras: list[dict] | None = None) -> tuple[list[float | None], float]:
    """P(sufficient) per state, or None where the call failed; and the seconds spent."""
    endpoint = url() + "/score"
    body = json.dumps({"states": states, "questions": list(questions) if questions else None,
                       "extras": extras}).encode()
    started = time.time()
    for attempt in range(4):
        try:
            request = urllib.request.Request(endpoint, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=60) as response:
                probabilities = json.load(response)["p"]
            if len(probabilities) == len(states):
                return [float(p) for p in probabilities], time.time() - started
        except Exception as error:  # noqa: BLE001 - retried, then reported as no
            last = error
            time.sleep(1 + attempt)
    print(f"[estimator] scoring failed after retries: {last!r}", flush=True)
    return [None] * len(states), time.time() - started


_BEST: dict[str, tuple[float, list, list]] = {}
_LOCK = threading.Lock()


def reset_best() -> None:
    with _LOCK:
        _BEST.clear()


def remember(task_id: str, p: float | None, ids: list, names: list) -> None:
    if p is None or not ids:
        return
    with _LOCK:
        current = _BEST.get(task_id)
        if current is None or p > current[0]:
            _BEST[task_id] = (float(p), list(ids), list(names))


def best(task_id: str) -> tuple[float, list, list] | None:
    with _LOCK:
        return _BEST.get(task_id)
