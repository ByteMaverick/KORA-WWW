"""KORA run by the harness: this repository's `kora` package, unchanged (README: KORA under the harness)."""
from __future__ import annotations

import argparse
import os
import re
import threading

from harness import verdict_log
from harness.llm import run_llm
from harness.runner import QueryMetrics

#: KORA's own clock, inside the harness's 200 s: the step in flight when it runs out needs room to finish.
SECONDS = 140.0   # fixed, not read from the environment
TOKENS = 400_000

#: Dataset -> KORA's graph shape and its endpoint's variable; the Unseen sets read their original's shape.
SHAPES = {
    "cwq": ("freebase-sparql", "FREEBASE_SPARQL_URL"), "webqsp": ("freebase-sparql", "FREEBASE_SPARQL_URL"),
    "grailqa": ("freebase-sparql", "FREEBASE_SPARQL_URL"),
    "metaqa": ("metaqa", "NEO4J_METAQA_URI"),
}
for _ds in ("webqsp", "cwq"):
    for _unseen in (f"{_ds}-unseen", f"{_ds}-unseen-real"):
        SHAPES[_unseen] = SHAPES[_ds]
#: KORA's row, kept in fw_metrics["kora"]: everything but the per-task attempts and the controller's log.
KEPT = ("status", "stop_reason", "phase", "hit", "oracle_hit", "first_is_gold", "recall", "precision", "f1",
        "answer_ids", "answer_order", "projection", "sufficiency", "judge", "prompts", "rounds", "dispatches",
        "validation", "decompositions", "tasks", "llm_calls", "kg_queries", "kg_calls", "embedding_calls",
        "tokens", "seconds", "judge_seconds", "error", "sets", "properties")

_harnesses: dict = {}
_lock = threading.Lock()


def dataset_of(args: argparse.Namespace, sample=None) -> str:
    name = (getattr(args, "dataset", "") or getattr(sample, "benchmark", "") or "").strip()
    if name not in SHAPES:
        raise NotImplementedError(f"[kora] no KORA graph for dataset {name!r}; one of {', '.join(SHAPES)}")
    return name


def environment(stop: str, dataset: str) -> dict[str, str]:
    """The settings one run of this stop rule on this dataset is read with (KORA reads them from the environment)."""
    from kora.app import STOP_MODES, stop_mode
    shape, variable = SHAPES[dataset]
    env = {"KORA_STOP": stop_mode(stop), **STOP_MODES[stop_mode(stop)]}
    env.setdefault("KORA_DECOMPOSER", "structural")
    env.setdefault("KORA_ORACLE", "generous")
    env["KORA_ROUNDS"] = "2"
    endpoint = os.environ.get("NEO4J_METAQA_URI" if shape.startswith("metaqa") else "SPARQL_ENDPOINT", "")
    if not endpoint:
        raise RuntimeError(f"[kora] no endpoint for {dataset}: the harness sets NEO4J_METAQA_URI or SPARQL_ENDPOINT")
    env[variable] = endpoint
    return env


#: KORA_* names the harness lets through from the environment: where things are (the embedding server, its cache).
INFRASTRUCTURE = ("KORA_EMBED_BASE_URL", "KORA_EMBED_CACHE_DIR", "KORA_EMBED_MODEL", "KORA_EMBED_QUERY_PREFIX")
#: KORA_* names only KORA's own entry point reads: under the harness the model client, the budget and the stop rule
#: are the harness's, so these never reach a run.
NOT_READ_HERE = ("KORA_LLM_PROVIDER", "KORA_LLM_BASE_URL", "KORA_LLM_MODEL", "KORA_REASONING", "KORA_LLM_RETRIES",
                 "KORA_LLM_TIMEOUT", "KORA_SECONDS", "KORA_TOKENS")
#: Settings KORA reads that the harness fixes (the stop rule from --stop, the paper's two query rounds): a different
#: value in the environment or .env is refused, never silently overridden.
FIXED = ("KORA_STOP", "KORA_ROUNDS")


def check_settings(env: dict[str, str]) -> dict[str, str]:
    """Refuse KORA settings that are not the stop rule's; return what KORA runs with, for the row."""
    disagree = sorted(k for k in FIXED if k in env and os.environ.get(k, "").strip()
                      and os.environ[k].strip().lower() != env[k])
    if disagree:
        raise RuntimeError("[kora] " + "; ".join(f"{k}={os.environ[k]} in the environment or .env disagrees with the "
                                                 f"harness's {k}={env[k]}" for k in disagree)
                           + " (KORA_STOP comes from --stop, KORA_ROUNDS is the paper's 2): unset it or make it agree")
    foreign = sorted(k for k in os.environ if k.startswith("KORA_") and k not in env and k not in INFRASTRUCTURE
                     and k not in NOT_READ_HERE and os.environ[k] != env.get(k))
    if foreign:
        raise RuntimeError(f"[kora] run settings in the environment that the stop rule does not set: "
                           f"{', '.join(foreign)}; the paper's KORA runs take theirs from --stop alone (unset them)")
    return {**{k: os.environ[k] for k in INFRASTRUCTURE if k in os.environ}, **env}


class HarnessLLM:
    """KORA's `LLMProvider`, answered by harness.llm.run_llm; every call counted under the KORA prompt that made it."""

    def __init__(self, engine: str, max_tokens: int, temperature: float = 0.0):
        from kora.prompts.registry import RENDERED
        from kora.tools.llm import LLMError, LLMResponse
        self.engine, self.max_tokens, self.temperature = engine, max_tokens, temperature
        self.model = engine
        self._rendered, self._error, self._response = RENDERED, LLMError, LLMResponse
        self.qm: QueryMetrics | None = None

    def generate(self, prompt: str, *, system: str | None = None):
        text, metrics = run_llm(prompt=prompt, engine=self.engine, temperature=self.temperature,
                                max_tokens=self.max_tokens, system_prompt=system or "")
        if self.qm is not None:
            self.qm.add(metrics, call_site=self._rendered.get() or "kora")
        if not metrics.success or not text:
            raise self._error(f"{self.engine} request failed: {metrics.error or 'empty response'}")
        return self._response(text=text, input_tokens=metrics.prompt_tokens or 0,
                              output_tokens=metrics.completion_tokens or 0,
                              latency_seconds=metrics.latency_seconds or 0.0, model=self.engine)


def llm_settings(engine: str) -> dict:
    """KORA's cap and temperature per call from the settings table: 3,000 tokens, on gpt-oss its allowance."""
    from harness.settings import decoding
    d = decoding("kora", engine)
    return {"max_tokens": d["max_tokens"], "temperature": d["temperature"]}


def harness_for(stop: str, dataset: str, args: argparse.Namespace):
    """One KORA harness per (stop rule, dataset) in this process, built once: graph, embedder, prompts."""
    key = (stop, dataset)
    with _lock:
        if key in _harnesses:
            return _harnesses[key]
        env = environment(stop, dataset)
        settings_in_use = check_settings(env)
        os.environ.update(env)
        from kora.run import Harness
        from kora.modules.controller.models import Budget
        from kora.modules.decomposition.decompose import StatedNames
        from kora.tools.ner import RecognizedAnchors
        shape, _ = SHAPES[dataset]
        harness = Harness(shape, judge=stop == "gold", budget=Budget(tokens=TOKENS, seconds=SECONDS))
        llm = HarnessLLM(args.engine, **llm_settings(args.engine))
        harness.llm = harness.judge_llm = llm
        harness.stated = StatedNames(RecognizedAnchors(llm, harness.resolver))
        harness.settings_in_use = settings_in_use
        _harnesses[key] = harness
        return harness


def item_of(sample, oracle=None) -> dict:
    """KORA's item; the gold is empty except under --stop gold, where it comes from the stop callback."""
    anchors = list((sample.topic_entity or {}).items())
    return {"q": sample.question, "gold": list(oracle.golds) if oracle is not None else [], "id": sample.qid,
            "hops": None,
            "needs_order_by": False, "level": None, "function": None,
            "anchor": anchors[0] if anchors else None, "anchors": anchors}


def answer_of(row: dict, gold_given: bool) -> list:
    """KORA's answer: the row's gold-free `projection`, else `answer_order` when no gold was given."""
    if "projection" in row:
        return row["projection"] or []
    if "answer_order" not in row:   # a run KORA could not finish (no anchor, an error): no answer, its error kept
        return []
    if gold_given:
        raise RuntimeError("[kora] the gold run's row has no gold-free `projection`")
    return row.get("answer_order") or []


def run_kora(sample, args: argparse.Namespace, stop: str):
    dataset = dataset_of(args, sample)
    harness = harness_for(stop, dataset, args)
    qm = QueryMetrics(question=sample.question)
    harness.llm.qm = qm
    oracle = getattr(args, "stop_oracle_callback", None) if stop == "gold" else None
    row = harness.run(item_of(sample, oracle))
    harness.llm.qm = None
    record_verdicts(stop, row)
    names = [str(n) for n in answer_of(row, gold_given=oracle is not None) if n]
    answer = "; ".join(names)
    summary = qm.summary()
    summary["kora"] = {k: row[k] for k in KEPT if k in row}
    summary["kora"]["task_graph"] = [{"id": t["id"], "depth": t["depth"], "status": t["status"],
                                      "attempts": len(t["attempts"]), "goal": t["goal"][:120]}
                                     for t in row.get("graph", [])]
    summary["kora_stop"] = stop
    summary["kora_settings"] = {**getattr(harness, "settings_in_use", {}), "seconds": SECONDS, "tokens": TOKENS}
    if row.get("error"):
        summary["error"] = row["error"]
    print(f"  KORA {stop} [{row.get('status')}/{row.get('stop_reason')}] {answer[:160]}")
    return names, summary   # a list: the harness scores its items


def run(question: str, args: argparse.Namespace, sample=None):
    """The harness's handler: KORA under the run's --stop. Dataset mode only."""
    if sample is None:
        raise NotImplementedError("[kora] runs in dataset mode only")
    return run_kora(sample=sample, args=args, stop=args.stop)


# KORA's stop decisions, read from its row after the run (README: KORA's verdicts).
_OFFER = re.compile(r"offer (?P<task>\S+): (?P<n>\d+) -> root, judged (?P<said>yes|no) -- (?P<reason>.*)", re.S)
_COMPOSED = re.compile(r"composed (?P<task>\S+): (?P<combine>\S+) -> (?P<n>\d+), judged (?P<said>yes|no) (?P<reason>.*)", re.S)
_SCORE = re.compile(r"estimator p\(sufficient\)=(?P<p>\d+(?:\.\d+)?)|levels score (?P<levels>\d+(?:\.\d+)?)")
LEAF_ACCEPT = {"SUCCESS": True, "unselected": True, "INSUFFICIENT_RESULT": False}
ROOT = "T0"


def judge_of(stop: str) -> tuple[str, str | None, float | None]:
    """The judge that decides where the gold does not, and the estimator server and cut KORA resolved."""
    from kora.app import STOP_MODES
    if STOP_MODES[stop].get("KORA_ROOT_JUDGE") == "knowledge":
        return "knowledge", None, None
    try:
        from kora.modules.sufficiency import estimator
        if estimator.enabled():
            from kora.app import estimator_tau
            url = estimator.url()
            explicit = estimator_tau()
            return "estimator", url, explicit if explicit is not None else getattr(estimator, "_SERVED", {}).get(url)
    except Exception:
        pass
    return "llm", None, None


def record_verdicts(stop: str, row: dict) -> None:
    """Every judgement in KORA's row, one line each in the question's verdicts file. Never raises."""
    try:
        from kora.app import STOP_MODES
        arm = f"kora_{stop}"
        oracle_root = STOP_MODES[stop].get("KORA_ROOT_JUDGE") == "oracle"
        judge, url, threshold = judge_of(stop)
        graph = [t for t in row.get("graph") or [] if isinstance(t, dict)]
        depth_of = {t.get("id"): t.get("depth") for t in graph}

        def write(target, candidate, accept, level, score=None, refused=None, error=None, depth=None, step=None):
            gold = oracle_root and target == ROOT
            fields = dict(level=level, judge="gold" if gold else judge, source="kora_row")
            if gold:
                verdict_log.record(arm, "gold", candidate, accept, refused=None, gold_hit=bool(accept), depth=depth,
                                   step=step, **fields)
            else:
                verdict_log.record(arm, "judge", candidate, accept, score=score, refused=refused, error=error,
                                   judge_url=url if judge == "estimator" else None,
                                   threshold=threshold if judge == "estimator" else None, depth=depth, step=step,
                                   **fields)

        for t in graph:
            for a in t.get("attempts") or []:
                for e in a.get("executions") or []:
                    outcome = str(e.get("outcome") or "")
                    if outcome not in LEAF_ACCEPT:
                        continue
                    candidate = {"task": t.get("id"), "goal": str(t.get("goal") or "")[:200], "attempt": a.get("n"),
                                 "candidate": e.get("candidate"), "query": e.get("query"), "rows": e.get("rows"),
                                 "outcome": outcome}
                    if outcome == "SUCCESS":   # the attempt's entities are this candidate's (the first 40)
                        candidate["ids"] = list(a.get("entities") or [])
                    write(t.get("id"), candidate, LEAF_ACCEPT[outcome], "leaf", depth=t.get("depth"), step=a.get("n"))
        step = 0
        for line in row.get("log") or []:
            m = _OFFER.fullmatch(str(line)) or _COMPOSED.fullmatch(str(line))
            if m is None:
                continue
            step += 1
            offer = m.re is _OFFER
            target = ROOT if offer else m["task"]
            reason = m["reason"].strip()
            found = _SCORE.search(reason)
            score = None if found is None else float(found["p"] or found["levels"])
            unavailable = "unavailable" in reason
            candidate = {"task": m["task"], "target": target, "n": int(m["n"]), "reason": reason,
                         **({} if offer else {"combine": m["combine"]})}
            write(target, candidate, m["said"] == "yes", "root" if offer else "composed", score=score,
                  refused=True if unavailable else None, error=reason if unavailable else None,
                  depth=depth_of.get(target), step=step)
    except Exception:
        pass
