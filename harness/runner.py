"""The run loop: every question of a set through one system, one row per question (README: the run folder)."""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from harness import graph_meter
from harness.dataset import QuestionSample, load
from harness.llm import LLMCallMetrics

# One class wherever it is imported from (harness/settings.py).
from harness.settings import QuestionTimeoutError  # noqa: E402,F401


@contextlib.contextmanager
def _question_timeout(seconds: int):
    """Unix alarm-based timeout for one question evaluation."""
    if not seconds or seconds <= 0:
        yield
        return

    previous_handler = signal.getsignal(signal.SIGALRM)

    def _handle_timeout(signum, frame):
        raise QuestionTimeoutError(f"question_timeout_after_{seconds}s")

    signal.signal(signal.SIGALRM, _handle_timeout)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)


def _zero_eval() -> dict:
    """All-zero eval for a timed-out or failed question, so it counts in the denominator."""
    return {
        "mrr":          0.0, "hit@1":        0.0, "hit@3":        0.0,
        "hit@5":        0.0, "hit@10":       0.0, "hit@20":       0.0,
        "recall@5":     0.0, "recall@10":    0.0, "recall@20":    0.0,
        "recall@all":   0.0, "precision@5":  0.0, "precision@10": 0.0,
        "precision@20": 0.0, "exact_match":  0.0, "f1":           0.0,
    }


def _dispatch(
    question: str,
    framework: str,
    args: argparse.Namespace,
    answer_ids: Optional[List[str]] = None,
    sample=None,            # full QuestionSample — forwarded to KG-aware handlers
) -> Tuple[str, dict, Optional[Dict[str, float]], float]:
    """One question through a framework: (answer, fw_metrics, eval_result, wall_time)."""
    from harness.systems import handler_of
    handler = handler_of(framework)

    # MetaQA: a node's id is its name, so no entity resolution.
    if sample is not None and getattr(sample, "kg", "") == "metaqa":
        if getattr(args, "kg_name", None) != "metaqa":
            args.kg_name = "metaqa"

    # The framework gets the sample with its gold locked; a gold-stop arm gets the gold only as its stop callback.
    from harness.dataset import blind, is_gold_arm
    from harness.stop_oracle import StopOracle
    args.stop_oracle_callback = (
        StopOracle(sample.answers, question=sample.question or "", question_type=getattr(sample, "question_type", None))
        if is_gold_arm(framework, getattr(args, "stop", "")) and sample is not None and sample.answers else None)
    t0 = time.perf_counter()
    try:
        answer, fw_metrics = handler(question, args, sample=blind(sample))
    finally:
        args.stop_oracle_callback = None
    wall_time = time.perf_counter() - t0

    # A list is scored as its items; the row's text joins them with "; " (", " would split "Washington, D.C.").
    listed = None
    if isinstance(answer, (list, tuple)):
        listed = [str(a).strip() for a in answer if str(a).strip()]
        answer = "; ".join(listed)
        if isinstance(fw_metrics, dict):
            fw_metrics["answer_items"] = listed

    # Thinking blocks are stripped so hidden reasoning cannot make a false match.
    import re as _re
    answer = _re.sub(r"<think>.*?</think>", "", str(answer or ""), flags=_re.DOTALL).strip()

    # The row's own text-match scores (its `eval` field); the reported numbers come from harness/metrics.
    eval_result: Optional[Dict[str, float]] = None
    # Gold once each: text labels when the row has them, ids only when not (a model answers in names).
    labels = list(answer_ids or []) + list((sample.answers or []) if sample is not None else [])
    ids = list((sample.answer_ids or []) if sample is not None else [])
    answer_targets: List[str] = list(dict.fromkeys(
        str(a) for a in (labels or ids) if a is not None and str(a).strip()))

    if answer_targets:
        hit, em, prec, rec, f1 = _score_text_answer(answer, answer_targets, listed)
        eval_result = {
            "mrr": hit,
            "hit@1": hit,
            "hit@3": hit,
            "hit@5": hit,
            "hit@10": hit,
            "hit@20": hit,
            "recall@5": hit,
            "recall@10": hit,
            "recall@20": hit,
            "recall@all": hit,
            "precision@5": hit / 5 if hit else 0.0,
            "precision@10": hit / 10 if hit else 0.0,
            "precision@20": hit / 20 if hit else 0.0,
            "exact_match": em,
            "precision": prec,
            "recall": rec,
            "f1": f1,
        }

    return answer, fw_metrics, eval_result, wall_time


# Re-exported: the answer-text scorer lives in harness/metrics/reasoning.py.
from harness.metrics.reasoning import score_text as _score_text_answer  # noqa: E402,F401


def _load_ids(path: str) -> set:
    """Ids from a JSON list, {"ids": [...]} or {"questions": [{"id"}]} file."""
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    if isinstance(d, dict):
        d = d.get("ids") or d.get("questions") or []
    return {str(x["id"]) if isinstance(x, dict) else str(x) for x in d}


def run_dataset(args: argparse.Namespace) -> int:
    """Every (or --max-samples) question of a set through the chosen framework, one JSONL row each."""
    from harness.dataset import SPLIT_SUFFIX, graph_of
    benchmark = args.dataset
    kg = graph_of(benchmark)
    args.dataset_benchmark = f"{benchmark}-{SPLIT_SUFFIX[args.split]}"

    samples: List[QuestionSample] = load(benchmark, args.split, max_samples=getattr(args, "max_samples", None))
    if not samples:
        print(f"[harness] no questions in {benchmark} {args.split}.")
        return 1

    original_sample_count = len(samples)
    num_shards = int(getattr(args, "num_shards", 1) or 1)
    shard_index = int(getattr(args, "shard_index", 0) or 0)
    if num_shards < 1:
        print("[harness] --num-shards must be >= 1.")
        return 2
    if shard_index < 0 or shard_index >= num_shards:
        print(f"[harness] --shard-index must be in [0, {num_shards - 1}].")
        return 2
    if num_shards > 1:
        samples = [s for i, s in enumerate(samples) if i % num_shards == shard_index]
        print(f"[harness] shard {shard_index}/{num_shards}: {len(samples)} of {original_sample_count} questions.")
        if not samples:
            print("[harness] shard is empty; exiting cleanly.")
            return 0

    # --ids filters after sharding, so a question keeps the shard and output file it has in the full set.
    only_file = getattr(args, "ids", None)
    if only_file:
        keep = _load_ids(only_file)
        before = len(samples)
        samples = [s for s in samples if str(s.qid) in keep]
        print(f"[harness] subset {os.path.basename(only_file)}: {len(samples)} of {before} questions in this shard.")
        if not samples:
            print("[harness] no listed question in this shard; exiting cleanly.")
            return 0

    if getattr(args, "api_base", ""):
        os.environ["OPENAI_BASE_URL"] = args.api_base.rstrip("/")
        os.environ["OPENAI_API_BASE"] = args.api_base.rstrip("/")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HARNESS_TRACE_DIR"] = str(out_dir / "traces" / f"s{shard_index}")
    (out_dir / "run.json").write_text(json.dumps(
        {"system": args.system, "stop": args.stop, "framework": args.framework, "dataset": benchmark,
         "split": args.split, "model": args.engine, "timeout_seconds": args.timeout}, indent=1) + "\n")
    print(f"[harness] {len(samples)} questions | {args.system} (stop: {args.stop}) | model {args.engine}")

    out_path = out_dir / f"s{shard_index}.jsonl"
    done_qids: set = set()
    if getattr(args, "resume", False) and out_path.exists():
        # --resume: the questions already done are skipped.
        with open(out_path, encoding="utf-8") as _rf:
            for _line in _rf:
                _line = _line.strip()
                if not _line:
                    continue
                try:
                    done_qids.add(json.loads(_line)["qid"])
                except Exception:
                    pass
        print(f"[harness] resuming: {len(done_qids)} questions already done are skipped.")
        out_file = open(out_path, "a", encoding="utf-8")
    else:
        out_file = open(out_path, "w", encoding="utf-8")

    n_errors = 0
    hits: List[float] = []
    run_start = time.perf_counter()
    try:
        for idx, sample in enumerate(samples):
            if sample.qid in done_qids:
                continue
            import harness.llm as _llm_mod
            _llm_mod.TRACE_QID = str(sample.qid)   # names this question's trace file
            graph_meter.install()
            graph_meter.reset()                     # this question's graph queries start at zero
            print(f"[harness] {idx + 1}/{len(samples)} {sample.qid}", flush=True)

            question_started = time.perf_counter()
            from harness.settings import untimed
            untimed["seconds"] = 0.0          # time this question spends with the alarm paused is not its own
            question_timeout_seconds = int(getattr(args, "timeout", 0) or 0)
            try:
                with _question_timeout(question_timeout_seconds):
                    answer, fw_metrics, eval_result, wall_time = _dispatch(
                        question   = sample.question,
                        framework  = args.framework,
                        args       = args,
                        answer_ids = list(sample.answers),
                        sample     = sample,
                    )
                # The question's time runs from when its alarm is armed, as the time limit does.
                wall_time = time.perf_counter() - question_started - untimed["seconds"]
                # A framework reporting an error and no answer failed the question (KORA, SoG: fw_metrics["error"]).
                error = ((fw_metrics.get("error") or None) if isinstance(fw_metrics, dict) and not answer else None)
            except NotImplementedError as e:
                print(f"\n[harness] not implemented: {e}")
                return 1
            except QuestionTimeoutError as e:
                wall_time = time.perf_counter() - question_started - untimed["seconds"]
                answer = ""
                fw_metrics = {
                    "error": str(e),
                    "timeout_seconds": question_timeout_seconds,
                    "success": False,
                }
                eval_result = _zero_eval()
                error = str(e)
                n_errors += 1
                print(f"[harness] question timed out after {question_timeout_seconds}s: {sample.qid}")
            except Exception as e:
                answer      = ""
                wall_time   = time.perf_counter() - question_started
                fw_metrics  = {
                    "error": str(e) or type(e).__name__,
                    "success": False,
                    "wall_time_s": round(wall_time, 3),
                }
                eval_result = _zero_eval()
                error       = str(e) or type(e).__name__   # never empty: an empty error would read as no error
                n_errors   += 1

            if eval_result:
                hits.append(float(eval_result.get("hit@1", 0.0)))

            # Graph queries beside the LLM calls and tokens, whatever client the framework used.
            if isinstance(fw_metrics, dict):
                fw_metrics.update(graph_meter.snapshot())
                # Every row carries the same LLM fields: a framework that makes no model call records zeros.
                for key in ("num_llm_calls", "total_prompt_tokens", "total_completion_tokens", "total_tokens"):
                    fw_metrics.setdefault(key, 0)

            record: dict = {
                "qid":         sample.qid,
                "question":    sample.question,
                "answers":     sample.answers,
                "answer_ids":  sample.answer_ids,
                "framework":   args.framework,
                "system":      args.system,
                "stop":        args.stop,
                "engine":      args.engine,
                "answer":      answer,
                "fw_metrics":  fw_metrics,
                "eval":        eval_result,
                "wall_time_s": round(wall_time, 3),
                "benchmark":   benchmark,
                "split":       args.split,
                "kg":          kg,
                "error":       error,
            }
            if num_shards > 1:
                record["num_shards"] = num_shards
                record["shard_index"] = shard_index
            out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_file.flush()
    finally:
        out_file.close()

    elapsed = time.perf_counter() - run_start
    done = len(hits)
    print(f"[harness] done: {done} questions in {elapsed:.1f} s, {n_errors} errors; "
          f"text-match hit {sum(hits) / done if done else 0.0:.3f} (the reported numbers come from `score`)")
    print(f"[harness] rows in {out_path}")
    return 0


# Per-question totals of a system's LLM calls: calls, tokens, latency, by call site.
@dataclass
class QueryMetrics:
    """Accumulates LLMCallMetrics objects for a single question."""

    question: str = ""
    _calls: List[dict] = field(default_factory=list)
    _start: float = field(default_factory=time.perf_counter)

    def add(self, metrics: LLMCallMetrics, call_site: str = "") -> None:
        """Record one LLM call.  call_site is a short label like 'relation_prune'."""
        self._calls.append({
            "call_site":         call_site,
            "engine":            metrics.engine,
            "provider":          metrics.provider,
            "latency_s":         metrics.latency_seconds,
            "prompt_tokens":     metrics.prompt_tokens,
            "completion_tokens": metrics.completion_tokens,
            "total_tokens":      metrics.total_tokens,
            "temperature":       metrics.temperature,
            "success":           metrics.success,
            "error":             metrics.error,
        })

    @property
    def num_calls(self) -> int:
        return len(self._calls)

    @property
    def num_failed(self) -> int:
        return sum(1 for c in self._calls if not c["success"])

    @property
    def total_prompt_tokens(self) -> int:
        return sum(c["prompt_tokens"] for c in self._calls)

    @property
    def total_completion_tokens(self) -> int:
        return sum(c["completion_tokens"] for c in self._calls)

    @property
    def total_tokens(self) -> int:
        return self.total_prompt_tokens + self.total_completion_tokens

    @property
    def total_llm_latency(self) -> float:
        """Sum of all individual LLM call latencies (seconds)."""
        return sum(c["latency_s"] for c in self._calls)

    @property
    def wall_time(self) -> float:
        """Wall-clock seconds since this QueryMetrics was created."""
        return time.perf_counter() - self._start

    @property
    def tokens_per_second(self) -> float:
        t = self.total_llm_latency
        return round(self.total_completion_tokens / t, 1) if t > 0 else 0.0

    def summary(self) -> dict:
        """Return a JSON-serialisable dict of all aggregated metrics."""
        engines_used = list({c["engine"] for c in self._calls})
        by_site: dict[str, dict] = {}
        for c in self._calls:
            site = c["call_site"] or "unknown"
            s = by_site.setdefault(site, {"calls": 0, "prompt_tokens": 0,
                                          "completion_tokens": 0, "latency_s": 0.0})
            s["calls"]             += 1
            s["prompt_tokens"]     += c["prompt_tokens"]
            s["completion_tokens"] += c["completion_tokens"]
            s["latency_s"]         = round(s["latency_s"] + c["latency_s"], 3)

        return {
            "question":                self.question,
            "num_llm_calls":           self.num_calls,
            "num_failed_calls":        self.num_failed,
            "total_prompt_tokens":     self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens":            self.total_tokens,
            "total_llm_latency_s":     round(self.total_llm_latency, 3),
            "wall_time_s":             round(self.wall_time, 3),
            "tokens_per_second":       self.tokens_per_second,
            "engines_used":            engines_used,
            "breakdown_by_call_site":  by_site,
            "individual_calls":        self._calls,
        }

    def report(self) -> None:
        """Pretty-print a summary to stdout."""
        s = self.summary()
        print("\n── Query Metrics ────────────────────────────────────────")
        print(f"  Question            : {s['question'][:80]}")
        print(f"  LLM calls           : {s['num_llm_calls']}  (failed: {s['num_failed_calls']})")
        print(f"  Prompt tokens       : {s['total_prompt_tokens']}")
        print(f"  Completion tokens   : {s['total_completion_tokens']}")
        print(f"  Total tokens        : {s['total_tokens']}")
        print(f"  Total LLM latency   : {s['total_llm_latency_s']} s")
        print(f"  Wall time           : {s['wall_time_s']} s")
        print(f"  Tokens / second     : {s['tokens_per_second']}")
        print(f"  Engines used        : {', '.join(s['engines_used'])}")
        print("  ── By call site ──────────────────────────────────────")
        for site, v in s["breakdown_by_call_site"].items():
            print(f"    {site:<25} calls={v['calls']}  "
                  f"tok={v['prompt_tokens']}+{v['completion_tokens']}  "
                  f"lat={v['latency_s']}s")
        print("─────────────────────────────────────────────────────────\n")
