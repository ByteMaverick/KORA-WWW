"""
baselines/official_llm.py
--------------------------
One chat call for the authors'-code baselines whose own call is a single prompt (ToG, PoG): the harness's
harness.llm.run_llm on the harness's model servers, exactly as every other baseline calls them. Returns
(text, LLMCallMetrics); `stop` is passed through as the OpenAI API's stop sequences. `complete`: one text completion,
for KB-BINDER's Codex call.
"""
from __future__ import annotations

import os
import time

from harness.llm import LLMCallMetrics, run_llm


def chat(prompt: str, system: str, engine: str, temperature: float, max_tokens: int,
         stop: list[str] | None = None, top_p: float | None = 1.0, top_k: int | None = -1
         ) -> tuple[str, LLMCallMetrics]:
    """top_p and top_k: the system's row of the settings table (baselines/official_sampling.sampling); the defaults
    are the OpenAI API's."""
    text, metrics = run_llm(prompt=prompt, engine=engine, temperature=temperature, max_tokens=max_tokens,
                            system_prompt=system, stop=stop, top_p=top_p, top_k=top_k)
    return (text or ""), metrics


def complete(prompt: str, engine: str, temperature: float, max_tokens: int, n: int = 1,
             stop: list[str] | None = None, top_p: float | None = 1.0, top_k: int | None = -1
             ) -> tuple[list[str], LLMCallMetrics]:
    """A text completion on the model server (/v1/completions), n samples in one request: KB-BINDER's Codex call."""
    from openai import OpenAI
    from harness.llm import _trace_call
    base = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE") or "http://localhost:8000/v1"
    started, texts, usage, error = time.time(), [""] * n, None, None
    try:
        r = OpenAI(base_url=base, api_key=os.getenv("OPENAI_API_KEY") or "EMPTY", max_retries=2).completions.create(
            model=engine, prompt=prompt, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop or None,
            **({"top_p": top_p} if top_p is not None else {}),
            **({"extra_body": {"top_k": top_k}} if top_k is not None else {}))
        texts, usage = [c.text or "" for c in r.choices], r.usage
    except Exception as e:
        error = str(e)[:300]
    p, c = (getattr(usage, "prompt_tokens", 0) or 0), (getattr(usage, "completion_tokens", 0) or 0)
    metrics = LLMCallMetrics(engine=engine, provider="local", latency_seconds=time.time() - started,
                             prompt_tokens=p, completion_tokens=c, total_tokens=p + c, temperature=temperature,
                             max_tokens=max_tokens, success=error is None, error=error,
                             extras={"endpoint": "completions", "n": n})
    _trace_call(engine, base, "", prompt, texts, metrics)
    return texts, metrics
