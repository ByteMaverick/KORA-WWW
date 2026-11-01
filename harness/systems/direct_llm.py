"""Direct LLM: no graph, the model alone, with one answer-format line."""
from __future__ import annotations

import argparse
from typing import Tuple

from harness.llm import run_llm
from harness.settings import decoding

#: Direct LLM's one answer-format line: without it, prose answers name golds in passing and inflate the
#: mention-based Hit of the no-graph floor.
_DIRECT_FORMAT = "Answer with the name or names only, no explanation; separate several answers with commas."


def run_direct_llm(question: str, args: argparse.Namespace, sample=None) -> Tuple[str, dict]:
    """Framework: direct_llm — no retrieval, pure LLM: the question and one answer-format line."""
    from harness.runner import QueryMetrics

    qm = QueryMetrics(question=question)
    d = decoding("direct_llm", args.engine)   # the settings table
    answer, metrics = run_llm(
        prompt      = f"{question}\n{_DIRECT_FORMAT}",
        engine      = args.engine,
        temperature = d["temperature"],
        max_tokens  = d["max_tokens"],
        api_base    = getattr(args, "api_base", ""),
    )
    qm.add(metrics, call_site="direct_llm")
    return answer, qm.summary()
