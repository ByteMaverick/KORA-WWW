"""
baselines/official_sampling.py
-------------------------------
The decoding an authors' own OpenAI client sends: harness.settings.decoding(framework, engine) as keyword
arguments for `client.chat.completions.create`, for the systems whose authors' code calls the server itself (SoG);
and `sampling`, the row's top_p and top_k for the adapters that call harness.llm.run_llm or
baselines.official_llm.chat (ToG, PoG, CoR), whose own calls carry only a temperature and a cap.
Every row states both: a value a request leaves out is filled by the server, not by the authors.

  temperature, top_p, presence_penalty, max_tokens   OpenAI parameters, sent as they are
  top_k                                              not an OpenAI parameter: sent in extra_body, which vLLM reads

A value the table leaves None is not sent (the table states the server's default for it), and one the caller's
arguments carry is replaced by the table's, so what the server receives is the table, whatever the authors' code
asked for. The caller's other arguments (messages, tools, extra_body's chat_template_kwargs) are kept.
"""
from __future__ import annotations

from harness.settings import decoding

TOP_LEVEL = ("temperature", "top_p", "presence_penalty", "max_tokens")
EXTRA_BODY = ("top_k",)


def sampling(framework: str, engine: str) -> dict:
    """The row's top_p and top_k, as keyword arguments for run_llm / official_llm.chat."""
    table = decoding(framework, engine)
    return {"top_p": table["top_p"], "top_k": table["top_k"]}


def openai_kwargs(framework: str, engine: str, kw: dict | None = None) -> dict:
    """`kw` (the authors' call arguments, or none) with the table's decoding for this system on this engine."""
    table = decoding(framework, engine)
    out = dict(kw or {})
    extra = dict(out.get("extra_body") or {})
    for key in TOP_LEVEL:
        out.pop(key, None)
        if table.get(key) is not None:
            out[key] = table[key]
    for key in EXTRA_BODY:
        extra.pop(key, None)
        if table.get(key) is not None:
            extra[key] = table[key]
    out.pop("extra_body", None)
    if extra:
        out["extra_body"] = extra
    return out
