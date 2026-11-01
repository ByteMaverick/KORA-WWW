"""The model client every system shares: an OpenAI-compatible call with retries, token counts and a trace."""
import os
import time
import tiktoken
from dataclasses import dataclass, field
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage

# API key variable per provider
_PROVIDER_ENV_VARS = {
    "openai":     "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    # local servers need no key — handled separately
}


def _resolve_api_key(provider: str, api_key: str) -> str:
    """The API key: the argument, else the provider's environment variable; local servers need none."""
    if provider == "local":
        return api_key or "EMPTY"

    if api_key and api_key.strip():
        return api_key.strip()

    env_var = _PROVIDER_ENV_VARS.get(provider)
    if env_var:
        value = os.environ.get(env_var, "").strip()
        if value:
            return value

    raise EnvironmentError(
        f"\n[run_llm] ✗ No API key found for provider '{provider}'.\n"
    )


@dataclass
class LLMCallMetrics:
    engine: str
    provider: str
    latency_seconds: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    temperature: float
    max_tokens: int
    success: bool
    error: Optional[str] = None
    extras: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {
            "engine":            self.engine,
            "provider":          self.provider,
            "latency_s":         round(self.latency_seconds, 3),
            "prompt_tokens":     self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens":      self.total_tokens,
            "tokens_per_second": round(self.completion_tokens / self.latency_seconds, 1)
                                 if self.latency_seconds > 0 else 0,
            "success":           self.success,
            "error":             self.error,
            **self.extras,
        }


def _count_tokens_tiktoken(text: str) -> int:
    """Fallback token counter when provider doesn't return usage metadata."""
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return len(text.split())


def _normalise_engine_name(engine: str) -> str:
    """Strip local-routing prefixes while preserving the provider model name."""
    e = engine.strip()
    for prefix in ("ollama:", "local:"):
        if e.lower().startswith(prefix):
            return e[len(prefix):].strip()
    return e


def _looks_like_local_api(api_base: str) -> bool:
    """Return True for Ollama/vLLM/LM Studio style OpenAI-compatible URLs."""
    base = (
        api_base
        or os.getenv("OPENAI_BASE_URL", "")
        or os.getenv("OPENAI_API_BASE", "")
        or os.getenv("OLLAMA_API_BASE", "")
    ).lower()
    return any(marker in base for marker in (
        "localhost",
        "127.0.0.1",
        "0.0.0.0",
        ":11434",
        ":8000",
    ))


def _detect_provider(engine: str, api_base: str = "") -> str:
    e = engine.lower()
    if e.startswith(("ollama:", "local:")) or _looks_like_local_api(api_base):
        return "local"
    if any(x in e for x in ["gpt-", "o1", "o3"]):
        return "openai"
    elif any(x in e for x in ["llama", "mistral", "mixtral",
                               "qwen", "deepseek", "phi", "gemma",
                               "yi", "olmo", "command-r", "codellama"]):
        return "local"
    else:
        # openrouter or any other openai-compatible endpoint
        return "openrouter"


def _sends_top_k(base_url: str) -> bool:
    """top_k goes in the request body (extra_body) to OpenAI-compatible servers, never to api.openai.com."""
    return "api.openai.com" not in (base_url or "").lower()


def _build_llm(
    engine: str,
    temperature: float,
    max_tokens: int,
    api_key: str,
    api_base: str,
    top_p: float | None = 1.0,
    top_k: int | None = -1,
):
    """(llm, provider): a LangChain chat model for the engine, its API key resolved from the environment."""
    provider     = _detect_provider(engine, api_base)
    model_name   = _normalise_engine_name(engine)
    resolved_key = _resolve_api_key(provider, api_key)  # raises if missing

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        # o1 / o3 / gpt-5 take max_completion_tokens; older langchain-openai knows only max_tokens.
        _uses_completion_tokens = any(
            engine.lower().startswith(p)
            for p in ("o1", "o3", "gpt-5", "o4")
        )
        try:
            if _uses_completion_tokens:
                llm = ChatOpenAI(
                    model=engine,
                    openai_api_key=resolved_key,
                    max_completion_tokens=max_tokens,
                )
            else:
                llm = ChatOpenAI(
                    model=engine,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    openai_api_key=resolved_key,
                )
        except TypeError:
            # Older langchain-openai that doesn't know max_completion_tokens yet
            llm = ChatOpenAI(
                model=engine,
                openai_api_key=resolved_key,
                model_kwargs={"max_completion_tokens": max_tokens},
            )

    elif provider == "local":
        # Local server (vLLM / Ollama / LM Studio) — openai-compatible endpoint
        from langchain_openai import ChatOpenAI
        local_api_base = (
            api_base
            or os.getenv("OPENAI_BASE_URL")
            or os.getenv("OPENAI_API_BASE")
            or os.getenv("OLLAMA_API_BASE")
            or "http://localhost:11434/v1"
        )
        # The settings table's effort, so every system on a run thinks equally hard.
        from harness.settings import reasoning_effort
        effort = reasoning_effort(engine)
        # Sampling always sent: a vLLM server fills a missing value from the model's generation_config.
        llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            openai_api_key=resolved_key,   # "EMPTY" if no key provided
            openai_api_base=local_api_base,
            **({"top_p": top_p} if top_p is not None else {}),
            **({"extra_body": {"top_k": top_k}} if top_k is not None and _sends_top_k(local_api_base) else {}),
            **({"model_kwargs": {"reasoning_effort": effort}} if effort else {}),
        )

    else:
        # OpenRouter or any other openai-compatible external endpoint
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model=engine,
            temperature=temperature,
            max_tokens=max_tokens,
            openai_api_key=resolved_key,
            openai_api_base=api_base or "https://openrouter.ai/api/v1",
        )

    return llm, provider


def _extract_text_content(content) -> str:
    """response.content as a plain string, from a str, a list of text blocks or a list of strings."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                # Standard content block: {"type": "text", "text": "..."}
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                # Fallback: any dict with a "text" key
                elif "text" in block:
                    parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    # Last resort
    return str(content)


def _extract_text_from_response(response) -> str:
    """The reply text, also looked for in additional_kwargs and response_metadata when .content is empty."""
    text = _extract_text_content(response.content)
    if text.strip():
        return text

    ak = getattr(response, "additional_kwargs", {}) or {}

    if "content" in ak and ak["content"]:
        return str(ak["content"])

    meta = getattr(response, "response_metadata", {}) or {}
    choices = meta.get("choices", [])
    if choices:
        msg = choices[0].get("message", {})
        if msg.get("content"):
            return str(msg["content"])

    return text  # return empty string; caller will surface the issue


def _extract_token_counts(
    response,
    prompt: str,
    result: str,
    provider: str,
) -> tuple[int, int, dict]:
    """Prompt and completion tokens and extras from the response metadata; tiktoken estimates when absent."""
    extras = {}
    metadata = getattr(response, "response_metadata", {}) or {}
    usage_metadata = getattr(response, "usage_metadata", {}) or {}
    usage = (
        metadata.get("usage")                  # openai / openrouter
        or metadata.get("token_usage")         # some langchain versions
        or metadata.get("usageMetadata")       # google
        or usage_metadata                      # LangChain AIMessage standard
        or {}
    )

    if provider in ("openai", "local", "openrouter"):
        prompt_tokens = (
            usage.get("prompt_tokens", 0)
            or usage.get("input_tokens", 0)
            or usage.get("promptTokenCount", 0)
        )
        completion_tokens = (
            usage.get("completion_tokens", 0)
            or usage.get("output_tokens", 0)
            or usage.get("candidatesTokenCount", 0)
        )
        extras["finish_reason"] = metadata.get("finish_reason")

    else:
        prompt_tokens = completion_tokens = 0

    # Fall back to tiktoken if counts are missing
    if prompt_tokens == 0:
        prompt_tokens = _count_tokens_tiktoken(prompt)
    if completion_tokens == 0:
        completion_tokens = _count_tokens_tiktoken(result)

    return prompt_tokens, completion_tokens, extras


def run_llm(
    prompt: str,
    engine: str         = "gpt-3.5-turbo",
    temperature: float  = 0.0,
    max_tokens: int     = 512,
    api_key: str        = "",
    api_base: str       = "",
    system_prompt: str  = "You are an AI assistant that helps people find information.",
    max_retries: int    = 3,
    retry_delay: float  = 2.0,
    stop: list[str] | str | None = None,
    n: int              = 1,
    top_p: float | None = 1.0,
    top_k: int | None   = -1,
) -> tuple[str, LLMCallMetrics] | tuple[list[str], LLMCallMetrics]:
    """One LLM call with retries: (the text, or n texts when n > 1; its metrics) (README: the model client)."""
    llm, provider = _build_llm(engine, temperature, max_tokens, api_key, api_base, top_p=top_p, top_k=top_k)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompt),
    ]

    # Aggregated across all completions when n > 1
    all_results:       list[str] = []
    total_prompt_tok:  int       = 0
    total_compl_tok:   int       = 0
    extras:            dict      = {}
    error:             str|None  = None

    start = time.perf_counter()
    from harness.settings import reasoning_effort
    _effort = reasoning_effort(engine)

    # For OpenAI we bypass LangChain and use the raw SDK directly.
    _openai_client = None
    _is_reasoning_model = any(engine.lower().startswith(p)
                              for p in ("o1", "o3", "gpt-5", "o4"))
    if provider == "openai":
        import openai as _oai
        _openai_client = _oai.OpenAI(api_key=_resolve_api_key("openai", api_key))
        if _is_reasoning_model and max_tokens < 4000:
            print(
                f"[run_llm] ℹ '{engine}' is a reasoning model — "
                f"raising max_completion_tokens from {max_tokens} → 4000 "
                f"(reasoning tokens consume budget before content is written). "
                f"Pass --max-tokens 4000 or higher to suppress this message."
            )
            max_tokens = 4000

    for attempt in range(1, max_retries + 1):
        try:
            if provider == "openai":
                # Raw OpenAI SDK: supports n natively
                _uses_ct = any(engine.lower().startswith(p)
                               for p in ("o1", "o3", "gpt-5", "o4"))
                _tok_kw = (
                    {"max_completion_tokens": max_tokens}
                    if _uses_ct else
                    {"max_tokens": max_tokens, "temperature": temperature}
                )
                if stop:
                    _stop_list = [stop] if isinstance(stop, str) else list(stop)
                    _stop_list = [s for s in _stop_list if s and s.strip()]
                    if _stop_list:
                        _tok_kw["stop"] = _stop_list
                # Reasoning models don't support n > 1
                _n = 1 if _is_reasoning_model else n
                if _n > 1:
                    _tok_kw["n"] = _n

                # The effort here too: a model server at a hostname this module does not recognise lands here.
                if _effort:
                    _tok_kw["reasoning_effort"] = _effort
                # Sampling here too (a vLLM server fills a missing value); OpenAI's reasoning models take none.
                if not _uses_ct:
                    if top_p is not None:
                        _tok_kw["top_p"] = top_p
                    if top_k is not None and _sends_top_k(str(_openai_client.base_url)):
                        _tok_kw["extra_body"] = {"top_k": top_k}
                _oai_resp = _openai_client.chat.completions.create(
                    model=engine,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": prompt},
                    ],
                    **_tok_kw,
                )
                all_results = [
                    (c.message.content or "") for c in _oai_resp.choices
                ]
                _usage = _oai_resp.usage
                total_prompt_tok  = _usage.prompt_tokens     if _usage else 0
                total_compl_tok   = _usage.completion_tokens if _usage else 0
                _fr = _oai_resp.choices[0].finish_reason or ""
                extras = {"finish_reason": _fr}

            else:
                # Local servers and OpenRouter have no native n > 1: n sequential calls, aggregated.
                if stop:
                    stop_list     = [stop] if isinstance(stop, str) else list(stop)
                    stop_list     = [s for s in stop_list if s and s.strip()]
                    invoke_kwargs = {"stop": stop_list} if stop_list else {}
                else:
                    invoke_kwargs = {}

                for _ in range(n):
                    response = llm.invoke(messages, **invoke_kwargs)
                    text     = _extract_text_from_response(response)
                    all_results.append(text)
                    pt, ct, ex = _extract_token_counts(response, prompt, text, provider)
                    total_prompt_tok += pt
                    total_compl_tok  += ct
                    extras = ex   # keep last call's extras (finish_reason etc.)

            # Warn if any completion hit the token limit
            finish = extras.get("finish_reason") or extras.get("stop_reason", "")
            if finish == "length":
                if _is_reasoning_model:
                    print(
                        f"[run_llm] ⚠ finish_reason=length — reasoning model '{engine}' "
                        f"hit the token limit (max_completion_tokens={max_tokens}). "
                        f"Try --max-tokens 8000 or higher."
                    )
                else:
                    print(
                        f"[run_llm] ⚠ finish_reason=length — response was truncated "
                        f"(max_tokens={max_tokens}). Consider increasing --max-tokens."
                    )

            if _effort:
                extras["reasoning_effort"] = _effort   # what was sent, on the row
            error = None
            break  # success

        except Exception as e:
            error = str(e)
            print(f"[run_llm] attempt {attempt}/{max_retries} failed — {e}")
            if attempt < max_retries:
                time.sleep(retry_delay)

    latency = time.perf_counter() - start

    metrics = LLMCallMetrics(
        engine            = engine,
        provider          = provider,
        latency_seconds   = latency,
        prompt_tokens     = total_prompt_tok,
        completion_tokens = total_compl_tok,
        total_tokens      = total_prompt_tok + total_compl_tok,
        temperature       = temperature,
        max_tokens        = max_tokens,
        success           = error is None,
        error             = error,
        extras            = extras,
    )

    # n=1 returns one string; when every retry failed, metrics.success and metrics.error say so.
    result = all_results[0] if (n == 1 and all_results) else ("" if n == 1 else all_results)
    _trace_call(engine, api_base, system_prompt, prompt, all_results, metrics)
    return result, metrics


# Call traces (README: the LLM call trace); harness.runner sets TRACE_QID before each question.
TRACE_QID: str = ""


def _trace_call(engine, api_base, system_prompt, prompt, replies, metrics) -> None:
    trace_dir = os.getenv("HARNESS_TRACE_DIR", "")
    if not trace_dir:
        return
    try:
        import json as _json
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in (TRACE_QID or "no-qid"))
        os.makedirs(trace_dir, exist_ok=True)
        with open(os.path.join(trace_dir, f"{safe}.jsonl"), "a", encoding="utf-8") as f:
            f.write(_json.dumps({
                "ts": time.time(), "qid": TRACE_QID, "engine": engine,
                "endpoint": api_base or os.getenv("OPENAI_BASE_URL", ""),
                "system": system_prompt, "prompt": prompt, "replies": replies,
                "prompt_tokens": metrics.prompt_tokens,
                "completion_tokens": metrics.completion_tokens,
                "latency_s": round(metrics.latency_seconds, 3),
                "success": metrics.success, "error": metrics.error, "extras": metrics.extras,
            }, ensure_ascii=False) + "\n")
    except Exception:
        pass


def trace_chat(engine, endpoint, messages, completion, metrics) -> None:
    """Traces a chat call a baseline makes with its own OpenAI client (SoG) as run_llm's are; never fails."""
    try:
        import json as _json

        def text(m) -> str:
            m = m if isinstance(m, dict) else (m.model_dump() if hasattr(m, "model_dump") else {"content": str(m)})
            body = m.get("content") or ""
            if not isinstance(body, str):
                body = _json.dumps(body, ensure_ascii=False)
            for c in m.get("tool_calls") or []:
                f = c.get("function") or {}
                body += f"\n[tool call] {f.get('name')}({f.get('arguments')})"
            return body

        role = lambda m: m.get("role") if isinstance(m, dict) else getattr(m, "role", "message")
        system = "\n\n".join(text(m) for m in messages or [] if role(m) == "system")
        prompt = "\n\n".join(f"[{role(m)}]\n{text(m)}" for m in messages or [] if role(m) != "system")
        reply = ""
        if completion is not None:
            msg = completion.choices[0].message
            reply = text({"content": msg.content, "tool_calls": [c.model_dump() for c in (msg.tool_calls or [])]})
        _trace_call(engine, str(endpoint or ""), system, prompt, [reply], metrics)
    except Exception:
        pass
