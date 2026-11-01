"""The Kora-facing language-model capability, and a client for OpenAI-compatible servers."""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel, ConfigDict

T = TypeVar("T")


class LLMError(RuntimeError):
    """Raised when a provider cannot produce a usable response."""


class LLMResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_seconds: float = 0.0
    model: str = "unknown"

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@runtime_checkable
class LLMProvider(Protocol):
    """Contract every language-model provider satisfies."""

    def generate(self, prompt: str, *, system: str | None = None) -> LLMResponse: ...


_JSON_BLOCK = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_THOUGHT = re.compile(r"<(thought|think)>.*?</\1>", re.DOTALL)
_TRAILING_COMMA = re.compile(r",\s*([}\]])")


def extract_json(text: str) -> Any:
    """Pull a JSON value out of a model response."""
    candidate = _THOUGHT.sub("", text).strip()
    fenced = _JSON_BLOCK.search(candidate)
    if fenced:
        candidate = fenced.group(1).strip()
    else:
        starts = [i for i in (candidate.find("{"), candidate.find("[")) if i != -1]
        if starts:
            candidate = candidate[min(starts) :]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as exc:
        relaxed = _TRAILING_COMMA.sub(r"\1", candidate)
        if relaxed != candidate:
            try:
                return json.loads(relaxed)
            except json.JSONDecodeError:
                pass
        raise LLMError(f"response is not valid JSON: {exc}") from exc


LLM_TIMEOUT = 30.0
LLM_RETRIES = 1


class OpenAILLM:
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 3_000,
        timeout: float | None = None,
        max_retries: int | None = None,
        base_url: str | None = None,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = (timeout if timeout is not None
                        else _seconds("KORA_LLM_TIMEOUT", LLM_TIMEOUT))
        self.max_retries = (max_retries if max_retries is not None
                            else _count("KORA_LLM_RETRIES", LLM_RETRIES))
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._base_url = (base_url or os.environ.get("KORA_LLM_BASE_URL")
                          or os.environ.get("OPENAI_BASE_URL") or None)
        self._client = None

    @property
    def client(self):
        if self._client is None:
            if not self._api_key:
                raise LLMError(
                    "OPENAI_API_KEY is not set; put it in .env or pass api_key=. "
                    "A local endpoint still needs one -- any value will do."
                )
            try:
                from openai import OpenAI
            except ImportError:
                raise LLMError("the openai package is missing; pip install openai") from None
            self._client = OpenAI(
                api_key=self._api_key, timeout=self.timeout,
                max_retries=self.max_retries,
                **({"base_url": self._base_url} if self._base_url else {}),
            )
        return self._client

    def _limits(self) -> dict:
        if self.model.startswith(("gpt-5", "o1", "o3", "o4")):
            limits: dict = {"max_completion_tokens": max(self.max_tokens, 4_000)}
            effort = os.environ.get("KORA_REASONING")
            if effort:
                limits["reasoning_effort"] = effort
            return limits
        limits = {"temperature": self.temperature, "max_tokens": self.max_tokens}
        if self.model.startswith("gpt-oss"):
            limits["max_tokens"] = max(self.max_tokens, 4_000)
            if os.environ.get("KORA_REASONING"):
                limits["reasoning_effort"] = os.environ["KORA_REASONING"]
        return limits

    def generate(self, prompt: str, *, system: str | None = None) -> LLMResponse:
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        started = time.monotonic()
        try:
            completion = self.client.chat.completions.create(
                model=self.model, messages=messages, **self._limits())
        except Exception as exc:
            raise LLMError(f"{self.model} request failed: {exc}") from None

        usage = completion.usage
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        output_tokens = getattr(usage, "completion_tokens", 0) or 0
        output_tokens = max(output_tokens, (getattr(usage, "total_tokens", 0) or 0) - prompt_tokens)
        return LLMResponse(
            text=(completion.choices[0].message.content or ""),
            input_tokens=prompt_tokens,
            output_tokens=output_tokens,
            latency_seconds=time.monotonic() - started,
            model=self.model,
        )


def _seconds(name: str, default: float) -> float:
    value = _number(name, float, default)
    return value if value > 0 else default


def _count(name: str, default: int) -> int:
    value = _number(name, int, default)
    return value if value >= 0 else default


def _number(name: str, kind, default: T) -> T:
    written = os.environ.get(name, "")
    try:
        return kind(written)
    except (TypeError, ValueError):
        return default
