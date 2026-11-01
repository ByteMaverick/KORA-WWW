"""The names a question states, found once per question so the Controller can ground them."""

from __future__ import annotations

from kora.provenance.models import Spend

__all__ = ["StatedNames", "is_a_name"]


def is_a_name(text: str) -> bool:
    """Is this mention a proper name worth looking up?"""
    text = (text or "").strip()
    return bool(text) and any(w[:1].isupper() for w in text.split())


class StatedNames:
    """The names a question spells out, found once per question."""

    def __init__(self, recogniser):
        self.recogniser = recogniser
        self.spend = Spend()
        self._seen: dict[str, list[str]] = {}

    def __call__(self, state) -> list[str]:
        question = state.question.text
        if question in self._seen:
            return self._seen[question]
        try:
            mentions, spend = self.recogniser.name(question)
        except Exception:
            # Naming is an improvement to the call, never a reason to fail it.
            return []
        self.spend.tokens += spend.tokens
        self.spend.seconds += spend.seconds
        self.spend.llm_calls += spend.llm_calls
        self._seen[question] = [m.text for m in mentions if is_a_name(m.text)]
        return self._seen[question]
