"""Naming what a question is anchored on, and grounding those names."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from kora.provenance.models import EntityRef as Entity
from kora.provenance.models import Spend
from kora.prompts import PROMPTS, Prompt
from kora.tools.llm import LLMError, LLMProvider, extract_json

DEFAULT_PROMPT = "ner.extract"


class Mention(BaseModel):
    """Something a question names, before it is grounded."""

    model_config = ConfigDict(extra="forbid")

    text: str
    kind: str = "other"
    score: float = 0.0


class Anchors(BaseModel):
    """What a question is anchored on, and where that came from."""

    model_config = ConfigDict(extra="forbid")

    entities: list[Entity] = Field(default_factory=list)
    mentions: list[Mention] = Field(default_factory=list)
    source: str = "benchmark"
    spend: Spend = Field(default_factory=Spend)

    @property
    def grounded(self) -> bool:
        return bool(self.entities)


class RecognizedAnchors:
    """Name the entities with a model, then ground them against the graph."""

    def __init__(
        self,
        llm: LLMProvider,
        resolver: Any,
        prompt: Prompt | None = None,
        kg_name: str = "the knowledge graph",
        limit: int = 4,
    ):
        self.llm = llm
        self.resolver = resolver
        self.prompt = prompt or PROMPTS.get(DEFAULT_PROMPT)
        self.kg_name = kg_name
        self.limit = limit

    def anchors(self, question: str, metadata: dict[str, Any]) -> Anchors:  # noqa: ARG002
        mentions, spend = self.name(question)
        entities = self.ground(mentions)
        return Anchors(
            entities=entities, mentions=mentions,
            source="recognised", spend=spend,
        )

    def name(self, question: str) -> tuple[list[Mention], Spend]:
        rendered = self.prompt.render(question=question, kg=self.kg_name)
        response = self.llm.generate(rendered, system=self.prompt.system)
        spend = Spend(tokens=response.total_tokens,
                      seconds=response.latency_seconds, llm_calls=1)
        try:
            payload = extract_json(response.text)
        except LLMError:
            return [], spend
        raw = payload.get("entities") if isinstance(payload, dict) else payload
        if not isinstance(raw, list):
            return [], spend
        return [
            Mention(text=str(item.get("mention", "")).strip(),
                    kind=str(item.get("kind", "other")))
            for item in raw
            if isinstance(item, dict) and str(item.get("mention", "")).strip()
        ][: self.limit], spend

    def ground(self, mentions: list[Mention]) -> list[Entity]:
        found: list[Entity] = []
        seen: set[str] = set()
        for mention in mentions:
            match = self.resolver.resolve_one(mention.text)
            if match and match not in seen:
                seen.add(match)
                found.append(Entity(id=match, label=mention.text))
        return found
