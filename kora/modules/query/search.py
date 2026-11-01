from __future__ import annotations

from dataclasses import dataclass

from kora.modules.query.models import K_RELATIONS, Candidate, Context, Execution
from kora.prompts import PROMPTS
from kora.tools.dialect import quote_if_needed
from kora.tools.llm import LLMError, extract_json

MAX_LINES = 120


@dataclass(frozen=True)
class Fact:
    entity: str
    relation: str
    direction: str
    target: str | None
    name: str | None
    kind: str | None

    @property
    def query(self) -> str:
        written = quote_if_needed(self.entity)
        pattern = f"{written} {self.relation} ?x" if self.direction == "out" else f"?x {self.relation} {written}"
        return f"SELECT ?x WHERE {{ {pattern} }}"

    def line(self) -> str:
        if self.target is None:
            return f"  {quote_if_needed(self.entity)} {self.relation} ({self.direction}) [{self.relation}]"
        who = f"{self.name or '(unnamed record)'} ({self.kind or '?'}) [{self.target}]"
        if self.direction == "out":
            return f"  {quote_if_needed(self.entity)} {self.relation} -> {who}"
        return f"  {who} {self.relation} -> {quote_if_needed(self.entity)}"


def neighbourhood(kg, embedder, ctx: Context) -> list[Fact]:
    return _facts(kg, embedder, ctx.anchor_ids, ctx.task.goal, ctx.failed_routes)


def pick(llm, kg, embedder, ctx: Context, facts: list[Fact]) -> list[tuple[Candidate, Execution]]:
    for hop in (1, 2):
        if not facts:
            return []
        said = _ask(llm, ctx, hop, facts)
        relations = {f.relation: f for f in facts if f.target is None}
        chosen = [relations[a] for a in said["answers"] if a in relations]
        if chosen:
            candidate = Candidate(id="search", query=chosen[0].query, relations=[chosen[0].relation])
            execution = Execution.run(kg, candidate, ctx.task.sets)
            ctx.usage.kg_queries += 1
            ctx.usage.seconds += execution.latency_seconds
            return [(candidate, execution)]
        targets = {f.target: f for f in facts if f.target}
        picks = [a for a in said["answers"] if a in targets]
        if picks:
            fact = targets[picks[0]]
            same = [p for p in picks if targets[p].relation == fact.relation
                    and targets[p].direction == fact.direction and targets[p].entity == fact.entity]
            candidate = Candidate(id="search", query=fact.query, relations=[fact.relation])
            names = {p: targets[p].name for p in same if targets[p].name}
            execution = Execution(candidate_id="search", query=fact.query,
                                  rows=[{"?x": p} for p in same], names=names)
            return [(candidate, execution)]
        step = said["step"]
        if hop == 2 or step not in targets:
            return []
        facts = _facts(kg, embedder, [step], ctx.task.goal, ctx.failed_routes)
    return []


def _facts(kg, embedder, entities: list[str], goal: str, failed=None) -> list[Fact]:
    facts: list[Fact] = []
    for entity in entities:
        every = kg.properties(entity)
        scores = embedder.similarity(goal, [e.predicate for e in every])
        ranked = [e for _, e in sorted(zip(scores, every, strict=True), key=lambda pair: -pair[0])]
        if len(every) > K_RELATIONS:
            facts += [Fact(entity, e.predicate, e.direction, None, None, None) for e in ranked]
            continue
        opened = kg.opened(entity, [e.predicate for e in ranked], failed=failed)
        described = {(e.predicate, e.direction): e for e in opened.entries}
        facts += [Fact(entity, e.predicate, e.direction, x.id, x.name, x.kind)
                  for e in ranked if (e.predicate, e.direction) in described
                  for x in described[(e.predicate, e.direction)].examples]
    return facts[:MAX_LINES]


def _ask(llm, ctx: Context, hop: int, facts: list[Fact]) -> dict:
    prompt = PROMPTS.get("query.explore")
    response = llm.generate(prompt.render(goal=ctx.task.goal, kind=ctx.task.expected_entity_type or "(any)",
                                          hop=hop, max_hops=2, listing="\n".join(f.line() for f in facts)),
                            system=prompt.system)
    ctx.usage.charge(response)
    try:
        payload = extract_json(response.text)
    except LLMError:
        payload = {}
    payload = payload if isinstance(payload, dict) else {}
    answers = payload.get("answers")
    return {"answers": [str(a) for a in answers] if isinstance(answers, list) else [],
            "step": str(payload.get("step") or "")}
