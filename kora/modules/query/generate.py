from __future__ import annotations

from kora.modules.query.models import Candidate, Context, Page
from kora.prompts import PROMPTS
from kora.tools.llm import LLMError, extract_json

GENERATED = 4


def generate(llm, ctx: Context, page: Page, seeded: list[Candidate] = (), prompt=None) -> list[Candidate]:
    task = ctx.task
    prompt = prompt or PROMPTS.get("query.generate")
    response = llm.generate(prompt.render(
        question=task.question or "(none)", goal=task.goal, expected_output=_output(task),
        entities=_entities(ctx), requirements=_requirements(task), schema=page.text,
        seeds=_seeds(seeded), memory="", history=_history(ctx),
        strategy=task.hint or "(first attempt)",
        k=GENERATED), system=prompt.system)
    ctx.usage.charge(response)
    written = {" ".join(c.query.split()) for c in seeded}
    queries = dict.fromkeys(" ".join(str(item.get("query", "")).split()) for item in _items(response.text))
    fresh = [q for q in queries if q and q not in written]
    return [Candidate(id=f"q{n}", query=q) for n, q in enumerate(fresh[:GENERATED], start=1)]


def _output(task) -> str:
    kinds = [task.expected_entity_type] if task.expected_entity_type else []
    return f"entity_set[{' | '.join(kinds)}]" if kinds else "entity_set"


def _requirements(task) -> str:
    return "\n".join(f"- {r}" for r in task.requirements) or "- satisfy the task goal"


def _seeds(seeded: list[Candidate]) -> str:
    return "\n".join(f"  {c.id}: {c.query}" for c in seeded) or "(none)"


def _history(ctx: Context) -> str:
    ran = [f"  {e.query}  -> {a.verdicts[e.candidate_id].outcome}, {len(e.rows)} rows"
           for a in ctx.attempts for e in a.executions]
    earlier = [f"  {r}: {v}" for r, v in {r: v for _, r, v in ctx.task.tried}.items()]
    header = ["PREVIOUS ATTEMPTS IN THIS CALL (do not repeat these queries)"] if ran else []
    return "\n".join(header + ran + earlier)


def _entities(ctx: Context) -> str:
    labels = ctx.task.labels
    given_ids = set(ctx.task.anchor_ids)
    ids = ctx.anchor_ids
    given = ", ".join(f"{i} ({labels[i]})" if i in labels else i for i in ids if i in given_ids) or "(none)"
    discovered = [i for i in ids if i not in given_ids]
    if not discovered:
        return given
    return (f"{given}\nFOUND BY AN EARLIER ATTEMPT (query from these -- they are the "
            f"step this question needs): {', '.join(discovered)}")


def _items(text: str) -> list[dict]:
    try:
        payload = extract_json(text)
    except LLMError:
        return []
    found = payload.get("candidates") if isinstance(payload, dict) else None
    return [c for c in found if isinstance(c, dict)] if isinstance(found, list) else []
