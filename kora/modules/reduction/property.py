"""The operate's property, chosen by a model from the members' real properties."""

from __future__ import annotations

import re

from kora.provenance.models import EntityRef, Spend
from kora.modules.reduction.models import Candidate, Choice, Operate
from kora.modules.reduction.reduction import _DATE, _orderable, _overlap
from kora.prompts import PROMPTS
from kora.tools.kg import value_of
from kora.tools.llm import LLMError, extract_json

PAGES, TRIED = 3, 8
CAP, NAMED, SAMPLE, BATCH = 30, 5, 40, 200
STEP = " / "
PLAIN = {"lt": "less than", "le": "at most", "gt": "more than", "ge": "at least", "eq": "equal to",
         "ne": "not equal to"}
DATED = {**PLAIN, "lt": "before", "le": "on or before", "gt": "after", "ge": "on or after"}
LATEST = 2100


class Chooser:
    """One per run: a choice per (restriction, kind), asked once, and every choice kept for the row."""

    def __init__(self, llm, kg):
        self.llm, self.kg = llm, kg
        self.chosen: dict[tuple[str, str], str | None] = {}
        self.choices: list[Choice] = []

    def __call__(self, op: Operate, members: list[EntityRef], *, kind: str = "",
                 says: str = "") -> dict[str, list[str]] | None:
        restriction = restriction_of(op.prop, says)
        key = (restriction, kind)
        if key in self.chosen:
            self.choices.append(Choice(path=self.chosen[key], cached=True))
        else:
            candidates = candidates_for(self.kg, members, op.prop)
            choice = choose_property(self.llm, op, members, candidates, kind, restriction=restriction)
            self.chosen[key] = choice.path
            self.choices.append(choice)
        path = self.chosen[key]
        if path is None:
            return None
        return values_for(self.kg, members, path)


def restriction_of(prop: str, says: str) -> str:
    plain = " ".join(says.replace("{", "").replace("}", "").split())
    if not plain:
        return prop
    return f"{prop} ({plain})"


def candidates_for(kg, members: list[EntityRef], words: str) -> list[Candidate]:
    """Every literal path on the first three members with any, best first; at most `CAP`."""
    pages = _pages(kg, members)
    coverage: dict[str, int] = {}
    samples: dict[str, list[str]] = {}
    for page in pages:
        for path, values in page.items():
            if not values:
                continue
            coverage[path] = coverage.get(path, 0) + 1
            samples.setdefault(path, values[:2])
    ordered = sorted(coverage, key=lambda p: (-coverage[p], [-n for n in _overlap(words, p)]))
    return [Candidate(path=p, samples=samples[p], coverage=coverage[p], sampled=len(pages)) for p in ordered[:CAP]]


def _pages(kg, members: list[EntityRef]) -> list[dict[str, list[str]]]:
    pages = []
    for member in members[:TRIED]:
        if member.entity_type == "value":
            continue
        page = _page(kg, member.id)
        if page:
            pages.append(page)
        if len(pages) == PAGES:
            break
    return pages


def _page(kg, entity_id: str) -> dict[str, list[str]]:
    if hasattr(kg, "properties_with_samples"):
        return kg.properties_with_samples([entity_id]).get(entity_id, {})
    try:
        entries = list(kg.properties(entity_id))
    except Exception:
        entries = []
    page = {}
    for entry in entries:
        if entry.direction == "out":
            found = _orderable(_read(kg, entity_id, entry.predicate))
            if found:
                page[entry.predicate] = found[:2]
    return page


def choose_property(llm, op: Operate, members: list[EntityRef], candidates: list[Candidate], kind: str,
                    *, restriction: str = "", prompt=None) -> Choice:
    restriction = restriction or op.prop
    if not candidates:
        choice = Choice(error="no member has a literal property to choose from")
    elif len(candidates) == 1:
        choice = Choice(path=candidates[0].path, why="the one candidate")
    else:
        prompt = prompt or PROMPTS.get("reduction.property")
        response = llm.generate(render(prompt, op, members, candidates, kind, restriction), system=prompt.system)
        spend = Spend(tokens=response.total_tokens, llm_calls=1, seconds=response.latency_seconds)
        choice = read_reply(response.text, candidates).model_copy(update={"called": True, "spend": spend})
    return choice


def render(prompt, op: Operate, members: list[EntityRef], candidates: list[Candidate], kind: str,
           restriction: str) -> str:
    lines = []
    for n, c in enumerate(candidates, start=1):
        shown = ", ".join(s if len(s) <= SAMPLE else s[:SAMPLE] + "..." for s in c.samples)
        lines.append(f"{n}. {c.path}   e.g. {shown}   ({c.coverage} of {c.sampled})")
    return prompt.render(restriction=restriction, comparison=comparison(op), kind=kind or "(unspecified)",
                         members=", ".join(e.label or e.id for e in members[:NAMED]),
                         candidates="\n".join(lines))


def comparison(op: Operate) -> str:
    if op.op in ("argmax", "argmin"):
        return f"{op.op} over the value"
    year = re.fullmatch(r"\d{4}", op.value or "") and int(op.value) <= LATEST
    words = DATED if _DATE.match(op.value or "") or year else PLAIN
    return f"{words.get(op.cmp, op.cmp)} {op.value}"


def read_reply(text: str, candidates: list[Candidate]) -> Choice:
    """The path, if it is one shown; otherwise None and why."""
    try:
        reply = extract_json(text)
    except LLMError as exc:
        return Choice(error=str(exc))
    if not isinstance(reply, dict):
        return Choice(error="the reply is not a JSON object")
    why, path = str(reply.get("why") or ""), reply.get("path")
    if path is None or str(path).strip().lower() in ("", "null", "none"):
        return Choice(why=why)
    shown = {_squeezed(c.path): c.path for c in candidates}
    if _squeezed(str(path)) not in shown:
        return Choice(why=why, error=f"{path!r} is not one of the candidates shown")
    return Choice(path=shown[_squeezed(str(path))], why=why)


def _squeezed(path: str) -> str:
    return re.sub(r"\s+", "", path)


def values_for(kg, members: list[EntityRef], path: str) -> dict[str, list[str]]:
    ids = [e.id for e in members if e.entity_type != "value"]
    found: dict[str, list[str]] = {}
    if hasattr(kg, "values_of"):
        for start in range(0, len(ids), BATCH):
            found.update(kg.values_of(ids[start:start + BATCH], path))
        return found
    for entity_id in ids:
        found[entity_id] = _read_path(kg, entity_id, path)
    return found


def _read_path(kg, entity_id: str, path: str) -> list[str]:
    relations = path.split(STEP)
    if len(relations) == 1:
        return _read(kg, entity_id, relations[0])
    rows = kg.execute(f"SELECT ?x WHERE {{ {entity_id} {relations[0]} ?m . ?m {relations[1]} ?x }}").rows
    return list(dict.fromkeys(str(row["?x"]) for row in rows if row.get("?x")))


def _read(kg, entity_id: str, prop: str) -> list[str]:
    if hasattr(kg, "value_of"):
        return kg.value_of(entity_id, prop)
    return value_of(kg, entity_id, prop)
