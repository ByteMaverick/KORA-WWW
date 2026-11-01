"""The knowledge judge: the model rates the root's list from what it knows of the world."""

from __future__ import annotations

import re

from kora.provenance.models import Spend, SufficiencyResult
from kora.provenance.state import KoraState
from kora.provenance.view import SufficiencyView
from kora.modules.sufficiency.sufficiency import ROOT, SufficiencyPolicy, on_the_facts
from kora.prompts import PROMPTS, Prompt
from kora.tools.llm import LLMError, LLMProvider

PROMPT = "sufficiency.knowledge"
LEVELS = {"none": 0.0, "few": 0.25, "half": 0.5, "most": 0.75, "all": 1.0}
WANTED = {"one": "one answer", "few": "a few answers", "many": "many answers", "": "an unknown number of answers"}
NAMES_SHOWN = 12
QUERY_SHOWN = 500
_LEVELS = re.compile(r"^\W*(?:candidate\s*)?(\d+)\W+correct\W*(none|few|half|most|all)\W+covers\W*"
                     r"(none|few|half|most|all)", re.I | re.M)
_THINKING = re.compile(r"<(thought|think)>.*?</\1>", re.S)


def knowledge_values(view: SufficiencyView) -> dict[str, object]:
    """What `sufficiency.knowledge` is rendered from at one decision point."""
    accepted = view.accepted_entity_types or ([view.expected_entity_type] if view.expected_entity_type else [])
    names = [e.label or e.id for e in view.current_output]
    shown = " ; ".join(names[:NAMES_SHOWN])
    more = len(names) - min(len(names), NAMES_SHOWN)
    query = " ; ".join(view.path)[:QUERY_SHOWN] or "(the combined result of the question's sub-questions)"
    return {"question": view.question or view.goal, "kind": " | ".join(accepted) or "not stated",
            "wanted": WANTED.get(view.wanted_count, WANTED[""]), "query": query, "count": len(names),
            "names": shown + (f" ; and {more} more" if more > 0 else "")}


def level_scores(reply: str, candidates: int) -> dict[int, float]:
    """Candidate number -> the smaller of its two levels, plus a hundredth of the larger to break ties."""
    found = {}
    for n, correct, covers in _LEVELS.findall(_THINKING.sub("", reply or "")):
        if 1 <= int(n) <= candidates and int(n) not in found:
            low, high = sorted((LEVELS[correct.lower()], LEVELS[covers.lower()]))
            found[int(n)] = low + 0.01 * high
    return found


def level_score(reply: str) -> float | None:
    """`level_scores` for a single candidate; None when the reply has no line for it."""
    return level_scores(reply, 1).get(1)


class KnowledgeSufficiency:
    """Every target by the knowledge prompt; or, given `children`, the root by it and every other target by them."""

    def __init__(self, llm: LLMProvider, *, children: SufficiencyPolicy | None = None, prompt: Prompt | None = None,
                 threshold: float = 0.5):
        self.llm, self.children = llm, children
        self.prompt = prompt or PROMPTS.get(PROMPT)
        self.threshold = threshold

    def check(self, state: KoraState, target_id: str = ROOT) -> SufficiencyResult:
        view = state.sufficiency_view(target_id)
        if view.target_id != ROOT and self.children is not None:
            return self.children.check(state, view.target_id)
        fact = on_the_facts(view, contract=False)
        if fact is not None:
            return self._result(view, 0.0, fact, Spend(), False)
        try:
            response = self.llm.generate(self.prompt.render(**knowledge_values(view)), system=self.prompt.system)
        except LLMError as error:
            return self._result(view, 0.0, f"the judge failed: {str(error)[:120]}", Spend(llm_calls=1), False)
        spend = Spend(tokens=response.total_tokens, seconds=response.latency_seconds, llm_calls=1)
        found = level_score(response.text)
        if found is None:
            return self._result(view, 0.0, "the judge's reply has no levels line", spend, False)
        right = found >= self.threshold
        return self._result(view, found, f"levels score {found:.2f}: the list is "
                                          f"{'right' if right else 'not right'} by the model's knowledge", spend, right)

    @staticmethod
    def _result(view: SufficiencyView, score: float, reason: str, spend: Spend, sufficient: bool) -> SufficiencyResult:
        return SufficiencyResult(target_id=view.target_id, sufficient=sufficient, reason=reason,
                                 answer_entities=view.current_output, confidence=score, spend=spend,
                                 basis="judge")
