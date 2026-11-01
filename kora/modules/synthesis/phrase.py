from __future__ import annotations

from kora.modules.synthesis.synthesis import AnswerDraft
from kora.prompts import PROMPTS
from kora.tools.llm import LLMError, extract_json

__all__ = ["phrase"]


def phrase(draft: AnswerDraft, llm) -> AnswerDraft:
    prompt = PROMPTS.get("synthesis.phrase")
    rendered = prompt.render(kind=draft.kind or "-", count=draft.count, projection=draft.projection,
                             text=draft.text or "(nothing)", names="\n".join(draft.names) or "(none)",
                             unmet=", ".join(draft.unmet) or "(nothing)")
    try:
        said = extract_json(llm.generate(rendered, system=prompt.system).text)
    except LLMError:
        return draft
    used = said.get("names") if isinstance(said, dict) else None
    if not isinstance(used, list) or not said.get("sentence") or not set(map(str, used)) <= set(draft.names):
        return draft
    return draft.model_copy(update={"text": str(said["sentence"])})
