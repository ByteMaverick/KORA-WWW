"""Sufficiency: is the target answered?"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Protocol, runtime_checkable

from kora.provenance.models import (
    ClaimVerdict,
    EntityRef,
    RequirementStatus,
    Spend,
    SufficiencyResult,
    entity_type_matches,
)
from kora.provenance.state import KoraState
from kora.provenance.view import SufficiencyView
from kora.prompts import PROMPTS, Prompt
from kora.tools.llm import LLMError, LLMProvider, extract_json

ROOT = "T0"

__all__ = ["ROOT", "SufficiencyPolicy", "SufficiencyResult"]
SHOWN = 15


FOUND = "found"


@runtime_checkable
class SufficiencyPolicy(Protocol):
    """Global state and a target id in, verdict out."""

    def check(self, state: KoraState, target_id: str = ROOT) -> SufficiencyResult: ...


F1_ENOUGH = 0.7


class OracleSufficiency:
    """The gold, compared with what the root stands behind."""

    def __init__(self, gold: list[str], resolve: Callable[[list[str]], list[str]] | None = None,
                 threshold: float = 0.5, *, children: SufficiencyPolicy | None,
                 precision: float = 0.5, finals: bool = False,
                 f1: float = F1_ENOUGH, stop_on_hit: bool = True):
        self.gold = [g for g in gold if g]
        self.resolve = resolve or _no_names
        self.threshold = threshold
        self.precision = precision
        self.children = children
        self.finals = finals
        self.f1 = f1
        self.stop_on_hit = stop_on_hit

    def hits_at_1(self, output: list[EntityRef]) -> bool:
        if not output or not self.gold:
            return False
        first = (output[0].label or output[0].id or "").strip().lower()
        return any(first and (first in g.lower() or g.lower() in first)
                   for g in self.gold)

    def check(self, state: KoraState, target_id: str = ROOT) -> SufficiencyResult:
        view = state.sufficiency_view(target_id)
        if view.target_id != ROOT and not (self.finals and state.is_final(view.target_id)):
            return self.children.check(state, view.target_id)
        output = self._as_the_gold_asks(state, view.current_output)
        score = self.recall(output)
        tight = self.precision_of(output)
        f1 = 0.0 if score + tight <= 0 else 2 * score * tight / (score + tight)
        hit = self.hits_at_1(output)
        strict = score >= self.threshold and tight >= self.precision
        generous = (self.f1 > 0 and f1 >= self.f1) or (self.stop_on_hit and hit)
        sufficient = bool(self.gold) and (strict or generous)
        unresolved = [r.id for r in view.requirements if r.status is not RequirementStatus.RESOLVED]
        return SufficiencyResult(
            target_id=view.target_id,
            sufficient=sufficient,
            # Deliberately only figures: the reason can reach a prompt.
            reason=(f"the first entity it stands behind is the gold (hits@1); "
                    f"recall {score:.2f}, precision {tight:.2f}"
                    if sufficient and hit and not strict else
                    f"F1 {f1:.2f} at or above {self.f1:.2f}; "
                    f"recall {score:.2f}, precision {tight:.2f}"
                    if sufficient and not strict else
                    f"recall {score:.2f} of the gold at or above {self.threshold:.2f}, "
                    f"precision {tight:.2f}"
                    if sufficient else
                    f"recall {score:.2f} of the gold, below {self.threshold:.2f}"
                    if score < self.threshold else
                    f"recall {score:.2f} but precision {tight:.2f}, below {self.precision:.2f}: "
                    f"the set is too broad"),
            # What was scored, so hits@1, precision and `got` read the same set as the recall.
            answer_entities=output,
            satisfied_requirements=[r.id for r in view.requirements] if sufficient else [],
            missing_requirements=[] if sufficient else unresolved,
            recall=score,
            coverage=state.coverage(),
        )

    def _as_the_gold_asks(self, state: KoraState, projected: list[EntityRef]
                          ) -> list[EntityRef]:
        if not self.gold or not projected:
            return projected
        counted = state.produced_by(ROOT)
        if counted == projected or self.recall(projected) >= self.threshold:
            return projected
        return counted if self.recall(counted) > self.recall(projected) else projected

    def recall(self, entities: list[EntityRef]) -> float:
        if not self.gold:
            return 0.0
        ids = [e.id for e in entities]
        got = {x.lower() for x in ids + [e.label for e in entities if e.label]
               + list(self.resolve(ids)) if x}
        found = sum(1 for g in self.gold if any(matches(g, x) for x in got))
        return found / len(self.gold)

    def precision_of(self, entities: list[EntityRef]) -> float:
        if not entities or not self.gold:
            return 0.0
        matched = 0
        for entity in entities[:50]:
            names = [x for x in [entity.id, entity.label or ""]
                     + list(self.resolve([entity.id])) if x]
            if any(matches(g, x) for x in names for g in self.gold):
                matched += 1
        return matched / min(len(entities), 50)


_WORDS = re.compile(r"[a-z0-9]+")


def matches(gold: str, candidate: str) -> bool:
    """Is this returned name the gold one?"""
    first, second = _WORDS.findall(gold.lower()), _WORDS.findall(candidate.lower())
    if not first or not second:
        return False
    if all(word.isdigit() for word in first):
        short, long = (first, second) if len(first) <= len(second) else (second, first)
        return long[:len(short)] == short
    short, long = (first, second) if len(first) <= len(second) else (second, first)
    return any(long[i:i + len(short)] == short
               for i in range(len(long) - len(short) + 1))


def _no_names(_ids: list[str]) -> list[str]:
    return []


DEFAULT_PROMPT = "sufficiency.verify"
SCORE_THRESHOLD = 0.10
CROWD = 50


class JudgeSufficiency:
    def __init__(self, llm: LLMProvider, prompt: Prompt | None = None, threshold: float | None = None):
        self.llm = llm
        self.prompt = prompt or PROMPTS.get(DEFAULT_PROMPT)
        self.threshold = SCORE_THRESHOLD if threshold is None else threshold

    def check(self, state: KoraState, target_id: str = ROOT) -> SufficiencyResult:
        view = state.sufficiency_view(target_id)
        reason = without_a_call(view)
        if reason == NO_CLAIMS:
            score = count_factor(len(view.current_output), view.wanted_count)
            return _verdict(view, [], score, NO_CLAIMS, Spend(), score >= self.threshold)
        if reason is not None:
            return _verdict(view, [], 0.0, reason, Spend(), False)
        from kora.modules.sufficiency import estimator as _estimator
        if _estimator.enabled():
            return self._estimator(view)
        payload, spend = self._ask(view)
        if payload is None:
            return _verdict(view, claim_verdicts(view.claims, {}), 0.0,
                            "the judge returned nothing usable", spend, False)
        claims = claim_verdicts(view.claims, payload)
        reason = reason_of(claims)
        accepted = passes([c.verdict for c in claims], payload.get("kind_ok") is True)
        confidence = _confidence_of(claims, payload, view)
        return _verdict(view, claims, confidence, reason, spend, accepted)

    def _estimator(self, view: SufficiencyView) -> SufficiencyResult:
        from kora.modules.sufficiency import estimator as _estimator
        values = view_values(view)
        state = _estimator.state_of(values['claims'], values['path'], values['count'], values['kind'],
                               values['wanted_count'], values['candidates'])
        (p,), seconds = _estimator.score([state], [getattr(view, 'goal', '') or ''],
                                    [_estimator.extra([e.id for e in view.current_output])])
        ids = [line.split(':', 1)[0] for line in view.claims]
        if p is None:
            claims = [ClaimVerdict(id=i, verdict='NOT ADDRESSED', why='estimator unavailable') for i in ids]
            return _verdict(view, claims, 0.0, 'estimator unavailable: no verdict', Spend(seconds=seconds), False)
        ok = p >= _estimator.THRESHOLD
        _estimator.remember(view.target_id, p, [e.id for e in view.current_output],
                       [getattr(e, 'label', '') or '' for e in view.current_output])
        claims = [ClaimVerdict(id=i, verdict='ESTABLISHED' if ok else 'NOT ADDRESSED', why=f'estimator p={p:.2f}') for i in ids]
        return _verdict(view, claims, p, f'estimator p(sufficient)={p:.2f}', Spend(seconds=seconds), ok)

    def _ask(self, view: SufficiencyView) -> tuple[dict | None, Spend]:
        response = self.llm.generate(render_view(view, self.prompt), system=self.prompt.system)
        spend = Spend(tokens=response.total_tokens, seconds=response.latency_seconds, llm_calls=1)
        try:
            payload = extract_json(response.text)
        except LLMError:
            return None, spend
        usable = isinstance(payload, dict) and isinstance(payload.get("claims"), list)
        return (payload if usable else None), spend


NO_CLAIMS = "no claims: judged on kind and count"


def without_a_call(view: SufficiencyView) -> str | None:
    """Why the judge decides this view in code, or None when it asks the model."""
    fact = on_the_facts(view, contract=view.target_id != ROOT)
    if fact is not None:
        return fact
    return NO_CLAIMS if not view.claims and view.planned else None


def view_values(view: SufficiencyView) -> dict[str, object]:
    """What `sufficiency.verify` is rendered from: the claims, the path, the candidate set."""
    accepted = view.accepted_entity_types or ([view.expected_entity_type] if view.expected_entity_type else [])
    return {"claims": "\n".join(view.claims) or "(none)", "path": "\n".join(view.path) or "(nothing traversed)",
            "count": len(view.current_output), "kind": " | ".join(accepted) or "(any)",
            "wanted_count": view.wanted_count, "candidates": _entities(view.current_output),
            "reply": reply_claims(view.claims)}


def render_view(view: SufficiencyView, prompt: Prompt) -> str:
    return prompt.render(**view_values(view))


def on_the_facts(view: SufficiencyView, contract: bool = True) -> str | None:
    """What code decides without a model."""
    output = view.current_output
    if not output:
        return "nothing was produced"
    inputs = {e.id for e in view.anchors} | {e.id for role, refs in view.bindings.items() if role != FOUND for e in refs}
    if inputs and all(e.id in inputs for e in output):
        return "the output is what the task started from; nothing was learned"
    if not contract:
        return None
    if len(output) > CROWD and view.wanted_count == "one" and not view.consumers:
        return (f"{len(output)} entities where the question wants one: the relation was too broad; "
                "constrain the kind or choose a more specific relation")
    expected = view.accepted_entity_types or ([view.expected_entity_type] if view.expected_entity_type else [])
    kinds = [e.entity_type for e in output]
    if expected and kinds and all(kinds) and not any(entity_type_matches(k, w) for k in kinds for w in expected):
        return f"the output is of kind {kinds[0]}, not the kind the contract asks for ({' | '.join(expected)})"
    return None


def claim_verdicts(lines: list[str], payload: dict) -> list[ClaimVerdict]:
    raw = payload.get("claims")
    said = {str(c.get("id")): c for c in (raw if isinstance(raw, list) else []) if isinstance(c, dict)}
    found = []
    for line in lines:
        cid = line.split(":", 1)[0]
        entry = said.get(cid, {})
        verdict = str(entry.get("verdict") or "NOT ADDRESSED").upper()
        found.append(ClaimVerdict(id=cid, verdict=verdict if verdict in VERDICTS else "NOT ADDRESSED",
                                  why=str(entry.get("why") or "")[:200]))
    return found


VERDICTS = ("ESTABLISHED", "NOT", "NOT ADDRESSED")


def reply_claims(lines: list[str]) -> str:
    """The reply's "claims" list with every claim id already in place."""
    return json.dumps([{"id": line.split(":", 1)[0], "verdict": "|".join(VERDICTS), "why": ""} for line in lines])


def reason_of(claims: list[ClaimVerdict]) -> str:
    return "; ".join(f"{c.id} {c.verdict}: {c.why}" for c in claims)[:300] or "no claims to verify"


def confidence(payload: dict, view: SufficiencyView) -> float:
    return _confidence_of(claim_verdicts(view.claims, payload), payload, view)


def _confidence_of(claims: list[ClaimVerdict], payload: dict, view: SufficiencyView) -> float:
    return score([c.verdict for c in claims], payload.get("kind_ok") is True, payload.get("doubt"),
                 len(view.current_output), view.wanted_count)


def passes(verdicts: list[str], kind_ok: bool) -> bool:
    """Whether verdicts accept a set: the right kind, one claim the path shows, none it contradicts."""
    return kind_ok and "ESTABLISHED" in verdicts and "NOT" not in verdicts


def score(verdicts: list[str], kind_ok: bool, doubt, n: int, wanted: str) -> float:
    if not verdicts:
        return 0.0
    established = sum(v == "ESTABLISHED" for v in verdicts) / len(verdicts)
    return established * (1.0 if kind_ok else 0.4) * count_factor(n, wanted) * (0.8 if _text(doubt) else 1.0)


def count_factor(n: int, wanted: str) -> float:
    if wanted == "one":
        return 1.0 if n == 1 else 0.6 if n <= 3 else 0.3
    if wanted == "few":
        return 1.0 if n <= 10 else 0.6
    return 1.0


def _verdict(view: SufficiencyView, claims: list[ClaimVerdict], score: float, reason: str,
             spend: Spend, sufficient: bool) -> SufficiencyResult:
    established = [c.id for c in claims if c.verdict == "ESTABLISHED"]
    return SufficiencyResult(
        target_id=view.target_id, sufficient=sufficient, reason=reason,
        answer_entities=view.current_output, claims=claims, confidence=score,
        satisfied_requirements=established,
        missing_requirements=[c.id for c in claims if c.id not in established],
        coverage=len(established) / len(claims) if claims else 0.0, spend=spend, basis="judge")


def _text(value) -> str:
    text = str(value).strip() if value is not None else ""
    return "" if text.lower() in ("null", "none", "") else text


def _shown(shown: int, total: int, what: str) -> str:
    return (f"  (showing {shown} of {total} {what}; the other {total - shown} are part of the same "
            f"set and are not shown to you. They are not missing.)")


def _entities(entities: list[EntityRef]) -> str:
    if not entities:
        return "  (none)"
    lines = [f"  {e.label or e.id} ({e.id})" + (f" :type {e.entity_type}" if e.entity_type else "")
             for e in entities[:SHOWN]]
    if len(entities) > SHOWN:
        lines.append(_shown(SHOWN, len(entities), "entities"))
    return "\n".join(lines)
