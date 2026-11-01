from __future__ import annotations

import json

from kora.provenance.models import EntityRef
from kora.provenance.view import ROOT, SufficiencyView, members
from kora.modules.query.models import Candidate, Context, Execution, Outcome, Verdict
from kora.modules.sufficiency.sufficiency import (
    claim_verdicts,
    on_the_facts,
    passes,
    reason_of,
    reply_claims,
)
from kora.prompts import PROMPTS
from kora.tools.dialect import output_variables, parse_query
from kora.tools.llm import LLMError, extract_json


def judge(judge, ctx: Context, results: list[tuple[Candidate, Execution]]) -> dict[str, Verdict]:
    facts = {c.id: _no_rows(e) for c, e in results if e.error or not e.rows}
    rows = [(c, e) for c, e in results if c.id not in facts]
    verdicts = {**facts, **(judge.verdicts(ctx, rows) if rows else {})}
    return verdicts


class LeafJudge:
    def __init__(self, llm, prompt=None, system: str | None = None):
        self.llm = llm
        self.system = PROMPTS.get("sufficiency.verify").system if system is None else system
        self.prompt = prompt or PROMPTS.get("query.judge")

    def verdicts(self, ctx: Context, results: list[tuple[Candidate, Execution]]) -> dict[str, Verdict]:
        decided, asked, values = self.shown(ctx, results)
        return {**decided, **(self._asked(ctx, asked, values) if asked else {})}

    def shown(self, ctx: Context, results: list[tuple[Candidate, Execution]]
              ) -> tuple[dict[str, Verdict], list[tuple[Candidate, Execution]], dict[str, str]]:
        facts = {c.id: on_the_facts(_view(ctx, c, e)) for c, e in results}
        decided = {c.id: _decided(c, e, facts[c.id]) for c, e in results if facts[c.id]}
        asked = [(c, e) for c, e in results if not facts[c.id]]
        values = {"candidates": "\n\n".join(_block(ctx, c, e) for c, e in asked),
                  "reply": reply_candidates(ctx, asked)} if asked else {}
        return decided, asked, values

    def _asked(self, ctx: Context, results: list[tuple[Candidate, Execution]],
               values: dict[str, str]) -> dict[str, Verdict]:
        from kora.modules.sufficiency import estimator as _estimator
        if _estimator.enabled():
            return self._estimator(ctx, results)
        response = self.llm.generate(self.prompt.render(**values), system=self.system)
        ctx.usage.charge(response)
        payload = _payload(response.text)
        said = {str(r.get("id")): r for r in _candidates(payload)}
        verdicts = {c.id: self._verdict(ctx, c, e, said.get(c.id, {})) for c, e in results}
        verdicts = _marked(verdicts, payload)
        return verdicts

    def _estimator(self, ctx: Context, results: list[tuple[Candidate, Execution]]) -> dict[str, Verdict]:
        from kora.modules.sufficiency import estimator as _estimator
        states, variables, columns, extras = [], [], [], []
        for c, e in results:
            variable = _variable(c, e)
            found = e.column(variable)
            names = [e.names.get(v, v) for v in found]
            named = members(names)
            claims = [line.replace('{' + ctx.task.variable + '}', '{' + f'{ctx.task.variable} = {named}' + '}')
                      if ctx.task.variable else line for line in ctx.task.claims]
            states.append(_estimator.state_of('\n'.join(claims), _path(ctx, e, named, len(found)), len(found),
                                         ctx.task.expected_entity_type or '(any)', ctx.task.expected_count,
                                         _estimator.entity_lines(list(zip(names, found)), len(found))))
            variables.append(variable)
            columns.append(found)
            extras.append(_estimator.extra(found, c.query, getattr(ctx.task, "chain", None)))
        probabilities, _seconds = _estimator.score(states, [ctx.task.goal or ''] * len(states), extras)
        verdicts = {}
        for (c, e), variable, found, p in zip(results, variables, columns, probabilities):
            _estimator.remember(ctx.task.task_id, p, found, [e.names.get(v, v) for v in found])
            ok = p is not None and p >= _estimator.THRESHOLD
            reason = 'estimator unavailable: no verdict' if p is None else f'estimator p(sufficient)={p:.2f}'
            verdicts[c.id] = Verdict(outcome=Outcome.SUCCESS if ok else Outcome.INSUFFICIENT_RESULT,
                                     reason=reason, answer_variable=variable, entities=found)
        scored = {c.id: p for (c, _), p in zip(results, probabilities) if p is not None and verdicts[c.id].ok}
        if scored:
            best = max(scored, key=scored.get)
            verdicts[best] = verdicts[best].model_copy(update={'best': True})
        return verdicts

    def _verdict(self, ctx: Context, candidate: Candidate, execution: Execution, said: dict) -> Verdict:
        variable = _variable(candidate, execution)
        found = execution.column(variable)
        claims = claim_verdicts(ctx.task.claims, said)
        verdicts, kind_ok = [c.verdict for c in claims], said.get("kind_ok") is True
        return Verdict(
            outcome=Outcome.SUCCESS if passes(verdicts, kind_ok) else Outcome.INSUFFICIENT_RESULT,
            reason=reason_of(claims),
            answer_variable=variable, entities=found)


class KnowledgeLeafJudge:
    """The knowledge judge inside the query agent (`KORA_ROOT_JUDGE=knowledge`)."""

    def __init__(self, llm, prompt=None, threshold: float = 0.5):
        self.llm, self.threshold = llm, threshold
        self.prompt = prompt or PROMPTS.get("query.knowledge")

    def verdicts(self, ctx: Context, results: list[tuple[Candidate, Execution]]) -> dict[str, Verdict]:
        from kora.modules.sufficiency.knowledge import WANTED, level_scores
        facts = {c.id: on_the_facts(_view(ctx, c, e)) for c, e in results}
        decided = {c.id: _decided(c, e, facts[c.id]) for c, e in results if facts[c.id]}
        asked = [(c, e) for c, e in results if not facts[c.id]]
        if not asked:
            return decided
        blocks = "\n\n".join(_knowledge_block(n, c, e) for n, (c, e) in enumerate(asked, 1))
        text = self.prompt.render(question=ctx.task.goal, kind=ctx.task.expected_entity_type or "not stated",
                                  wanted=WANTED.get(ctx.task.expected_count or "", WANTED[""]), candidates=blocks)
        try:
            response = self.llm.generate(text, system=self.prompt.system)
            ctx.usage.charge(response)
            scores = level_scores(response.text, len(asked))
        except LLMError:
            scores = {}
        best = max(scores, key=scores.get) if scores else None
        found = {}
        for n, (c, e) in enumerate(asked, 1):
            variable = _variable(c, e)
            right = n == best and scores[n] >= self.threshold
            found[c.id] = Verdict(outcome=Outcome.SUCCESS if right else Outcome.INSUFFICIENT_RESULT,
                                  reason=(f"levels score {scores.get(n, 0.0):.2f} by the model's knowledge"
                                          if n in scores else "the judge's reply has no line for it"),
                                  answer_variable=variable, entities=e.column(variable), best=right)
        return {**decided, **found}


def _knowledge_block(n: int, candidate: Candidate, execution: Execution) -> str:
    from kora.modules.sufficiency.knowledge import NAMES_SHOWN, QUERY_SHOWN
    found = execution.column(_variable(candidate, execution))
    names = [execution.names.get(v, v) for v in found]
    shown = " ; ".join(names[:NAMES_SHOWN]) + (f" ; and {len(names) - NAMES_SHOWN} more" if len(names) > NAMES_SHOWN else "")
    return f"CANDIDATE {n}\nquery: {(candidate.query or execution.query)[:QUERY_SHOWN]}\nreturned {len(found)}: {shown}"


def reply_candidates(ctx: Context, results: list[tuple[Candidate, Execution]]) -> str:
    """The reply's "candidates" list with every candidate id and its claim ids in place."""
    claims = json.loads(reply_claims(ctx.task.claims))
    entries = [json.dumps({"id": c.id, "claims": claims, "kind_ok": None, "doubt": None}) for c, _ in results]
    return "[" + ",\n   ".join(entries) + "]"


def _view(ctx: Context, candidate: Candidate, execution: Execution) -> SufficiencyView:
    return SufficiencyView(
        target_id=ctx.task.task_id or ROOT,
        goal=ctx.task.goal,
        expected_entity_type=ctx.task.expected_entity_type,
        anchors=list(ctx.task.anchors),
        bindings={name.removeprefix("@"): [EntityRef(id=i) for i in ids] for name, ids in ctx.task.sets.items()},
        current_output=[EntityRef(id=v) for v in execution.column(_variable(candidate, execution))],
        wanted_count=ctx.task.expected_count)


def _decided(candidate: Candidate, execution: Execution, fact: str) -> Verdict:
    variable = _variable(candidate, execution)
    return Verdict(outcome=Outcome.INSUFFICIENT_RESULT, reason=fact,
                   answer_variable=variable, entities=execution.column(variable))


def _no_rows(execution: Execution) -> Verdict:
    if execution.error:
        return Verdict(outcome=Outcome.EXECUTION_ERROR, reason=f"the graph rejected the query: {execution.error}")
    return Verdict(outcome=Outcome.EMPTY_RESULT, reason="the query ran and returned no rows")


def _payload(text: str) -> dict:
    try:
        payload = extract_json(text)
    except LLMError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _candidates(payload: dict) -> list[dict]:
    found = payload.get("candidates")
    return [c for c in found if isinstance(c, dict)] if isinstance(found, list) else []


def _marked(verdicts: dict[str, Verdict], payload: dict) -> dict[str, Verdict]:
    best = str(payload.get("best") or "").strip()
    if best in verdicts and verdicts[best].ok:
        verdicts[best] = verdicts[best].model_copy(update={"best": True})
    return verdicts


def _variable(candidate: Candidate, execution: Execution) -> str:
    available = output_variables(execution.query)
    return candidate.output_variable if candidate.output_variable in available else next(iter(available), "")


def _block(ctx: Context, candidate: Candidate, execution: Execution) -> str:
    variable = _variable(candidate, execution)
    found = execution.column(variable)
    named = members([execution.names.get(v, v) for v in found])
    claims = [line.replace("{" + ctx.task.variable + "}", "{" + f"{ctx.task.variable} = {named}" + "}")
              if ctx.task.variable else line for line in ctx.task.claims]
    return (f"CANDIDATE {candidate.id}\nCLAIMS\n" + "\n".join(claims)
            + f"\nEVIDENCE (the path that was traversed)\n{_path(ctx, execution, named, len(found))}"
            + f"\nCANDIDATE ANSWER ({len(found)} entities, kind expected: "
            + f"{ctx.task.expected_entity_type or '(any)'}; the question wants {ctx.task.expected_count})\n  {named}")


def _path(ctx: Context, execution: Execution, named: str, n: int) -> str:
    parsed = parse_query(execution.query)
    relations = " / ".join(p for _, p, _ in parsed[1] if not p.startswith(":")) if parsed else ""
    labels = ctx.task.labels
    start = ", ".join(labels.get(a, a) for a in ctx.anchor_ids if a in execution.query) or "(a bound set)"
    return f"{start} -[{relations}]-> {{{named}}} ({n})"
