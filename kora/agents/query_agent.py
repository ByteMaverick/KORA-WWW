from __future__ import annotations

from kora.provenance.state import KoraState
from kora.provenance.view import TaskView
from kora.modules.query.execute import execute_all
from kora.modules.query.generate import generate
from kora.modules.query.judge import LeafJudge, judge
from kora.modules.query.models import Context, QueryResult
from kora.modules.query.page import build_page
from kora.modules.query.search import neighbourhood, pick
from kora.modules.query.seeds import seeds
from kora.modules.query.settle import as_record, evidence_of, rank, settle, traverse
from kora.modules.query.validate import validate

ROUNDS = 2


class QueryAgent:
    def __init__(self, llm, kg, embedder, judge=None, rounds: int = ROUNDS):
        self.llm, self.kg, self.embedder = llm, kg, embedder
        self.judge = judge or LeafJudge(llm)
        self.rounds = rounds

    def run(self, state: KoraState, task_id: str | None = None) -> QueryResult:
        view = state.get_view(task_id)
        result = self.solve(view)
        self.record(state, view, result)
        return result

    def solve(self, view: TaskView) -> QueryResult:
        ctx = Context(task=view)
        rounds = 1 if view.closing else self.rounds
        for number in range(1, rounds + 1):
            page       = build_page(self.kg, self.embedder, ctx)
            seeded     = seeds(ctx, page)
            candidates = validate(seeded + generate(self.llm, ctx, page, seeded), page, ctx)
            ordered    = rank(candidates, page, ctx)
            results    = execute_all(self.kg, ordered, ctx)
            verdicts   = judge(self.judge, ctx, results)
            chosen     = settle(number, candidates, results, verdicts, page, ctx)
            ctx.attempts.append(chosen)
            if chosen.ok:
                return ctx.succeeded(chosen)
            ctx.discovered += traverse(self.kg, chosen, ctx)
        if view.closing:
            return ctx.failed()
        facts    = neighbourhood(self.kg, self.embedder, ctx)
        results  = pick(self.llm, self.kg, self.embedder, ctx, facts)
        verdicts = judge(self.judge, ctx, results)
        chosen   = settle(self.rounds + 1, [c for c, _ in results], results, verdicts, page, ctx)
        ctx.attempts.append(chosen)
        return ctx.succeeded(chosen) if chosen.ok else ctx.failed()

    def record(self, state: KoraState, view: TaskView, result: QueryResult, decide: bool = True) -> None:
        state.budget_consumed(tokens=result.usage.tokens, seconds=result.usage.seconds,
                              llm_calls=result.usage.llm_calls, kg_queries=result.usage.kg_queries)
        for schema in result.pages:
            if state.schema_for(schema.entity_id) is None:
                state.schema_retrieved(schema.entity_id, schema, rationale=f"probed for {view.task_id or 'the question'}")
        evidence = [r for a in result.attempts for r in evidence_of(view, a)]
        if evidence:
            state.evidence_added(evidence, rationale=f"query agent: {view.goal}")
        if view.task_id is None:
            return
        for attempt in result.attempts:
            state.query_attempted(view.task_id, as_record(attempt, evidence_of(view, attempt)))
        if not decide:
            return
        if result.ok:
            state.task_resolved(view.task_id, rationale=result.query or "")
        else:
            state.task_failed(view.task_id, diagnosis=result.diagnosis or "", rationale=f"{result.attempt_count} attempts")

