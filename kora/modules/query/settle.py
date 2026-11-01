from __future__ import annotations

from kora.provenance.models import EvidenceRecord, QueryAttempt, QueryCandidate
from kora.provenance.view import TaskView
from kora.modules.query.models import (
    MAX_ANCHORS,
    Attempt,
    Candidate,
    Context,
    Execution,
    Outcome,
    Page,
    Verdict,
    reaches,
)
from kora.tools.dialect import parse_query, triples_from


def rank(candidates: list[Candidate], page: Page, ctx: Context) -> list[Candidate]:
    valid = [c for c in candidates if c.is_valid]
    keys = {c.id: (_kind(c, page, ctx), _hops(c)) for c in valid}
    ordered = sorted(valid, key=lambda c: keys[c.id])
    return ordered


def settle(number: int, candidates: list[Candidate], results: list[tuple[Candidate, Execution]],
           verdicts: dict[str, Verdict], page: Page, ctx: Context) -> Attempt:
    ran = [c for c, _ in results]
    chosen = next((c for c in ran if verdicts[c.id].ok and verdicts[c.id].best), None) \
        or next((c for c in ran if verdicts[c.id].ok), None) or min(
        (c for c in ran if verdicts[c.id].entities),
        key=lambda c: (_kind(c, page, ctx), ran.index(c), -len(verdicts[c.id].entities)),
        default=None) or (ran[-1] if ran else None)
    verdict = verdicts[chosen.id] if chosen else None
    attempt = Attempt(number=number, candidates=candidates, executions=[e for _, e in results], verdicts=verdicts,
                   selected=chosen.id if chosen else None,
                   outcome=verdict.outcome if verdict else Outcome.NO_VALID_CANDIDATE,
                   entities=_entities(verdict, ran, verdicts),
                   on_the_way=bool(verdict and verdict.on_the_way))
    return attempt


def traverse(kg, attempt: Attempt, ctx: Context) -> list[str]:
    room = MAX_ANCHORS - len(ctx.anchor_ids)
    taken: list[str] = []
    for e in attempt.entities:
        if len(taken) >= room:
            break
        if e in ctx.anchor_ids:
            continue
        if kg.label(e) and not kg.is_hub(e):
            taken.append(e)
    return taken


def evidence_of(view: TaskView, attempt: Attempt) -> list[EvidenceRecord]:
    execution = attempt.ran(attempt.selected)
    if not attempt.ok or execution is None:
        return []
    task = view.task_id or "fast_path"
    return [EvidenceRecord(id=f"{task}_a{attempt.number}_e{n}", subject_id=s, relation=r, object_id=o,
                           source_task_id=view.task_id or "", source_query_id=execution.candidate_id)
            for n, (s, r, o) in enumerate(triples_from(execution.query, execution.rows))]


def as_record(attempt: Attempt, evidence: list[EvidenceRecord]) -> QueryAttempt:
    return QueryAttempt(
        attempt=attempt.number,
        candidates=[QueryCandidate(id=c.id, query=c.query, relations=list(c.relations),
                                   rationale="; ".join(c.repairs) or None,
                                   result_count=_rows(attempt, c.id) or 0, outcome=_outcome(c, attempt))
                    for c in attempt.candidates],
        selected_candidate_id=attempt.selected, executed_query=attempt.query,
        result_count=_rows(attempt, attempt.selected),
        grounded_entity_ids=list(attempt.entities), on_the_way=attempt.on_the_way,
        evidence_ids=[r.id for r in evidence], success=attempt.ok,
        failure_type=None if attempt.ok else str(attempt.outcome),
        diagnosis=None if attempt.ok else (attempt.reason or None))


def _entities(verdict: Verdict | None, ran: list[Candidate], verdicts: dict[str, Verdict]) -> list[str]:
    if verdict is None:
        return []
    if verdict.ok:
        return list(verdict.entities)
    found = list(verdict.entities)
    for c in ran:
        found.extend(e for e in verdicts[c.id].entities if e not in found)
    return found


def _rows(attempt: Attempt, candidate_id: str | None) -> int | None:
    execution = attempt.ran(candidate_id)
    return len(execution.rows) if execution else None


def _outcome(candidate: Candidate, attempt: Attempt) -> str:
    if candidate.problems:
        return "rejected: " + ", ".join(dict.fromkeys(p.split(":", 1)[0] for p in candidate.problems))
    verdict = attempt.verdicts.get(candidate.id)
    if verdict is None:
        return ""
    return str(verdict.outcome) if candidate.id == attempt.selected or not verdict.ok else "unselected"


def _kind(candidate: Candidate, page: Page, ctx: Context) -> int:
    expected = ctx.task.expected_entity_type
    parsed = parse_query(candidate.query)
    typed = [o for _, p, o in (parsed[1] if parsed else []) if p == ":type"]
    relations = set(candidate.relations)
    if expected and reaches(typed + page.kinds(relations), expected):
        return 0
    if expected and reaches(sorted(relations), expected):
        return 1
    return 2


def _hops(candidate: Candidate) -> int:
    return len(candidate.relations)
