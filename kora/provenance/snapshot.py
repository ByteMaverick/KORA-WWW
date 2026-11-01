"""What leaves a run: one serialisable object, with everything in it."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from kora.provenance.models import (
    AnswerCandidate,
    AnswerContract,
    EvaluationResult,
    EvidenceRecord,
    GroundedEntity,
    LocalSchema,
    LocalTaskState,
    Plan,
    RequirementState,
    ResourceState,
    SetRecord,
    StateEvent,
    SufficiencyResult,
)


class TaskGraph(BaseModel):
    """Every task known to the run, and what could run now."""

    model_config = ConfigDict(extra="forbid")

    tasks: dict[str, LocalTaskState] = Field(default_factory=dict)
    ready_frontier: list[str] = Field(default_factory=list)
    active_task_ids: list[str] = Field(default_factory=list)


class ExecutionHistory(BaseModel):
    """Every decision, in order."""

    model_config = ConfigDict(extra="forbid")

    events: list[StateEvent] = Field(default_factory=list)


class BudgetSnapshot(BaseModel):
    """Resources as a flat record, with the remainders already worked out."""

    model_config = ConfigDict(extra="forbid")

    token_budget_total: int | None = None
    tokens_used: int = 0
    tokens_remaining: int | None = None

    time_budget_seconds: float | None = None
    time_used_seconds: float = 0.0
    time_remaining_seconds: float | None = None

    kg_queries_used: int = 0
    llm_calls_used: int = 0

    @classmethod
    def of(cls, resources: ResourceState) -> BudgetSnapshot:
        return cls(
            token_budget_total=resources.token_budget,
            tokens_used=resources.tokens_spent,
            tokens_remaining=resources.tokens_remaining,
            time_budget_seconds=resources.time_budget_seconds,
            time_used_seconds=resources.seconds_spent,
            time_remaining_seconds=resources.seconds_remaining,
            kg_queries_used=resources.kg_queries,
            llm_calls_used=resources.llm_calls,
        )


class KoraStateSnapshot(BaseModel):
    """One finished run, as plain data."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    question_id: str | None = None
    question: str

    root_entities: list[GroundedEntity] = Field(default_factory=list)

    answer_contract: AnswerContract | None = None
    requirements: dict[str, RequirementState] = Field(default_factory=dict)

    grounded_entities: dict[str, GroundedEntity] = Field(default_factory=dict)
    evidence: list[EvidenceRecord] = Field(default_factory=list)

    task_graph: TaskGraph = Field(default_factory=TaskGraph)
    budget: BudgetSnapshot = Field(default_factory=BudgetSnapshot)

    incumbent_answer: AnswerCandidate | None = None

    plan: Plan | None = None
    schemas: dict[str, LocalSchema] = Field(default_factory=dict)
    verdicts: dict[str, list[SufficiencyResult]] = Field(default_factory=dict)
    sets: dict[str, SetRecord] = Field(default_factory=dict)

    history: ExecutionHistory = Field(default_factory=ExecutionHistory)

    question_type: str = ""
    coverage: float = 0.0
    stop_reason: str = ""
    sufficient: bool = False

    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryIngestionRecord(BaseModel):
    """The object handed to a persistent store: one run, whole."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    state: KoraStateSnapshot
    evaluation: EvaluationResult
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def for_memory(self) -> tuple[dict[str, Any], list[dict[str, Any]], dict]:
        snap = self.state
        state = {
            "run_id": snap.run_id,
            "question": snap.question,
            "question_type": snap.question_type,
            "sufficient": snap.sufficient,
            "stop_reason": snap.stop_reason,
            "coverage": snap.coverage,
            "tokens": snap.budget.tokens_used,
        }

        trace: list[dict[str, Any]] = []
        for task in snap.task_graph.tasks.values():
            goal = task.contract.goal if task.contract else ""
            entity_types = []
            anchors = {}
            for entity_id in task.grounded_input_ids:
                entity = snap.grounded_entities.get(entity_id)
                if entity is None:
                    continue
                anchors[entity_id] = entity.entity_type or "entity"
                if entity.entity_type and entity.entity_type not in entity_types:
                    entity_types.append(entity.entity_type)

            for attempt in task.attempts:
                strategy = ""
                for candidate in attempt.candidates:
                    if candidate.id == attempt.selected_candidate_id:
                        strategy = candidate.strategy or ""
                trace.append({
                    "stage": "query_attempt",
                    "goal": goal,
                    "query": attempt.executed_query or "",
                    "strategy": strategy,
                    "outcome": "OK" if attempt.success
                               else (attempt.failure_type or "FAILED"),
                    "rows": attempt.result_count or 0,
                    "tokens": attempt.token_usage or 0,
                    "entity_types": entity_types,
                    "schema_terms": list(attempt.local_schema.get("relations", [])),
                    "anchors": anchors,
                })

        # The scores travel; the answers do not.
        scores = self.evaluation.model_dump()
        for gold in ("gold_answer", "predicted_answer"):
            scores.pop(gold, None)
        return state, trace, scores
