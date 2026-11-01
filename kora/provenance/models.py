"""The pieces a Kora run is made of."""

from __future__ import annotations

import re
import time
import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RequirementStatus(StrEnum):
    """Where a requirement stands."""

    UNRESOLVED = "UNRESOLVED"
    IN_PROGRESS = "IN_PROGRESS"
    RESOLVED = "RESOLVED"
    FAILED = "FAILED"
    UNVERIFIABLE = "UNVERIFIABLE"
    CONTRADICTED = "CONTRADICTED"


class TaskStatus(StrEnum):
    """Where a task is in its life."""

    BLOCKED = "BLOCKED"        # waiting on something it depends on
    READY = "READY"            # dependencies satisfied, free to run
    RUNNING = "RUNNING"
    DECOMPOSED = "DECOMPOSED"  # expanded into children, waiting on them
    STUCK = "STUCK"
    RESOLVED = "RESOLVED"
    FAILED = "FAILED"


TERMINAL = frozenset({TaskStatus.RESOLVED, TaskStatus.FAILED})


class EntityRef(BaseModel):
    """A reference to something in the graph."""

    model_config = ConfigDict(extra="forbid")

    id: str
    label: str | None = None
    entity_type: str | None = None


class QuestionContext(BaseModel):
    """The original question, and what it was grounded in."""

    model_config = ConfigDict(extra="forbid")

    text: str
    question_id: str = ""
    dataset: str = ""
    root_entities: list[EntityRef] = Field(default_factory=list)


class RequirementState(BaseModel):
    """One thing that must be established before the question can be answered."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    id: str
    description: str
    status: RequirementStatus = RequirementStatus.UNRESOLVED
    critical: bool = True
    weight: float = 1.0
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    blocking_task_ids: list[str] = Field(default_factory=list)
    resolved_value: Any = None
    diagnosis: str = ""


TEMPORAL_ANSWER_ENTITY_TYPES = (
    "date",
    "time.event",
    "time.time_span",
    "time.period",
    "sports.sports_championship_event",
    "sports.sports_championship",
    "sports.tournament_event",
)


def answer_entity_types(question: str, planned: str | None = None) -> list[str]:
    """Allowed answer kinds, broadening only explicit ``when`` questions."""
    found = [planned] if planned else []
    if re.search(r"\bwhen\b", question or "", re.IGNORECASE):
        found.extend(TEMPORAL_ANSWER_ENTITY_TYPES)
    return list(dict.fromkeys(value for value in found if value))


def entity_type_matches(actual: str, expected: str) -> bool:
    """Loose match across Freebase's slash, dot, underscore type spellings."""
    def normalise(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()

    left, right = normalise(actual), normalise(expected)
    return bool(left and right and (left in right or right in left))


class Constraint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    args: list[str] = Field(default_factory=list)
    says: str = ""
    op: str = ""        # "" | filter | argmax | argmin | count | verify
    cmp: str = ""       # filter and verify: lt | le | gt | ge | eq | ne
    value: str = ""     # filter and verify: the literal, normalised (31290, 2004-04-30)
    about: str = ""     # the property in the question's words ("undergraduates", "area")
    over: str = ""      # the variable the operation runs over, or "named": the question's named constants


class AnswerContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer_type: str = "entity"      # entity | value | boolean
    answer_projection: str = "identity"
    answer_entity_type: str | None = None
    accepted_entity_types: list[str] = Field(default_factory=list)
    answer_count: str = "few"
    requirements: list[RequirementState] = Field(default_factory=list)
    constraints: list[Constraint] = Field(default_factory=list)
    planner_summary: str | None = None


class GroundedEntity(BaseModel):
    """An entity Kora may query from, and how it got here."""

    model_config = ConfigDict(extra="forbid")

    id: str
    label: str | None = None
    entity_type: str | None = None
    source: str = "initial_grounding"    # initial_grounding | a task id; the first producer
    sources: list[str] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)

    def produced_by_task(self, task_id: str) -> bool:
        return self.source == task_id or task_id in self.sources


class SetRecord(BaseModel):
    """What one move produced, kept under a name whatever anyone thought of it."""

    model_config = ConfigDict(extra="forbid")

    name: str
    entities: list[EntityRef] = Field(default_factory=list)
    kind: str = ""
    produced_by: str = ""
    counts_as: str = "answer"      # answer | superset
    refused: bool = False


class EvidenceRecord(BaseModel):
    """One grounded fact, and where it came from."""

    model_config = ConfigDict(extra="forbid")

    id: str
    subject_id: str
    relation: str
    object_id: str | None = None
    object_value: str | None = None
    source_task_id: str = ""
    source_query_id: str = ""
    requirement_ids: list[str] = Field(default_factory=list)
    confidence: float = 1.0

    @model_validator(mode="after")
    def must_keep_provenance(self) -> EvidenceRecord:
        if not self.source_task_id and not self.source_query_id:
            raise ValueError(
                f"evidence {self.id} has no provenance: it must name the task "
                "or the query that produced it"
            )
        return self


class TaskBindings(BaseModel):
    """The entities a task may start from, by the role they play."""

    model_config = ConfigDict(extra="forbid")

    roles: dict[str, list[EntityRef]] = Field(default_factory=dict)

    def names(self) -> list[str]:
        return list(self.roles)

    def entities(self) -> list[EntityRef]:
        seen: dict[str, EntityRef] = {}
        for refs in self.roles.values():
            for ref in refs:
                seen.setdefault(ref.id, ref)
        return list(seen.values())

    def render(self) -> str:
        if not self.roles:
            return "(none)"
        return "\n".join(
            f"- {name}: " + ", ".join(e.label or e.id for e in refs)
            for name, refs in self.roles.items()
        )


class TaskNode(BaseModel):
    """One piece of work, and what it waits on."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    id: str
    goal: str
    status: TaskStatus = TaskStatus.BLOCKED
    parent_id: str | None = None
    dependencies: list[str] = Field(default_factory=list)
    children: list[str] = Field(default_factory=list)
    requirement_ids: list[str] = Field(default_factory=list)
    satisfies: list[str] = Field(default_factory=list)
    grounded_inputs: list[EntityRef] = Field(default_factory=list)
    bindings: TaskBindings = Field(default_factory=lambda: TaskBindings())
    output_name: str = ""
    depth: int = 0
    latest_children: list[str] = Field(default_factory=list)
    expected_output: str = ""
    counts_as: str = "answer"
    composed: list[EntityRef] = Field(default_factory=list)
    found: dict[str, str | list[str]] = Field(default_factory=dict)
    contract: TaskContract | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    attempts: int = 0
    diagnosis: str = ""
    combine: str = "union"


def labels_for(task: TaskNode | None = None,
               extra: Any = ()) -> dict[str, str]:
    """The names one task is entitled to abstract out of its own text, as name -> kind."""
    found: dict[str, str] = {}
    entities = list(task.grounded_inputs) + task.bindings.entities() if task is not None else []
    for entity in entities + list(extra or []):
        kind = str(entity.entity_type or "")
        if entity.label:
            found.setdefault(str(entity.label), kind)
        if entity.id:
            found.setdefault(str(entity.id), kind)
    return found


class TaskContract(BaseModel):
    """What a task is for, and what finishing it looks like."""

    model_config = ConfigDict(extra="forbid")

    goal: str
    expected_output_type: str | None = None
    expected_entity_type: str | None = None
    accepted_entity_types: list[str] = Field(default_factory=list)


class QueryCandidate(BaseModel):
    """One way of asking the graph, and why it was written that way."""

    model_config = ConfigDict(extra="forbid")

    id: str
    query: str
    strategy: str | None = None
    relations: list[str] = Field(default_factory=list)
    rationale: str | None = None
    result_count: int = 0
    outcome: str = ""


class ValidationResult(BaseModel):
    """Why a candidate was allowed to run, or was not."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    valid: bool
    syntax_valid: bool | None = None
    relations_exist: bool | None = None
    entities_grounded: bool | None = None
    variables_bound: bool | None = None
    output_shape_valid: bool | None = None
    rejection_reason: str | None = None


class RankedCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    rank: int
    score: float | None = None
    rationale: str | None = None


class QueryAttempt(BaseModel):
    """One pass: the schema seen, every candidate written, what ran, what came back."""

    model_config = ConfigDict(extra="forbid")

    attempt: int

    local_schema: dict[str, Any] = Field(default_factory=dict)

    candidates: list[QueryCandidate] = Field(default_factory=list)
    validations: list[ValidationResult] = Field(default_factory=list)
    ranking: list[RankedCandidate] = Field(default_factory=list)

    selected_candidate_id: str | None = None
    executed_query: str | None = None

    result_count: int | None = None
    grounded_entity_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)

    success: bool = False
    on_the_way: bool = False

    failure_type: str | None = None
    diagnosis: str | None = None
    reflection: str | None = None
    next_strategy: str | None = None

    latency_ms: float | None = None
    token_usage: int | None = None


class LocalTaskState(BaseModel):
    """What one task knows about its own subproblem."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    parent_task_id: str | None = None
    requirement_ids: list[str] = Field(default_factory=list)

    contract: TaskContract | None = None
    status: TaskStatus = TaskStatus.BLOCKED
    dependencies: list[str] = Field(default_factory=list)

    grounded_input_ids: list[str] = Field(default_factory=list)
    output_entity_ids: list[str] = Field(default_factory=list)

    attempts: list[QueryAttempt] = Field(default_factory=list)
    local_evidence_ids: list[str] = Field(default_factory=list)


class Spend(BaseModel):
    """What one step cost."""

    model_config = ConfigDict(extra="forbid")

    tokens: int = 0
    seconds: float = 0.0
    llm_calls: int = 0
    kg_queries: int = 0
    tool_calls: int = 0


class ResourceState(BaseModel):
    """What the run may still spend."""

    model_config = ConfigDict(extra="forbid")

    token_budget: int = 0
    time_budget_seconds: float = 0.0
    tokens_spent: int = 0
    seconds_spent: float = 0.0
    llm_calls: int = 0
    kg_queries: int = 0
    tool_calls: int = 0

    soft_seconds: float = 0.0

    reserve_tokens: int = 0
    reserve_seconds: float = 0.0

    started_at: float = Field(default_factory=time.monotonic)

    @property
    def tokens_remaining(self) -> int:
        return max(0, self.token_budget - self.tokens_spent)

    @property
    def seconds_remaining(self) -> float:
        return max(0.0, self.time_budget_seconds - self.seconds_spent)

    @property
    def spendable_tokens(self) -> int:
        return max(0, self.tokens_remaining - self.reserve_tokens)

    @property
    def spendable_seconds(self) -> float:
        return max(0.0, self.seconds_remaining - self.reserve_seconds)

    def wall_seconds(self) -> float:
        return time.monotonic() - self.started_at


class AnswerCandidate(BaseModel):
    """The best answer currently supportable, and what supports it."""

    model_config = ConfigDict(extra="forbid")

    answer: Any = None
    supporting_requirement_ids: list[str] = Field(default_factory=list)
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    confidence: float | None = None


class EventKind(StrEnum):
    """Every stage that writes something worth learning from later."""

    PLAN_CREATED = "PLAN_CREATED"

    TASK_CREATED = "TASK_CREATED"
    TASK_GRAPH_EXPANDED = "TASK_GRAPH_EXPANDED"
    TASK_STARTED = "TASK_STARTED"
    TASK_RESOLVED = "TASK_RESOLVED"
    TASK_FAILED = "TASK_FAILED"
    TASK_ABANDONED = "TASK_ABANDONED"
    TASK_REOPENED = "TASK_REOPENED"
    TASK_RESPLIT = "TASK_RESPLIT"
    REDUCED = "REDUCED"
    SET_PRODUCED = "SET_PRODUCED"
    MOVE_CHOSEN = "MOVE_CHOSEN"

    SCHEMA_RETRIEVED = "SCHEMA_RETRIEVED"
    CANONICALISED = "CANONICALISED"
    QUERY_CANDIDATES_GENERATED = "QUERY_CANDIDATES_GENERATED"
    QUERY_VALIDATED = "QUERY_VALIDATED"
    QUERY_RANKED = "QUERY_RANKED"
    QUERY_EXECUTED = "QUERY_EXECUTED"
    REFLECTION_ADDED = "REFLECTION_ADDED"

    LOCAL_COMPLETION_CHECKED = "LOCAL_COMPLETION_CHECKED"

    ENTITY_GROUNDED = "ENTITY_GROUNDED"
    EVIDENCE_ADDED = "EVIDENCE_ADDED"
    REQUIREMENT_RESOLVED = "REQUIREMENT_RESOLVED"
    REQUIREMENT_BLOCKED = "REQUIREMENT_BLOCKED"

    GLOBAL_SUFFICIENCY_CHECKED = "GLOBAL_SUFFICIENCY_CHECKED"
    RECOVERY_DECISION = "RECOVERY_DECISION"
    FINAL_ANSWER_CREATED = "FINAL_ANSWER_CREATED"

    BUDGET_CONSUMED = "BUDGET_CONSUMED"


class StateEvent(BaseModel):
    """One decision, recorded so it can be learned from."""

    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(default_factory=lambda: f"evt_{uuid.uuid4().hex[:10]}")
    event_type: EventKind
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    source_component: str = ""

    task_id: str | None = None
    requirement_id: str | None = None

    context: dict[str, Any] = Field(default_factory=dict)
    considered: list[Any] = Field(default_factory=list)
    chosen: Any = None
    rationale: str | None = None
    outcome: dict[str, Any] = Field(default_factory=dict)


class EvaluationResult(BaseModel):
    """How the run scored, and what it cost."""

    model_config = ConfigDict(extra="forbid")

    gold_answer: Any
    predicted_answer: Any

    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    exact_match: bool | None = None
    task_score: float | None = None

    total_tokens: int | None = None
    total_kg_queries: int | None = None
    total_latency_seconds: float | None = None


class Example(BaseModel):
    """One neighbour a relation reaches: an instance, with its name and kind."""

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str | None = None
    kind: str | None = None


class SchemaEntry(BaseModel):
    """One predicate available on an entity: what it is, and what it leads to."""

    model_config = ConfigDict(extra="forbid")

    predicate: str
    direction: str  # "out" | "in"
    sample_object: str | None = None
    count: int = 0
    target_type: str | None = None
    target_types: list[str] = Field(default_factory=list)
    sample_name: str | None = None
    mediator: bool = False
    examples: list[Example] = Field(default_factory=list)
    sampled: bool = False


class PathSchema(BaseModel):
    """A two-step route through a mediator, which is how n-ary facts are read."""

    model_config = ConfigDict(extra="forbid")

    relations: list[str]
    directions: list[str] = Field(default_factory=list)
    target_type: str | None = None
    sample_name: str | None = None
    count: int = 0
    mediator: bool = True
    mediator_type: str | None = None
    properties: list[str] = Field(default_factory=list)


class LocalSchema(BaseModel):
    """Question-conditioned local schema around a grounded entity."""

    model_config = ConfigDict(extra="forbid")

    entity_id: str
    entity_type: str | None = None
    entries: list[SchemaEntry] = Field(default_factory=list)
    paths: list[PathSchema] = Field(default_factory=list)

    @property
    def predicates(self) -> list[str]:
        return [e.predicate for e in self.entries]

    def render(self) -> str:
        lines = [f"entity: {self.entity_id}"]
        for entry in self.entries:
            arrow = "->" if entry.direction == "out" else "<-"
            sample = f"  e.g. {entry.sample_object}" if entry.sample_object else ""
            lines.append(f"  {arrow} {entry.predicate} ({entry.count}){sample}")
        return "\n".join(lines)


class Variable(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    kind: str = ""


class Answer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    variable: str = ""
    kind: str = ""
    projection: str = "identity"
    key: str = ""
    count: str = "few"
    type: str = "entity"


class Need(BaseModel):
    model_config = ConfigDict(extra="forbid")

    variable: str
    kind: str = ""
    from_constraints: list[str] = Field(default_factory=list)
    given: list[str] = Field(default_factory=list)


class Recipe(BaseModel):
    model_config = ConfigDict(extra="forbid")

    needs: list[Need] = Field(default_factory=list)
    closes: list[str] = Field(default_factory=list)


class ClaimVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    verdict: str
    why: str = ""


class PlannedRequirement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    goal: str
    critical: bool = True


class Plan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    goal: str
    variables: list[Variable] = Field(default_factory=list)
    constraints: list[Constraint] = Field(default_factory=list)
    answer: Answer = Field(default_factory=Answer)
    constants: dict[str, str] = Field(default_factory=dict)
    shape: str = "direct"
    recipe: Recipe = Field(default_factory=Recipe)
    requirements: list[PlannedRequirement] = Field(default_factory=list)
    rationale: str = ""
    spend: Spend = Field(default_factory=Spend)
    prompt_id: str = ""
    refused_operations: list[str] = Field(default_factory=list)


class SufficiencyResult(BaseModel):
    """Whether the target is answered, and on what grounds."""

    model_config = ConfigDict(extra="forbid")

    target_id: str = "T0"
    sufficient: bool
    reason: str = ""
    answer_entities: list[EntityRef] = Field(default_factory=list)
    satisfied_requirements: list[str] = Field(default_factory=list)
    missing_requirements: list[str] = Field(default_factory=list)
    claims: list[ClaimVerdict] = Field(default_factory=list)
    confidence: float = 0.0
    recall: float | None = None
    coverage: float = 0.0
    spend: Spend = Field(default_factory=Spend)
    basis: str = "oracle"
