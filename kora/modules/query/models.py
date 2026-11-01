from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from kora.provenance.models import LocalSchema, PathSchema, SchemaEntry
from kora.provenance.view import SHOWN, TaskView
from kora.tools.kg import KGResult

MAX_ANCHORS = 5
K_RELATIONS = 50
GENERIC = frozenset({"type", "kind", "entity", "entities", "null", "none", "thing", "things",
                     "name", "names", "value", "list", "set", "object", "item"})


def words(text: str) -> set[str]:
    spaced = "".join(" " + c if c.isupper() and n and text[n - 1].islower() else c
                     for n, c in enumerate(text or ""))
    return set("".join(c.lower() if c.isalnum() else " " for c in spaced).split())


def reaches(kinds: list[str], expected: str) -> bool:
    want = {w for w in words(expected) if len(w) >= 4 and w not in GENERIC}
    return any(want & words(kind) or any(w in _squashed(kind) for w in want if len(w) >= 6)
               for kind in kinds if kind)


def _squashed(text: str) -> str:
    return "".join(c for c in text.lower() if c.isalnum())


class Outcome(StrEnum):
    SUCCESS = "SUCCESS"
    NO_VALID_CANDIDATE = "NO_VALID_CANDIDATE"
    EMPTY_RESULT = "EMPTY_RESULT"
    EXECUTION_ERROR = "EXECUTION_ERROR"
    INSUFFICIENT_RESULT = "INSUFFICIENT_RESULT"
    MAX_ATTEMPTS_REACHED = "MAX_ATTEMPTS_REACHED"


class Usage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tokens: int = 0
    seconds: float = 0.0
    llm_calls: int = 0
    kg_queries: int = 0

    def charge(self, response) -> None:
        self.tokens += response.total_tokens
        self.seconds += response.latency_seconds
        self.llm_calls += 1


class Candidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    query: str
    relations: list[str] = Field(default_factory=list)
    output_variable: str = "?x"
    problems: list[str] = Field(default_factory=list)
    repairs: list[str] = Field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.problems


class Execution(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    query: str
    rows: list[dict[str, str]] = Field(default_factory=list)
    error: str | None = None
    latency_seconds: float = 0.0
    names: dict[str, str] = Field(default_factory=dict)

    @classmethod
    def of(cls, candidate: Candidate, result: KGResult) -> Execution:
        return cls(candidate_id=candidate.id, query=result.query, rows=result.rows,
                   error=result.error, latency_seconds=result.latency_seconds)

    @classmethod
    def run(cls, kg, candidate: Candidate, sets: dict[str, list[str]]) -> Execution:
        try:
            execution = cls.of(candidate, kg.execute(candidate.query, sets=sets))
        except Exception as error:
            return cls(candidate_id=candidate.id, query=candidate.query, error=f"{type(error).__name__}: {error}")
        shown = list(dict.fromkeys(v for row in execution.rows for v in row.values()))[:SHOWN]
        try:
            names = kg.labels(shown)
        except Exception:
            return execution
        return execution.model_copy(update={"names": {i: n for i, n in names.items() if n}})

    def column(self, name: str) -> list[str]:
        return list(dict.fromkeys(row[name] for row in self.rows if row.get(name)))


class Verdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    outcome: Outcome
    reason: str = ""
    answer_variable: str = ""
    entities: list[str] = Field(default_factory=list)
    on_the_way: bool = False
    best: bool = False

    @property
    def ok(self) -> bool:
        return self.outcome == Outcome.SUCCESS


class Attempt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    number: int
    candidates: list[Candidate] = Field(default_factory=list)
    executions: list[Execution] = Field(default_factory=list)
    verdicts: dict[str, Verdict] = Field(default_factory=dict)
    selected: str | None = None
    outcome: Outcome = Outcome.NO_VALID_CANDIDATE
    entities: list[str] = Field(default_factory=list)
    on_the_way: bool = False

    @property
    def ok(self) -> bool:
        return self.outcome == Outcome.SUCCESS

    def ran(self, candidate_id: str | None) -> Execution | None:
        return next((e for e in self.executions if e.candidate_id == candidate_id), None)

    @property
    def query(self) -> str | None:
        chosen = self.ran(self.selected)
        return chosen.query if chosen else None

    @property
    def reason(self) -> str:
        chosen = self.verdicts.get(self.selected or "")
        if chosen and chosen.reason:
            return chosen.reason[:300]
        problems = [f"{c.id}: {'; '.join(c.problems)}" for c in self.candidates if c.problems]
        return "; ".join(problems)[:300]


class Page(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = "(no relations)"
    lines: list[tuple[str, SchemaEntry]] = Field(default_factory=list)
    listed: list[tuple[str, SchemaEntry]] = Field(default_factory=list)
    routes: list[tuple[str, PathSchema]] = Field(default_factory=list)
    vocabulary: set[str] = Field(default_factory=set)
    types: set[str] = Field(default_factory=set)
    schemas: list[LocalSchema] = Field(default_factory=list)
    schema_relations: set[str] = Field(default_factory=set)

    @property
    def relations(self) -> set[str]:
        return ({e.predicate for _, e in self.listed} | {r for _, p in self.routes for r in p.relations}
                | {x for _, p in self.routes for x in p.properties} | self.schema_relations)

    @property
    def properties(self) -> set[str]:
        return {x.rsplit(".", 1)[-1] for _, p in self.routes for x in p.properties}

    @property
    def mediators(self) -> set[str]:
        return {e.predicate for _, e in self.lines if e.mediator and not e.target_types and not e.target_type}

    @property
    def oriented(self) -> set[tuple[str, str, str]]:
        return {(anchor, e.predicate, e.direction) for anchor, e in self.listed}

    def kinds(self, relations: set[str]) -> list[str]:
        found = [k for _, e in self.lines if e.predicate in relations
                 for k in [*e.target_types, e.target_type or "", *(x.kind or "" for x in e.examples)]]
        return found + [p.target_type or "" for _, p in self.routes if set(p.relations) <= relations]


class QueryResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str = "FAILED"
    query: str | None = None
    entities: list[str] = Field(default_factory=list)
    attempts: list[Attempt] = Field(default_factory=list)
    pages: list[LocalSchema] = Field(default_factory=list)
    failure_type: Outcome | None = None
    diagnosis: str | None = None
    usage: Usage = Field(default_factory=Usage)

    @property
    def ok(self) -> bool:
        return self.status == "SUCCESS"

    @property
    def attempt_count(self) -> int:
        return len(self.attempts)


class Context(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: TaskView
    attempts: list[Attempt] = Field(default_factory=list)
    discovered: list[str] = Field(default_factory=list)
    pages: list[LocalSchema] = Field(default_factory=list)
    failed_routes: set[tuple[str, str]] = Field(default_factory=set)
    usage: Usage = Field(default_factory=Usage)

    @property
    def anchor_ids(self) -> list[str]:
        return list(dict.fromkeys(self.task.anchor_ids + self.discovered))[:MAX_ANCHORS]

    @property
    def tried(self) -> dict[str, str]:
        marks = {relation: verdict for _, relation, verdict in self.task.tried}
        for attempt in self.attempts:
            for candidate in attempt.candidates:
                verdict = attempt.verdicts.get(candidate.id)
                for relation in candidate.relations if verdict else []:
                    marks[relation] = str(verdict.outcome)
        return marks

    def succeeded(self, attempt: Attempt) -> QueryResult:
        return QueryResult(status="SUCCESS", query=attempt.query, entities=list(attempt.entities),
                           attempts=list(self.attempts), pages=list(self.pages), usage=self.usage)

    def failed(self) -> QueryResult:
        last = next((a for a in reversed(self.attempts) if a.reason), None) or \
            (self.attempts[-1] if self.attempts else Attempt(number=0))
        return QueryResult(status="FAILED", attempts=list(self.attempts), pages=list(self.pages),
                           failure_type=last.outcome, usage=self.usage,
                           diagnosis=last.reason or "no attempt produced a usable result")
