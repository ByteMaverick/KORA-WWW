"""One canonical state per question."""

from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from kora.provenance import view as views
from kora.provenance.models import (
    TERMINAL,
    AnswerCandidate,
    AnswerContract,
    EntityRef,
    EventKind,
    EvidenceRecord,
    GroundedEntity,
    LocalSchema,
    LocalTaskState,
    Plan,
    QueryAttempt,
    QuestionContext,
    RequirementState,
    RequirementStatus,
    ResourceState,
    SetRecord,
    StateEvent,
    SufficiencyResult,
    TaskNode,
    TaskStatus,
)
from kora.provenance.snapshot import (
    BudgetSnapshot,
    ExecutionHistory,
    KoraStateSnapshot,
    TaskGraph,
)
from kora.provenance.view import SufficiencyView, TaskView


class InvalidTransition(ValueError):
    """A change that would put the state somewhere it may not go."""


ROOT_TASK = "T0"


class KoraState(BaseModel):
    """Everything known about one question, right now."""

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(default_factory=lambda: f"run_{uuid.uuid4().hex[:12]}")
    question: QuestionContext
    contract: AnswerContract | None = None

    grounded_entities: dict[str, GroundedEntity] = Field(
        default_factory=dict
    )
    evidence: list[EvidenceRecord] = Field(default_factory=list)

    tasks: dict[str, TaskNode] = Field(default_factory=dict)
    task_states: dict[str, LocalTaskState] = Field(
        default_factory=dict
    )

    resources: ResourceState = Field(default_factory=ResourceState)

    incumbent: AnswerCandidate | None = None

    plan: Plan | None = None
    schemas: dict[str, LocalSchema] = Field(default_factory=dict)
    verdicts: dict[str, list[SufficiencyResult]] = Field(default_factory=dict)
    sets: dict[str, SetRecord] = Field(default_factory=dict)
    hypergraph: Any = None
    task_graphs: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    stop_reason: str = ""
    sufficient: bool = False
    step: int = 0

    history: list[StateEvent] = Field(default_factory=list)

    regime: str = Field(default="discard", exclude=True)

    @classmethod
    def begin(
        cls,
        question: str,
        question_id: str = "",
        dataset: str = "",
        root_entities: list[EntityRef] | None = None,
        token_budget: int = 0,
        time_budget_seconds: float = 0.0,
    ) -> KoraState:
        entities = list(root_entities or [])
        return cls(
            question=QuestionContext(
                text=question, question_id=str(question_id or ""), dataset=dataset,
                root_entities=entities,
            ),
            grounded_entities={
                entity.id: GroundedEntity(
                    id=entity.id, label=entity.label,
                    entity_type=entity.entity_type, source="initial_grounding",
                )
                for entity in entities
            },
            resources=ResourceState(
                token_budget=token_budget, time_budget_seconds=time_budget_seconds,
            ),
        )

    def requirement(self, requirement_id: str) -> RequirementState:
        for found in self.contract.requirements if self.contract else []:
            if found.id == requirement_id:
                return found
        raise InvalidTransition(f"no such requirement: {requirement_id}")

    def schema_for(self, entity_id: str) -> LocalSchema | None:
        return self.schemas.get(entity_id)

    def verdicts_for(self, target_id: str) -> list[SufficiencyResult]:
        return self.verdicts.get(target_id, [])

    def task(self, task_id: str) -> TaskNode:
        found = self.tasks.get(task_id)
        if found is None:
            raise InvalidTransition(f"no such task: {task_id}")
        return found

    def frontier(self) -> list[str]:
        ready = []
        for task in self.tasks.values():
            if task.status in TERMINAL or task.status in (TaskStatus.RUNNING,
                                                          TaskStatus.DECOMPOSED,
                                                          TaskStatus.STUCK):
                continue
            if self._waiting_on(task):
                continue
            ready.append(task.id)
        return sorted(ready)

    def produced_by(self, task_id: str) -> list[EntityRef]:
        task = self.tasks.get(task_id)
        if task is None:
            return []
        if task.composed:
            return list(task.composed)
        return self._own(task)

    def _own(self, task: TaskNode) -> list[EntityRef]:
        given = {e.id for e in task.grounded_inputs}
        attempts = self.task_states[task.id].attempts if task.id in self.task_states else []
        last = attempts[-1].grounded_entity_ids if attempts and attempts[-1].grounded_entity_ids else None
        passed = {i for a in attempts for i in a.grounded_entity_ids}
        mine = [g for g in self.grounded_entities.values()
                if g.produced_by_task(task.id) and g.id not in given]
        # ...or one bound as `found`: an offer can repeat an id a failed pass also returned.
        found = {e.id for e in task.bindings.roles.get("found", [])}
        offered = [g for g in mine if g.id not in passed or g.id in found]
        refs = [
            EntityRef(id=g.id, label=g.label, entity_type=g.entity_type)
            for g in (offered or mine) if offered or last is None or g.id in last
        ]
        return sorted(refs, key=lambda e: e.label is None)

    def projected(self, entities: list[EntityRef]) -> list[EntityRef]:
        from kora.modules.synthesis.synthesis import apply  # local: avoids a cycle

        projection = self.contract.answer_projection if self.contract else "identity"
        return apply(projection, entities)

    def coverage(self) -> float:
        requirements = self.contract.requirements if self.contract else []
        total = sum(found.weight for found in requirements)
        if total <= 0:
            return 0.0
        done = sum(
            found.weight for found in requirements
            if found.status is RequirementStatus.RESOLVED
        )
        return done / total

    def _waiting_on(self, task: TaskNode) -> list[str]:
        unmet = []
        for dependency in task.dependencies:
            other = self.tasks.get(dependency)
            if other is None or other.status is not TaskStatus.RESOLVED:
                unmet.append(dependency)
        return unmet

    def _refresh_statuses(self) -> None:
        for task in self.tasks.values():
            if task.status in TERMINAL:
                continue
            if task.status in (TaskStatus.RUNNING, TaskStatus.DECOMPOSED, TaskStatus.STUCK):
                continue
            waiting = self._waiting_on(task)
            settled = TaskStatus.BLOCKED if waiting else TaskStatus.READY
            if task.status is not settled:
                task.status = settled

    def _record(
        self,
        event_type: EventKind,
        source_component: str,
        rationale: str = "",
        task_id: str | None = None,
        requirement_id: str | None = None,
        context: dict[str, Any] | None = None,
        considered: list[Any] | None = None,
        chosen: Any = None,
        outcome: dict[str, Any] | None = None,
    ) -> None:
        event = StateEvent(
            event_type=event_type,
            source_component=source_component,
            task_id=task_id,
            requirement_id=requirement_id,
            context=context or {},
            considered=considered or [],
            chosen=chosen,
            rationale=rationale or None,
            outcome=outcome or {},
        )
        self.history.append(event)

    def planned(self, plan: Plan, contract: AnswerContract | None, rationale: str = "") -> None:
        self.plan = plan
        self.contract = contract
        self._record(
            EventKind.PLAN_CREATED, "planner", rationale, task_id=None,
            context={"goal": plan.goal, "shape": plan.shape},
            chosen={"requirements": [r.id for r in plan.requirements]},
            outcome={"requirements": len(plan.requirements), "is_the_contract": contract is not None},
        )

    def task_created(self, task: TaskNode, rationale: str = "") -> None:
        if task.id in self.tasks:
            raise InvalidTransition(f"task {task.id} already exists")
        for dependency in task.dependencies:
            if dependency not in self.tasks:
                raise InvalidTransition(
                    f"task {task.id} depends on {dependency}, which does not exist"
                )
        self.tasks[task.id] = task
        self._refresh_statuses()
        self._record(
            EventKind.TASK_CREATED, "controller", rationale, task_id=task.id,
            context={"goal": task.goal, "dependencies": list(task.dependencies)},
            chosen=task.id,
            outcome={"status": str(task.status)},
        )

    def task_started(self, task_id: str, rationale: str = "") -> None:
        task = self.task(task_id)
        if task.status in TERMINAL:
            raise InvalidTransition(
                f"task {task_id} is {task.status} and cannot start again"
            )
        task.status = TaskStatus.RUNNING
        task.attempts += 1
        self._record(
            EventKind.TASK_STARTED, "controller", rationale, task_id=task_id,
            context={"attempt": task.attempts},
        )

    def task_resolved(self, task_id: str, rationale: str = "") -> None:
        task = self.task(task_id)
        task.status = TaskStatus.RESOLVED
        self._refresh_statuses()
        self._record(
            EventKind.TASK_RESOLVED, "controller", rationale, task_id=task_id,
            outcome={"frontier": self.frontier()},
        )

    def ready(self) -> list[str]:
        return self.frontier()

    def stuck(self) -> list[str]:
        return sorted(t.id for t in self.tasks.values() if t.status is TaskStatus.STUCK)

    def split_done(self, task_id: str) -> bool:
        task = self.task(task_id)
        return bool(task.latest_children) and all(
            self.tasks[c].status is TaskStatus.RESOLVED for c in task.latest_children if c in self.tasks)

    def task_stuck(self, task_id: str, diagnosis: str = "", rationale: str = "") -> None:
        task = self.task(task_id)
        if task.status is TaskStatus.RESOLVED:
            raise InvalidTransition(f"task {task_id} is RESOLVED and cannot be stuck")
        if task.composed:
            self._withdraw(task_id, {e.id for e in task.composed})
            task.composed = []
        task.status = TaskStatus.STUCK
        task.diagnosis = diagnosis or task.diagnosis
        self._record(EventKind.TASK_FAILED, "controller", rationale or "stuck", task_id=task_id,
                     outcome={"diagnosis": task.diagnosis, "stuck": self.stuck()})

    def task_split(self, parent_id: str, children: list[TaskNode], combine: str,
                   rationale: str = "") -> None:
        parent = self.task(parent_id)
        if parent.status is not TaskStatus.STUCK:
            raise InvalidTransition(f"task {parent_id} is {parent.status}; only a STUCK task is split")
        for c in children:
            c.parent_id, c.depth = parent_id, parent.depth + 1
            self.task_created(c)
        parent.status = TaskStatus.DECOMPOSED
        parent.children = list(dict.fromkeys(parent.children + [c.id for c in children]))
        parent.latest_children = [c.id for c in children]
        parent.combine = combine
        self._refresh_statuses()
        self._record(EventKind.TASK_GRAPH_EXPANDED, "controller", rationale, task_id=parent_id,
                     chosen=[c.id for c in children],
                     outcome={"combine": combine, "dependencies": [[c.id, d] for c in children for d in c.dependencies]})

    def task_composed(self, task_id: str, entities: list[EntityRef], rationale: str = "") -> None:
        task = self.task(task_id)
        task.composed = list(entities)
        for e in entities:
            self.entity_grounded(e, task_id=task_id, rationale=rationale or "composed from its pieces")
        self._record(EventKind.SET_PRODUCED, "controller", rationale, task_id=task_id,
                     context={"name": task_id, "composed": True},
                     outcome={"size": len(entities), "sample": [e.id for e in entities][:6]})

    def task_done(self, task_id: str, rationale: str = "") -> None:
        task = self.task(task_id)
        task.status = TaskStatus.RESOLVED
        self._refresh_statuses()
        self._record(EventKind.TASK_RESOLVED, "controller", rationale, task_id=task_id,
                     outcome={"ready": self.ready()})

    def task_output_withdrawn(self, task_id: str, entity_ids: list[str],
                              rationale: str = "") -> None:
        self._withdraw(task_id, set(entity_ids))
        task = self.task(task_id)
        task.composed = [e for e in task.composed if e.id not in set(entity_ids)]
        self._record(
            EventKind.ENTITY_GROUNDED, "controller", rationale, task_id=task_id,
            outcome={"withdrawn": list(entity_ids)[:20]},
        )

    def _withdraw(self, task_id: str, only: set[str] | None = None) -> None:
        for entity_id, grounded in list(self.grounded_entities.items()):
            if not grounded.produced_by_task(task_id):
                continue
            if only is not None and entity_id not in only:
                continue
            others = [x for x in grounded.sources if x != task_id]
            if grounded.source == task_id and not others:
                del self.grounded_entities[entity_id]
            else:
                grounded.sources = others
                if grounded.source == task_id:
                    grounded.source = others[0]

    def task_failed(self, task_id: str, diagnosis: str = "",
                    rationale: str = "") -> None:
        task = self.task(task_id)
        task.status = TaskStatus.FAILED
        task.diagnosis = diagnosis
        self._refresh_statuses()
        self._record(
            EventKind.TASK_FAILED, "controller", rationale, task_id=task_id,
            outcome={"diagnosis": diagnosis, "frontier": self.frontier()},
        )

    def task_inputs_bound(self, task_id: str, role: str, entities: list[EntityRef],
                          anchors: list[EntityRef] | None = None,
                          rationale: str = "") -> None:
        task = self.task(task_id)
        task.bindings.roles[role] = list(entities)
        known = {e.id for e in task.grounded_inputs}
        task.grounded_inputs.extend(
            e for e in (entities if anchors is None else anchors) if e.id not in known
        )
        self._refresh_statuses()
        self._record(
            EventKind.TASK_CREATED, "controller", rationale or f"bound {role}",
            task_id=task_id,
            context={"role": role, "entities": [e.id for e in entities][:20]},
        )

    def entity_grounded(self, entity: EntityRef, task_id: str = "",
                        rationale: str = "") -> None:
        source = task_id or "initial_grounding"
        known = self.grounded_entities.get(entity.id)
        if known is not None:
            # A second producer of the same entity: both stand behind it.
            if source not in known.sources and source != known.source:
                known.sources.append(source)
            known.label = known.label or entity.label
            known.entity_type = known.entity_type or entity.entity_type
            grounded = known
        else:
            grounded = GroundedEntity(
                id=entity.id, label=entity.label, entity_type=entity.entity_type,
                source=source, sources=[source],
            )
        self.grounded_entities[entity.id] = grounded
        self._record(
            EventKind.ENTITY_GROUNDED, "query_agent", rationale,
            task_id=task_id or None,
            outcome={"entity_id": entity.id, "entity_type": entity.entity_type},
        )

    def evidence_added(self, records: list[EvidenceRecord],
                       rationale: str = "") -> None:
        self.evidence.extend(records)
        for found in records:
            for requirement_id in found.requirement_ids:
                requirement = self.requirement(requirement_id)
                if found.id not in requirement.supporting_evidence_ids:
                    requirement.supporting_evidence_ids.append(found.id)
        self._record(
            EventKind.EVIDENCE_ADDED, "query_agent", rationale,
            outcome={"evidence_ids": [found.id for found in records]},
        )

    def requirement_resolved(self, requirement_id: str,
                             rationale: str = "") -> None:
        requirement = self.requirement(requirement_id)
        requirement.status = RequirementStatus.RESOLVED
        self._record(
            EventKind.REQUIREMENT_RESOLVED, "sufficiency", rationale,
            requirement_id=requirement_id,
            outcome={"coverage": self.coverage()},
        )

    def requirement_blocked(self, requirement_id: str, task_id: str,
                            diagnosis: str = "", rationale: str = "") -> None:
        requirement = self.requirement(requirement_id)
        if task_id and task_id not in requirement.blocking_task_ids:
            requirement.blocking_task_ids.append(task_id)
        if diagnosis:
            requirement.diagnosis = diagnosis
        self._record(
            EventKind.REQUIREMENT_BLOCKED, "controller", rationale,
            requirement_id=requirement_id, task_id=task_id or None,
            outcome={"diagnosis": diagnosis, "status": str(requirement.status),
                     "coverage": self.coverage()},
        )

    def budget_consumed(self, tokens: int = 0, seconds: float = 0.0,
                        llm_calls: int = 0, kg_queries: int = 0,
                        rationale: str = "") -> None:
        for name, amount in (("tokens", tokens), ("seconds", seconds),
                             ("llm_calls", llm_calls), ("kg_queries", kg_queries)):
            if amount < 0:
                raise InvalidTransition(f"{name} spent cannot be negative")
        self.resources.tokens_spent += tokens
        self.resources.seconds_spent += seconds
        self.resources.llm_calls += llm_calls
        self.resources.kg_queries += kg_queries
        self._record(
            EventKind.BUDGET_CONSUMED, "runtime", rationale,
            outcome={"tokens": tokens, "seconds": seconds,
                     "llm_calls": llm_calls, "kg_queries": kg_queries,
                     "tokens_remaining": self.resources.tokens_remaining},
        )

    def schema_retrieved(self, entity_id: str, schema: LocalSchema,
                         rationale: str = "") -> None:
        self.schemas[entity_id] = schema
        self._record(
            EventKind.SCHEMA_RETRIEVED, "query_agent", rationale,
            context={"entity_id": entity_id},
            outcome={"relations": [e.predicate for e in schema.entries],
                     "routes": len(schema.paths)},
        )

    def judged(self, task_id: str, verdict: SufficiencyResult,
               rationale: str) -> None:
        if not rationale:
            raise InvalidTransition("a verdict must say on what basis it was reached")
        self.task(task_id)
        self.verdicts.setdefault(task_id, []).append(verdict)
        self._record(
            EventKind.LOCAL_COMPLETION_CHECKED, "sufficiency", rationale,
            task_id=task_id, chosen=verdict.sufficient,
            context={"basis": verdict.basis, "coverage": verdict.coverage},
            outcome={"sufficient": verdict.sufficient, "reason": verdict.reason,
                     "satisfied": list(verdict.satisfied_requirements),
                     "missing": list(verdict.missing_requirements)},
        )

    def set_produced(self, name: str, entities: list[EntityRef], kind: str = "",
                     produced_by: str = "", counts_as: str = "answer",
                     rationale: str = "") -> None:
        self.sets[name] = SetRecord(name=name, entities=list(entities), kind=kind,
                                    produced_by=produced_by, counts_as=counts_as)
        self._record(
            EventKind.SET_PRODUCED, "controller", rationale, task_id=produced_by or None,
            context={"name": name, "kind": kind, "counts_as": counts_as},
            outcome={"size": len(entities), "sample": [e.id for e in entities][:6]},
        )

    def plan_of(self, task_id: str) -> Plan | None:
        return self.plan

    def graph_of(self, task_id: str) -> Any:
        return self.task_graphs.get(task_id, self.hypergraph)

    def canonicalised(self, graph: Any, rationale: str = "") -> None:
        self.hypergraph = graph
        self._record(
            EventKind.CANONICALISED, "decomposer", rationale,
            outcome={"constraints": len(getattr(graph, "constraints", [])),
                     "variables": len(getattr(graph, "variables", []))},
        )

    def local(self, task_id: str) -> LocalTaskState:
        task = self.task(task_id)
        if task_id not in self.task_states:
            self.task_states[task_id] = LocalTaskState(
                task_id=task_id, parent_task_id=task.parent_id,
                requirement_ids=list(task.requirement_ids),
                contract=task.contract, status=task.status,
                dependencies=list(task.dependencies),
                grounded_input_ids=[e.id for e in task.grounded_inputs],
            )
        return self.task_states[task_id]

    def query_attempted(self, task_id: str, attempt: QueryAttempt,
                        rationale: str = "") -> None:
        local = self.local(task_id)
        local.attempts.append(attempt)
        local.local_evidence_ids.extend(attempt.evidence_ids)
        local.output_entity_ids.extend(attempt.grounded_entity_ids)

        chosen = None
        for candidate in attempt.candidates:
            if candidate.id == attempt.selected_candidate_id:
                chosen = candidate.strategy or candidate.id
        self._record(
            EventKind.QUERY_EXECUTED, "query_agent", rationale, task_id=task_id,
            context={"attempt": attempt.attempt,
                     "local_schema": attempt.local_schema},
            considered=[c.strategy or c.id for c in attempt.candidates],
            chosen=chosen,
            outcome={"success": attempt.success,
                     "result_count": attempt.result_count,
                     "failure_type": attempt.failure_type,
                     "diagnosis": attempt.diagnosis},
        )

    def view(self, purpose: str, task_id: str | None = None):
        build = views.BY_PURPOSE.get(purpose)
        if build is None:
            raise InvalidTransition(
                f"no view for {purpose!r}; there are "
                f"{', '.join(sorted(views.BY_PURPOSE))}")
        return build(self, task_id)

    def get_view(self, task_id: str | None = None) -> TaskView:
        return views.task_view(self, task_id)

    def sufficiency_view(self, target_id: str = ROOT_TASK) -> SufficiencyView:
        return views.sufficiency_view(self, target_id)

    def is_final(self, task_id: str) -> bool:
        task = self.tasks.get(task_id)
        return task is not None and task.parent_id is None

    def snapshot(self) -> KoraStateSnapshot:
        tasks = {}
        for task_id, task in self.tasks.items():
            local = self.local(task_id).model_copy(update={"status": task.status})
            tasks[task_id] = local

        requirements = {}
        for found in (self.contract.requirements if self.contract else []):
            requirements[found.id] = found

        return KoraStateSnapshot(
            run_id=self.run_id,
            question_id=self.question.question_id or None,
            question=self.question.text,
            root_entities=[
                self.grounded_entities[e.id]
                for e in self.question.root_entities
                if e.id in self.grounded_entities
            ],
            answer_contract=self.contract,
            requirements=requirements,
            grounded_entities=dict(self.grounded_entities),
            evidence=list(self.evidence),
            task_graph=TaskGraph(
                tasks=tasks,
                ready_frontier=self.frontier(),
                active_task_ids=[
                    t.id for t in self.tasks.values()
                    if t.status is TaskStatus.RUNNING
                ],
            ),
            budget=BudgetSnapshot.of(self.resources),
            incumbent_answer=self.incumbent,
            plan=self.plan,
            schemas=dict(self.schemas),
            verdicts={k: list(v) for k, v in self.verdicts.items()},
            sets=dict(self.sets),
            history=ExecutionHistory(events=list(self.history)),
            question_type=self.plan.shape if self.plan else "",
            coverage=self.coverage(),
            stop_reason=self.stop_reason,
            sufficient=self.sufficient,
            metadata=dict(self.metadata),
        )
