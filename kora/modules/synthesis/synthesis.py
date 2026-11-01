from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from kora.provenance.models import Answer, EntityRef, RequirementStatus
from kora.modules.reduction.models import NO, YES

__all__ = ["AnswerDraft", "Step", "apply", "synthesise"]

ROOT = "T0"


class Step(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str
    goal: str
    anchors: list[str] = Field(default_factory=list)
    produced: list[str] = Field(default_factory=list)
    settled: list[str] = Field(default_factory=list)
    after: list[str] = Field(default_factory=list)


class AnswerDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entities: list[EntityRef] = Field(default_factory=list)
    names: list[str] = Field(default_factory=list)
    text: str = ""
    projection: str = "identity"
    kind: str = ""
    count: str = "few"
    established: list[str] = Field(default_factory=list)
    unmet: list[str] = Field(default_factory=list)
    derivation: list[Step] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)


def synthesise(state) -> AnswerDraft:
    answer = state.plan.answer if state.plan else Answer()
    boolean = answer.type == "boolean"
    entities = [] if boolean else apply(answer.projection, state.produced_by(ROOT))
    names = [e.label or e.id for e in entities]
    resolved = established(state)
    return AnswerDraft(
        entities=entities,
        names=names,
        text=truth(state) if boolean else ", ".join(names),
        projection=answer.projection,
        kind=answer.kind,
        count=answer.count,
        established=resolved,
        unmet=[r.id for r in requirements(state) if r.id not in resolved],
        derivation=derivation(state),
        evidence_ids=[r.id for r in state.evidence if r.source_task_id in state.tasks],
    )


def truth(state) -> str:
    """A verify's yes or no, read off the graph's values, when the root holds one; else the judge's verdicts."""
    held = state.produced_by(ROOT)
    if len(held) == 1 and held[0].id in (YES.id, NO.id):
        return YES.label if held[0].id == YES.id else NO.label
    verdicts = state.verdicts_for(ROOT)
    claims = [c.verdict for c in verdicts[-1].claims] if verdicts else []
    if "NOT" in claims:
        return "no"
    if claims and set(claims) == {"ESTABLISHED"} and (state.produced_by(ROOT) or not state.plan.variables):
        return "yes"
    return "unknown"


def apply(projection: str, entities: list[EntityRef]) -> list[EntityRef]:
    row = len(entities) == 1 and entities[0].entity_type == "value" and entities[0].id.strip().isdigit()
    if projection != "count" or row:
        return entities
    return [EntityRef(id=str(len(entities)), label=str(len(entities)), entity_type="value")]


def requirements(state) -> list:
    return state.contract.requirements if state.contract else []


def established(state) -> list[str]:
    return [r.id for r in requirements(state) if r.status is RequirementStatus.RESOLVED]


def derivation(state) -> list[Step]:
    return [step(state, task) for task in state.tasks.values()
            if state.produced_by(task.id) or settled(state, task)]


def step(state, task) -> Step:
    return Step(task_id=task.id, goal=task.goal,
                anchors=[e.label or e.id for e in task.grounded_inputs],
                produced=[e.label or e.id for e in state.produced_by(task.id)][:20],
                settled=settled(state, task), after=list(task.dependencies))


def settled(state, task) -> list[str]:
    return [r for r in task.satisfies if r in established(state)]
