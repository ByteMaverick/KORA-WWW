"""The question's structure: one call, five checks, a shape and a recipe derived in code."""

from __future__ import annotations

import re
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from kora.provenance.models import (
    Answer, AnswerContract, Constraint, EntityRef, Need, Plan, PlannedRequirement,
    Recipe, RequirementState, Spend, Variable, answer_entity_types,
)
from kora.modules.decomposition.structural import graph as structure
from kora.prompts import PROMPTS, Prompt
from kora.tools.llm import LLMError, LLMProvider, extract_json

DEFAULT_PROMPT = "planning.constraints"
EXAMPLES = "planning.shapes"
UNCARRIED = "is a name the question states that no constraint carries"
PROJECTIONS = ("identity", "count", "argmax", "argmin")
COUNTS = ("one", "few", "many")
TYPES = ("entity", "value", "boolean")

__all__ = ["Plan", "PlanRequest", "Planner", "PlannerError", "recipe_of", "shape_of", "to_graph"]


class PlannerError(RuntimeError):
    def __init__(self, message: str, spend: Spend | None = None):
        super().__init__(message)
        self.spend = spend or Spend()


class PlanRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str
    grounded_entities: list[EntityRef] = Field(default_factory=list)
    anchor_labels: dict[str, str] = Field(default_factory=dict)

    def render_entities(self) -> str:
        return "\n".join(f"- {e.label or e.id}" for e in self.grounded_entities) or "(none)"

    @property
    def labels(self) -> set[str]:
        return {e.label for e in self.grounded_entities if e.label}


class Planner:
    def __init__(self, llm: LLMProvider, prompt: Prompt | None = None):
        self.llm = llm
        self.prompt = prompt or PROMPTS.get(DEFAULT_PROMPT)

    def plan(self, request: PlanRequest) -> Plan:
        spend, complaint, problems = Spend(), "", []
        for attempt in (1, 2):
            prompt = self.render(request, complaint)
            response = self.llm.generate(prompt, system=self.prompt.system)
            spend.tokens += response.total_tokens
            spend.seconds += response.latency_seconds
            spend.llm_calls += 1
            try:
                plan = repair(parse(response.text, request))
                problems = check(plan, request)
            except PlannerError as exc:
                problems = [str(exc)]
            if not problems:
                return derived(plan, request, spend, self.prompt.id)
            if attempt == 2 and all(p.endswith(SOFT) for p in problems):
                return derived(soften(plan), request, spend, self.prompt.id)
            complaint = "WHAT WAS WRONG WITH THE LAST REPLY\n" + "\n".join(f"- {p}" for p in problems)
        raise PlannerError("; ".join(problems), spend)

    def render(self, request: PlanRequest, complaint: str) -> str:
        return self.prompt.render(question=request.question, entities=request.render_entities(),
                                  memory="(nothing relevant)",
                                  examples=PROMPTS.get(EXAMPLES).rendered(), learned="",
                                  complaint=complaint)

    def run(self, state) -> Plan:
        plan = self.plan(state.view("planning"))
        state.budget_consumed(tokens=plan.spend.tokens, seconds=plan.spend.seconds,
                              llm_calls=plan.spend.llm_calls)
        state.planned(plan, contract(state, plan), rationale=plan.rationale or "the question's structure")
        return plan


def contract(state, plan: Plan) -> AnswerContract:
    accepted = (["value"] if plan.answer.projection == "count"
                else answer_entity_types(state.question.text, plan.answer.kind or None))
    return AnswerContract(
        answer_type=plan.answer.type, answer_projection=plan.answer.projection,
        answer_count=plan.answer.count,
        answer_entity_type=accepted[0] if accepted else None, accepted_entity_types=accepted,
        requirements=[RequirementState(id=r.id, description=r.goal, critical=r.critical)
                      for r in plan.requirements],
        constraints=list(plan.constraints), planner_summary=plan.rationale or None)


def parse(text: str, request: PlanRequest) -> Plan:
    try:
        payload = extract_json(text)
    except LLMError as exc:
        raise PlannerError(f"the reply is not JSON: {exc}") from None
    if not isinstance(payload, dict):
        raise PlannerError("the reply is not an object")
    variables, constraints, answer = payload.get("variables"), payload.get("constraints"), payload.get("answer")
    if not isinstance(variables, dict) or not isinstance(constraints, list) or not isinstance(answer, dict):
        raise PlannerError("variables must be an object, constraints a list and answer an object")
    return Plan(
        goal=request.question,
        variables=[Variable(name=str(k), kind=str(v or "")) for k, v in variables.items()],
        constraints=[Constraint(id=str(c.get("id") or f"c{n}"), args=[str(a) for a in c.get("args") or []],
                                says=str(c.get("says") or ""), op=_text(c.get("op")).lower(),
                                cmp=_text(c.get("cmp")).lower(), value=_text(c.get("value")),
                                about=_text(c.get("about")), over=_text(c.get("over")))
                     for n, c in enumerate(constraints, 1) if isinstance(c, dict)],
        answer=Answer(variable=_text(answer.get("variable")), kind=_text(answer.get("kind")),
                      projection=_text(answer.get("projection")) or "identity", key=_text(answer.get("key")),
                      count=_text(answer.get("count")) or "few", type=_text(answer.get("type")) or "entity"),
        rationale=str(payload.get("rationale") or ""))


def repair(plan: Plan) -> Plan:
    carried = [a for c in plan.constraints for a in c.args]
    for c in plan.constraints:
        for name in dict.fromkeys(_braced(c.says)):
            if not any(name in a or a in name for a in carried):
                c.args.append(name)
                carried.append(name)
    used = {a for c in plan.constraints for a in c.args}
    kept = [v for v in plan.variables if v.name in used or v.name == plan.answer.variable]
    if len(kept) != len(plan.variables):
        plan = plan.model_copy(update={"variables": kept})
    return plan


SOFT = (UNCARRIED, "needs a key to order by", "fewer restrictions than variables")


def soften(plan: Plan) -> Plan:
    """A tolerated plan, made runnable: an ordering with no key is dropped, the constraints kept."""
    a = plan.answer
    if a.projection in ("argmax", "argmin") and not a.key:
        plan = plan.model_copy(update={"answer": a.model_copy(update={"projection": "identity"})})
    return plan


def _text(value) -> str:
    text = str(value).strip() if value is not None else ""
    return "" if text.lower() in ("null", "none") else text


def constants_of(plan: Plan, request: PlanRequest) -> dict[str, str]:
    labels = {label.lower() for label in request.labels}
    question = request.question.lower()
    found: dict[str, str] = {}
    for c in plan.constraints:
        for arg in c.args:
            if structure.is_variable(arg):
                continue
            low = arg.lower()
            if low in labels or _mentions(arg, labels):
                found[arg] = "grounded"
            elif low.replace(",", "") in question.replace(",", ""):
                found[arg] = "literal"
            else:
                found[arg] = ""
    return found


def check(plan: Plan, request: PlanRequest) -> list[str]:
    problems: list[str] = []
    names = {v.name for v in plan.variables}
    for v in plan.variables:
        problems += _use_problems(plan, v.name)
    for arg, kind in constants_of(plan, request).items():
        if not kind:
            problems.append(f"{arg!r} is neither grounded nor in the question")
    problems += _answer_problems(plan, names)
    used = {a.lower() for c in plan.constraints for a in c.args}
    for label in sorted(label for label in request.labels if _stated(label, request.question)):
        if label.lower() not in used and not any(_mentions(a, {label.lower()}) for a in used):
            problems.append(f"{label!r} {UNCARRIED}")
    for c in plan.constraints:
        if _QUERY.search(_unbraced(c.says)):
            problems.append(f"{c.id} says a query or a relation name: say the restriction in the question's words")
    for v in plan.variables:
        if _QUERY.search(v.kind):
            problems.append(f"the kind of {v.name} is a query or a relation name")
    restrictions = sum(max(1, sum(structure.is_variable(a) for a in set(c.args)) - 1) for c in plan.constraints)
    if restrictions < len(plan.variables):
        problems.append(f"{len(plan.variables)} variables and {restrictions} restrictions: "
                        f"fewer restrictions than variables")
    return problems


def _use_problems(plan: Plan, name: str) -> list[str]:
    over = [c for c in plan.constraints if name in c.args]
    if not over:
        return [f"{name} is a variable no constraint uses"]
    joins = len(over) >= 2 or any(structure.is_variable(a) and a != name for c in over for a in c.args)
    if name != plan.answer.variable and not joins:
        return [f"{name} is an intermediate that joins nothing: it must be in two constraints "
                f"or share one with another variable"]
    return []


def _answer_problems(plan: Plan, names: set[str]) -> list[str]:
    a, problems = plan.answer, []
    if a.type == "boolean":
        if a.variable:
            problems.append("a boolean answer has no answer variable")
    elif a.variable not in names or not any(a.variable in c.args for c in plan.constraints):
        problems.append(f"the answer variable {a.variable or '(none)'} is not a declared variable in a constraint")
    if a.projection not in PROJECTIONS:
        problems.append(f"projection {a.projection!r} is not one of {', '.join(PROJECTIONS)}")
    if a.count not in COUNTS:
        problems.append(f"count {a.count!r} is not one of {', '.join(COUNTS)}")
    if a.type not in TYPES:
        problems.append(f"type {a.type!r} is not one of {', '.join(TYPES)}")
    if a.projection in ("argmax", "argmin") and not a.key:
        problems.append(f"{a.projection} needs a key to order by")
    return problems


def derived(plan: Plan, request: PlanRequest, spend: Spend, prompt_id: str) -> Plan:
    plan = operations(plan.model_copy(update={"constants": constants_of(plan, request)}))
    return plan.model_copy(update={
        "shape": shape_of(plan), "recipe": recipe_of(plan), "spend": spend, "prompt_id": prompt_id,
        "requirements": [PlannedRequirement(id=c.id, goal=c.says.replace("{", "").replace("}", ""))
                         for c in plan.constraints]})


OPERATIONS = ("filter", "argmax", "argmin", "count", "verify")
COMPARISONS = ("lt", "le", "gt", "ge", "eq", "ne")
NAMED = "named"
DATE_FORMS = ("%B %d, %Y", "%B %d %Y", "%d %B %Y", "%B %Y", "%Y-%m-%d", "%m/%d/%Y", "%Y")


def operations(plan: Plan) -> Plan:
    """Each operation checked in code; one that fails keeps its words as a relation and loses its operation."""
    kept, refused, carried = [], [], set()
    for c in plan.constraints:
        if c.op:
            c = c.model_copy(update={"value": normalised(c.value),
                                     "over": NAMED if c.over.lower() == NAMED else c.over})
            if problem := _operation_problem(c, plan, carried):
                refused.append(f"{c.id}: {problem}")
            else:
                carried.add(c.op)
                kept.append(c)
                continue
        kept.append(c.model_copy(update={"op": "", "cmp": "", "value": "", "about": "", "over": ""}))
    return plan.model_copy(update={"constraints": kept, "refused_operations": refused})


def _operation_problem(c: Constraint, plan: Plan, carried: set[str]) -> str:
    if c.op not in OPERATIONS:
        return f"op {c.op!r} is not one of {', '.join(OPERATIONS)}"
    compares = c.op in ("filter", "verify")
    if not compares and (c.cmp or c.value):
        return "cmp and value go only with filter or verify"
    if compares and (c.cmp not in COMPARISONS or not c.value):
        return f"{c.op} needs cmp (one of {', '.join(COMPARISONS)}) and the value it compares with"
    if c.cmp in ("lt", "le", "gt", "ge") and not _orderable(c.value):
        return f"{c.value!r} is not a number or a date to compare by {c.cmp}"
    if c.op != "count" and not c.about:
        return f"{c.op} needs about: the property, in the question's words"
    named = [a for a in c.args if plan.constants.get(a) == "grounded"]
    if c.over == NAMED and not named:
        return "it runs over the named things, and none of its arguments is a name the question states"
    if c.over != NAMED:
        if c.over not in c.args or c.over not in {v.name for v in plan.variables}:
            return f"over {c.over!r} is neither a variable of this constraint nor {NAMED!r}"
        if named:
            return f"it relates {c.over} to {named[0]}: that is a relation, asked of the graph"
        if not any(c.over in o.args and not o.op for o in plan.constraints if o.id != c.id):
            return f"no relation produces {c.over}, so there is no set to run over"
    if c.op in ("argmax", "argmin", "count"):
        if c.op != plan.answer.projection:
            return f"{c.op} disagrees with the answer's projection {plan.answer.projection!r}"
        if c.op in carried:
            return f"another constraint already carries the {c.op}"
    if c.op == "verify" and plan.answer.type != "boolean":
        return "verify is for a yes/no question"
    return ""


def normalised(value: str) -> str:
    """A number without its separators, a date as ISO at the precision it was written; else as written."""
    value = value.strip()
    if re.fullmatch(r"-?[\d,]+(\.\d+)?", value):
        return value.replace(",", "")
    for form in DATE_FORMS:
        try:
            parsed = datetime.strptime(value, form)
        except ValueError:
            continue
        return parsed.strftime("%Y-%m-%d" if "%d" in form else "%Y-%m" if "%m" in form or "%B" in form else "%Y")
    return value


def _orderable(value: str) -> bool:
    return bool(re.fullmatch(r"-?\d+(\.\d+)?", value) or re.match(r"^-?\d{4}(-\d{2}){0,2}$", value))


def shape_of(plan: Plan) -> str:
    a = plan.answer
    if a.type == "boolean":
        return "boolean"
    if a.projection == "count":
        return "aggregation"
    if a.projection in ("argmax", "argmin"):
        return "superlative"
    if any(kind == "literal" and any(ch.isdigit() for ch in text) for text, kind in plan.constants.items()):
        return "comparative"
    if any(v.name != a.variable for v in plan.variables):
        return "composition"
    if sum(a.variable in c.args for c in plan.constraints) >= 2:
        return "conjunction"
    return "direct"


def recipe_of(plan: Plan) -> Recipe:
    return Recipe(needs=_needs(plan), closes=_closes(plan))


def _closes(plan: Plan) -> list[str]:
    a = plan.answer
    if a.type == "boolean":
        return ["verify"]
    if a.projection == "count":
        return ["count"]
    if a.projection in ("argmax", "argmin"):
        return [f"{a.projection}({a.key})"]
    literals = [f"filter({arg})" for c in plan.constraints if a.variable in c.args
                for arg in c.args if plan.constants.get(arg) == "literal"]
    if literals:
        return literals
    if shape_of(plan) == "conjunction":
        return ["intersect"]
    return ["last"]


def _needs(plan: Plan) -> list[Need]:
    placed: list[str] = []
    needs: list[Need] = []
    while len(placed) < len(plan.variables):
        layer = [v for v in plan.variables if v.name not in placed and _reached(plan, v.name, placed)]
        layer = layer or [v for v in plan.variables if v.name not in placed][:1]
        for v in layer:
            over = [c for c in plan.constraints if v.name in c.args]
            needs.append(Need(variable=v.name, kind=v.kind, from_constraints=[c.id for c in over],
                              given=[p for p in placed if any(p in c.args for c in over)]))
        placed += [v.name for v in layer]
    return needs


def _reached(plan: Plan, name: str, placed: list[str]) -> bool:
    return any(name in c.args and any(a != name and (a in placed or not structure.is_variable(a)) for a in c.args)
               for c in plan.constraints)


def to_graph(plan: Plan, grounded: dict[str, EntityRef]) -> structure.ConstraintGraph:
    lowered = {label.lower(): e for label, e in grounded.items()}
    bound = {text: e for text, kind in plan.constants.items() if kind == "grounded"
             and (e := _entity(text, lowered)) is not None}
    return structure.ConstraintGraph(
        variables=[v.name for v in plan.variables],
        constraints=[structure.Constraint(id=c.id, source_requirement=c.id, arguments=list(c.args), template=c.says)
                     for c in plan.constraints],
        grounded=bound,
        literals=[text for text, kind in plan.constants.items() if kind == "literal"],
        answer=structure.AnswerSpec(projection=plan.answer.projection, variable=plan.answer.variable,
                                    key=plan.answer.key))


def _entity(text: str, lowered: dict[str, EntityRef]) -> EntityRef | None:
    low = text.lower()
    return lowered.get(low) or next((e for label, e in lowered.items() if label in low or low in label), None)


_QUERY = re.compile(r"\bSELECT\b|\bMATCH\b|\bWHERE\s*\{|\?\w+|\b[A-Z]{3,}_[A-Z_]{3,}\b|\bm\.[0-9a-z_]{3,}\b")


_FILLER = {"the", "of", "and", "for", "in", "on", "a", "an", "to", "by", "at", "from", "with"}


def _stated(label: str, question: str) -> bool:
    if re.fullmatch(r"[mg]\.[0-9a-z_]+", label):
        return False
    said = set(re.findall(r"[a-z0-9]+", question.lower()))
    words = [w for w in re.findall(r"[a-z0-9]+", label.lower()) if w not in _FILLER]
    initials = "".join(w[0] for w in words)
    return any(len(w) >= 3 and w in said for w in words) or (len(initials) >= 2 and initials in said)


def _mentions(item: str, labels: set[str]) -> bool:
    low = item.lower()
    return any(label in low for label in labels)


def _unbraced(says: str) -> str:
    return "".join(part.split("}", 1)[-1] for part in says.split("{"))


def _braced(says: str) -> list[str]:
    return [part.split("}", 1)[0] for part in says.split("{")[1:] if "}" in part]
