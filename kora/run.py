"""One question through the Controller, with the chosen stop rule, scored; and the validation rules that fired."""

from __future__ import annotations

import os
import re
import time
import traceback
from collections import Counter

from kora.agents.query_agent import QueryAgent
from kora.provenance.models import EntityRef
from kora.provenance.state import KoraState
from kora.modules.controller import build_controller
from kora.modules.controller.models import Budget
from kora.modules.controller.strategies import check_arm
from kora.modules.query.judge import KnowledgeLeafJudge, LeafJudge
from kora.modules.query.models import Outcome, Verdict
from kora.modules.sufficiency import JudgeSufficiency, KnowledgeSufficiency, OracleSufficiency, matches
from kora.prompts import PROMPTS
from kora.tools.dialect import count_variable, output_variables
from kora.tools.entity import EntityResolver
from kora.tools.kg import KGError

METAQA = {"uri": "NEO4J_METAQA_URI", "db": "metaqa", "id": "id",
          "label": "Entity", "prominence": "", "hubs": set()}
SPARQL = {"uri": "FREEBASE_SPARQL_URL", "id": "mid", "label": "", "hubs": set(),
          "sparql": True}
SHAPES = {"metaqa": METAQA, "freebase-sparql": SPARQL}


class GraphUnavailable(RuntimeError):
    """The graph did not answer."""


_GRAPH_DOWN = re.compile(r"Couldn't connect|ServiceUnavailable|defunct connection|Connection refused|"
                         r"Unable to retrieve routing information", re.IGNORECASE)


def graph_went_away(row: dict) -> str | None:
    """The first line of a result row that says the graph was unreachable."""
    texts = [str(row.get("error") or "")] + [str(line) for line in row.get("log") or []]
    for task in row.get("graph") or []:
        texts.append(str(task.get("diagnosis") or ""))
    for text in texts:
        if _GRAPH_DOWN.search(text):
            return text[:200]
    return None


def wanted(gold: list[str], projection: str) -> int | None:
    if projection != "count" or len(gold) != 1 or not gold[0].strip().isdigit():
        return None
    return int(gold[0].strip())


def is_the_count(values: list[str], count: int, counted: bool = False) -> bool:
    distinct = {v for v in values if v}
    if len(distinct) == count:
        return True
    return counted and len(distinct) == 1 and next(iter(distinct)).strip() == str(count)


class OracleJudge:
    """Local sufficiency answered by the gold set instead of by a model."""

    def __init__(self, gold: list[str], resolve, threshold: float = 0.5,
                 precision: float = 0.5, resolve_beyond=None):
        self.gold = [g for g in gold if g]
        self.resolve = resolve
        self.resolve_beyond = resolve_beyond
        self.threshold = threshold
        self.precision = precision

    def recall(self, values: list[str], counted: bool = False, projection: str = "identity") -> float:
        if not self.gold:
            return 0.0
        count = wanted(self.gold, projection)
        if count is not None:
            return float(is_the_count(values, count, counted))
        named = list(self.resolve(values))
        if self.resolve_beyond is not None:
            named += list(self.resolve_beyond(values))
        got = {x for x in values + named if x}
        found = sum(1 for g in self.gold if any(matches(g, x) for x in got))
        return found / len(self.gold)

    def precision_of(self, values: list[str], counted: bool = False,
                     projection: str = "identity") -> float:
        if not values or not self.gold:
            return 0.0
        count = wanted(self.gold, projection)
        if count is not None:
            return float(is_the_count(values, count, counted))
        matched = 0
        for value in values[:50]:
            names = [x for x in [value] + list(self.resolve([value])) if x]
            if any(matches(g, x) for x in names for g in self.gold):
                matched += 1
        return matched / min(len(values), 50)

    def verdicts(self, ctx, results):
        return {c.id: self.analyze(e, c, ctx, None) for c, e in results}

    def analyze(self, execution, candidate, context, _usage) -> Verdict:
        return self.measured(execution, candidate, context)[0]

    def measured(self, execution, candidate, context) -> tuple[Verdict, float, float | None]:
        available = output_variables(execution.query)
        variable = candidate.output_variable if candidate.output_variable in available \
            else (available[0] if available else "")
        values = execution.column(variable) if variable else []
        counted = count_variable(execution.query) is not None
        projection = context.task.projection if context is not None else "identity"
        score = self.recall(values, counted, projection)
        tight = self.precision_of(values, counted, projection) if score >= self.threshold else None
        if tight is not None and tight < self.precision:
            return Verdict(
                outcome=Outcome.INSUFFICIENT_RESULT, answer_variable=variable, entities=values,
                reason=(f"this returned {len(values)} value(s) and reaches what the task needs, "
                        f"but most of them are not it: the set is too broad -- add the "
                        f"question's other constraint or a :type"),
            ), score, tight
        if score >= self.threshold:
            return Verdict(outcome=Outcome.SUCCESS, answer_variable=variable, entities=values,
                           reason=f"recall {score:.2f} of what this task needed"), score, tight
        return Verdict(
            outcome=Outcome.EMPTY_RESULT if not values else Outcome.INSUFFICIENT_RESULT,
            answer_variable=variable, entities=values,
            reason=(f"this returned {len(values)} value(s) but only {score:.0%} of "
                    f"what the task needed; the relation is probably wrong"),
        ), score, tight


class SplitJudge:
    """The oracle on the root task; the model on everything else."""

    def __init__(self, oracle: OracleJudge, model):
        self.oracle, self.model = oracle, model

    def verdicts(self, ctx, results):
        final = ctx.task.task_id == "T0" or ctx.task.final
        return (self.oracle if final else self.model).verdicts(ctx, results)


def _root_judge() -> str:
    from kora.app import root_judge
    return root_judge()


def _oracle_rule() -> str:
    from kora.app import oracle_rule
    return oracle_rule()


def judge_arm(judge, oracle):
    if _root_judge() == "knowledge":   # the knowledge judge decides every candidate set inside the query agent
        return KnowledgeLeafJudge(judge.llm)
    if _root_judge() == "judge":
        return judge
    return SplitJudge(oracle, judge)


def named(value: str, gold: list[str]) -> bool:
    """Whether a returned id or label is one of the gold answers (`sufficiency.matches`)."""
    return any(matches(g, value) for g in gold if g)


def field_hit(ids: list[str], labels: list[str], gold: list[str]) -> bool:
    return any(named(x, gold) for x in labels + ids if x)


def first_is_gold(entities: list, gold: list[str]) -> bool:
    """Whether the first entity the run stands behind is a gold answer."""
    if not entities:
        return False
    first = entities[0]
    return any(named(v, gold) for v in (first.label, first.id) if v)


def answers(state: KoraState, verdict) -> dict[str, list[str]]:
    """The answer twice, first fifty, label or id: as scored, and as Kora would hand it to a user."""
    projected = state.sufficiency_view("T0").current_output
    return {"answer_order": [e.label or e.id for e in verdict.answer_entities][:50],
            "projection": [e.label or e.id for e in projected][:50]}


def properties(state: KoraState) -> dict[str, int]:
    """The operates' property choices: model calls, and how many fell back."""
    chosen = state.metadata.get("property") or []
    return {"property_calls": sum(1 for c in chosen if c["called"]),
            "property_fallbacks": sum(1 for c in chosen if c["fallback"])}


def precision_f1(ids: list[str], labels: list[str], gold: list[str],
                 recall: float) -> tuple[float, float]:
    """Precision of what the root stands behind, and F1 with the oracle's recall."""
    if not ids or not gold:
        return 0.0, 0.0
    have = [x for x in labels + ids if x]
    matched = sum(1 for x in have if named(x, gold))
    precision = min(matched, len(ids)) / len(ids)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, f1


class Harness:
    """The graph, the model, and one call per question."""

    def __init__(self, graph: str, *, budget: Budget, judge: bool = True, threshold: float = 0.5):
        from neo4j import GraphDatabase

        from kora.app import clock_fallback
        from kora.app import property_choice as property_setting
        from kora.tools.embed import Embedder
        from kora.tools.neo4j import Neo4jKG
        from kora.tools.llm import OpenAILLM

        self.shape = SHAPES[graph]
        shape = self.shape
        self.sparql = bool(shape.get("sparql"))
        if self.sparql:
            from kora.tools.sparql import FreebaseSPARQL
            self.kg = FreebaseSPARQL(os.environ[shape["uri"]])
            self.driver = None
        else:
            self.kg = Neo4jKG(uri=os.environ[shape["uri"]], database=shape["db"],
                              id_property=shape["id"], anchor_label=shape["label"],
                              hub_relations=shape["hubs"],
                              prominence_expr=shape["prominence"], schema_name=shape["db"])
            self.driver = GraphDatabase.driver(
                os.environ[shape["uri"]],
                auth=(os.environ.get("NEO4J_USER", "neo4j"),
                      os.environ.get("NEO4J_PASSWORD", "korakora")),
                notifications_min_severity="OFF")
        self.database = None if self.sparql else shape["db"]
        if self.driver is not None and not self._has_database(self.database):
            self.database = None
            self.kg = Neo4jKG(uri=os.environ[shape["uri"]], database=None,
                              id_property=shape["id"], anchor_label=shape["label"],
                              hub_relations=shape["hubs"], prominence_expr=shape["prominence"],
                              schema_name=shape["db"])
        self.llm = OpenAILLM(model=os.environ.get("KORA_LLM_MODEL", "gpt-4o-mini"))
        self.rounds = int(os.environ.get("KORA_ROUNDS", "2"))
        self.judge_llm = self.llm
        self.embedder = Embedder()
        self.check_graph()
        self.measure_judge = judge
        self.arm = check_arm(os.environ.get("KORA_DECOMPOSER", "structural"))
        self.property_choice = property_setting()
        self.fallback = clock_fallback()
        from kora.modules.sufficiency import estimator
        estimator.check()
        self.threshold = threshold
        self.budget = budget
        self.resolver = EntityResolver(self.kg, id_property=shape["id"],
                                       anchor_label=shape["label"],
                                       alias_properties=shape.get("aliases", ()))
        from kora.tools.ner import RecognizedAnchors
        from kora.modules.decomposition.decompose import StatedNames
        self.stated = StatedNames(RecognizedAnchors(self.llm, self.resolver))

    def _has_database(self, name: str | None) -> bool:
        try:
            with self.driver.session(database=name) as session:
                session.run("RETURN 1 AS ok").single()
            return True
        except Exception as error:  # noqa: BLE001
            return "DatabaseNotFound" not in str(error)

    def check_graph(self) -> None:
        try:
            if self.sparql:
                if not self.kg.health():
                    raise KGError("the endpoint answered nothing")
                return
            with self.driver.session(database=self.database) as session:
                session.run("RETURN 1 AS ok").single()
        except Exception as error:
            raise GraphUnavailable(f"the graph at {os.environ.get(self.shape['uri'])} did not answer: "
                                   f"{type(error).__name__}: {str(error)[:160]}") from error

    def names(self, ids: list[str]) -> list[str]:
        if not ids:
            return []
        if self.sparql:
            if hasattr(self.kg, "labels"):
                found = self.kg.labels(ids[:self.NAMED_AT_MOST])
                return [n for n in found.values() if n]
            return [n for n in (self.kg.label(i) for i in ids[:self.NAMED_FIRST]) if n]
        node = f"(n:{self.shape['label']})" if self.shape["label"] else "(n)"
        with self.driver.session(database=self.database) as session:
            rows = session.run(f"MATCH {node} WHERE n.{self.shape['id']} IN $m "
                               f"RETURN n.name AS name, n.symbol AS symbol", m=ids).data()
        # A node's symbol is one of its names (`SLCO1A2` beside the gene's full name).
        return [n for r in rows for n in (r["name"], r["symbol"]) if n]

    NAMED_FIRST = 50
    NAMED_AT_MOST = 500

    def names_beyond(self, ids: list[str]) -> list[str]:
        rest = list(ids[self.NAMED_FIRST:self.NAMED_AT_MOST])
        if not rest or not self.sparql or not hasattr(self.kg, "labels"):
            return []
        return [n for n in self.kg.labels(rest).values() if n]

    def oracle(self, item: dict) -> OracleJudge:
        return OracleJudge(item["gold"], self.names, self.threshold, resolve_beyond=self.names_beyond)

    def close(self) -> None:
        if self.driver is not None:
            self.driver.close()

    def run(self, item: dict, index: int = 0) -> dict:
        started = time.time()
        before_calls = getattr(self.kg, "calls", 0)
        before_embed = (getattr(self.embedder, "calls", 0), getattr(self.embedder, "tokens", 0))
        if not item["anchor"]:
            return {"i": index, "q": item["q"], "gold": item["gold"], "hit": False, "id": item.get("id"),
                    "error": "topic entity not in the dump", "log": []}
        roots = [EntityRef(id=entity_id, label=label) for entity_id, label in item.get("anchors") or [item["anchor"]]]
        state = KoraState.begin(item["q"], question_id=item.get("id") or "", root_entities=roots)
        oracle = self.oracle(item)
        judge = JudgeSufficiency(self.judge_llm)
        leaf = judge_arm(LeafJudge(self.judge_llm), oracle)
        if _root_judge() == "judge":
            sufficiency = judge
        elif _root_judge() == "knowledge":
            sufficiency = KnowledgeSufficiency(self.judge_llm)
        else:
            sufficiency = OracleSufficiency(item["gold"], self.names, self.threshold, children=judge,
                                            **({"f1": 0.0, "stop_on_hit": False} if _oracle_rule() == "strict" else {}))
        agent = QueryAgent(self.llm, self.kg, self.embedder, judge=leaf)
        agent.rounds = self.rounds
        controller = build_controller(agent, self.llm, self.kg, self.arm, sufficiency, self.budget,
                                      names=self.stated, resolver=self.resolver,
                                      property_choice=self.property_choice, fallback=self.fallback)
        try:
            outcome = controller.run(state)
            elapsed = round(time.time() - started, 1)
        except Exception as error:  # a run must survive a bad question
            failed = {"i": index, "q": item["q"], "gold": item["gold"], "hit": False, "id": item.get("id"),
                      "error": f"{type(error).__name__}: {str(error)[:200]}",
                      # Where it broke: "IndexError: list index out of range" names no line.
                      "log": traceback.format_exc().splitlines()[-12:]}
            return failed

        scorer = OracleSufficiency(item["gold"], self.names, self.threshold, children=judge)
        verdict = scorer.check(state, "T0")
        ids = [e.id for e in verdict.answer_entities]
        labels = self.names(ids)
        got = labels + ids
        recall = verdict.recall or 0.0
        precision, f1 = precision_f1(ids, labels, item["gold"], recall)
        oracle_hit = verdict.sufficient
        hit = field_hit(ids, labels, item["gold"])
        at_1 = first_is_gold(verdict.answer_entities, item["gold"])
        judging = time.time()
        judged = judge.check(state, "T0") if self.measure_judge else None
        judge_seconds = round(time.time() - judging, 1)
        tasks = state.tasks
        row = {
            "i": index, "q": item["q"], "gold": item["gold"], "hit": hit, "oracle_hit": oracle_hit,
            "hops": item.get("hops"), "id": item.get("id"),
            "roots": [e.label or e.id for e in roots],
            "status": outcome.status, "stop_reason": outcome.stop_reason,
            "phase": outcome.phase_at_finish,
            "recall": round(recall, 2), "precision": round(precision, 2), "f1": round(f1, 2),
            "first_is_gold": at_1,
            # Enough to rescore the run offline; fifty is what `precision_of` reads.
            "got": got,
            "answer_ids": ids,
            **answers(state, verdict),
            "sufficiency": verdict.reason,
            "gold_judge": None,
            # the first pages the model was shown, so a miss can be read against them
            "pages": ([{"anchor": s.entity_id, "relations": len(s.entries)}
                       for s in list(state.schemas.values())[:2]] if not oracle_hit else []),
            "judge": (None if judged is None else
                      {"sufficient": judged.sufficient, "reason": judged.reason[:200],
                       "basis": judged.basis}),
            "prompts": PROMPTS.hashes(),
            "rounds": outcome.rounds, "dispatches": outcome.dispatches,
            "validation": rules_fired(state),
            "decompositions": outcome.decompositions, **properties(state), "tasks": len(tasks),
            "llm_calls": state.resources.llm_calls,
            "kg_queries": state.resources.kg_queries,
            "kg_calls": getattr(self.kg, "calls", 0) - before_calls,
            "embedding_calls": getattr(self.embedder, "calls", 0) - before_embed[0],
            "embedding_tokens": getattr(self.embedder, "tokens", 0) - before_embed[1],
            "tokens": state.resources.tokens_spent, "seconds": elapsed, "judge_seconds": judge_seconds,
            "log": outcome.log,
            "sets": [{"name": s.name, "kind": s.kind, "size": len(s.entities),
                      "produced_by": s.produced_by, "counts_as": s.counts_as,
                      "refused": s.refused,
                      "sample": [e.label or e.id for e in s.entities][:6]}
                     for s in state.sets.values()],
            "graph": [{"id": t.id, "goal": t.goal, "status": str(t.status), "depth": t.depth,
                       "deps": t.dependencies,
                       "anchors": [e.label or e.id for e in t.grounded_inputs][:5],
                       "bindings": {role: [e.label or e.id for e in refs][:6]
                                    for role, refs in t.bindings.roles.items()},
                       "contract": "; ".join(r.description for r in
                                             (state.contract.requirements if state.contract else [])
                                             if r.id in t.satisfies),
                       "diagnosis": t.diagnosis[:160],
                       "attempts": [{
                           "n": a.attempt, "success": a.success,
                           "query": (a.executed_query or "")[:200],
                           "rows": a.result_count, "failure": a.failure_type,
                           "diagnosis": (a.diagnosis or "")[:120],
                           "candidates": [c.query[:160] for c in a.candidates],
                           "executions": [{"candidate": c.id, "query": c.query[:160],
                                           "rows": c.result_count, "outcome": c.outcome}
                                          for c in a.candidates],
                           "entities": a.grounded_entity_ids[:40],   # enough to tell whether gold was ever retrieved
                       } for a in (state.task_states[t.id].attempts
                                   if t.id in state.task_states else [])],
                       } for t in tasks.values()],
        }
        went_away = graph_went_away(row)
        if went_away:
            raise GraphUnavailable(f"question {index}: {went_away}")
        return row


# Which validation rules fire: the agent's rejections, counted per question.
REJECTED = "rejected: "


def rules_fired(state) -> dict[str, int]:
    counts: Counter = Counter()
    for task in state.task_states.values():
        for attempt in task.attempts:
            for candidate in attempt.candidates:
                counts["candidates"] += 1
                repairs = [r for r in (candidate.rationale or "").split("; ") if r]
                counts.update(f"repair.{r.split(':', 1)[0]}" for r in repairs)
                if not (candidate.outcome or "").startswith(REJECTED):
                    continue
                rules = [r.strip() for r in candidate.outcome.removeprefix(REJECTED).split(",") if r.strip()]
                counts["invalid"] += 1
                counts["invalid.seed" if candidate.id.startswith("s") else "invalid.model"] += 1
                counts.update(f"fired.{r}" for r in rules)
                if len(rules) == 1:
                    counts[f"alone.{rules[0]}"] += 1
    return dict(sorted(counts.items()))
