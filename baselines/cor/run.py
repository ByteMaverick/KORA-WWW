"""CoR (Chain-of-Relations; Liu, Zhou and Wang, Findings of ACL 2026; github.com/Chenhui7au/Chain-of-Relations @
f925579), the authors' code ported under chain_of_relations/: imports rewritten, three files edited (each edit marked
"harness:"), the rest checksummed against the authors' files (UPSTREAM.md).

Theirs: the algorithm, prompts (methods/cor/prompt.yml), stack, pruning, reasoning, filter, the generate-directly
fallback and the Freebase backend; relation width 3 and 1,000 relation candidates (their runner's).
From harness/settings.py: depth (the paper's 4 on CWQ, their runner's 3 elsewhere) and decoding (the paper's
temperature 0.01 for every step, their max_token 512, the model's allowance on gpt-oss, which reasons before it
answers). Model calls go through the harness's client; the graph is Freebase at the harness's endpoint. The estimator
and gold arms (cor_estimator, cor_gold) write every stop decision, with the executed set and CoR's own action, to the
question's verdicts file (harness/verdict_log.py). Not run on MetaQA: CoR writes SPARQL itself and MetaQA is served
from Neo4j (UPSTREAM.md).
"""
from __future__ import annotations

import argparse
import json
import os

from harness import verdict_log
from harness.dataset import QuestionSample
from harness.llm import run_llm
from harness.runner import QueryMetrics
from harness import settings
from harness.settings import decoding

from baselines.cor.chain_of_relations.methods.cor.agent import CoRAgent
from baselines.cor.chain_of_relations.schema import Entity

KGS = ("freebase",)


def _backend(kg: str):
	# The authors' backend reads its endpoint from FREEBASE_SPARQL_ENDPOINT when it is imported.
	os.environ.setdefault("FREEBASE_SPARQL_ENDPOINT", os.getenv("SPARQL_ENDPOINT", "http://127.0.0.1:8890/sparql"))
	from baselines.cor.chain_of_relations.kg_backend.freebase_backend import FreebaseBackend
	return FreebaseBackend()


class _HarnessLLM:
	"""CoR's LLMAPI.generate, answered by run_llm; each call is counted under the CoR step that made it."""

	def __init__(self, engine: str, max_tokens: int, qm: QueryMetrics, top_p: float = 1.0, top_k: int = -1):
		self.engine, self.max_tokens, self.qm, self.site = engine, max_tokens, qm, "cor"
		self.top_p, self.top_k = top_p, top_k   # the table's; their LLMAPI sent none (the API's defaults)

	def generate(self, user_prompt, temperature=0.01, max_tokens=256, system_prompt=None):
		# The temperature is the one CoR's step passes (the table's, set on the agent); the cap is the table's.
		text, metrics = run_llm(prompt=user_prompt, engine=self.engine, temperature=temperature,
		                        max_tokens=self.max_tokens, system_prompt=system_prompt or "",
		                        top_p=self.top_p, top_k=self.top_k)
		self.qm.add(metrics, call_site=self.site)
		usage = {"input_tokens": metrics.prompt_tokens, "output_tokens": metrics.completion_tokens,
		         "total_tokens": metrics.total_tokens}
		if not metrics.success or not text:
			usage["error"] = metrics.error or "empty_model_response"
			return None, usage
		return text, usage


class _Agent(CoRAgent):
	"""The authors' agent; only labels each model call with the step that made it (for the call counts and
	for telling the generate-directly fallback from the graph answers), and keeps the latest entity search and CoR's
	own verdict, which the estimator and gold arms write beside each stop decision (harness/verdict_log.py)."""

	own = None        # CoR's own verdict (its reasoning action) at the latest decision
	searched = None   # the latest entity_search: its root, chain and result

	def _at(self, site, fn, *a, **k):
		self.llm_api.site = site
		return fn(*a, **k)

	def relation_prune(self, *a, **k):
		return self._at("relation_prune", super().relation_prune, *a, **k)

	def reasoning(self, *a, **k):
		out = self._at("reasoning", super().reasoning, *a, **k)
		self.own = out[0] if isinstance(out, tuple) and out else None   # CoR's own verdict, for the verdicts file
		return out

	def entity_search(self, *a, **k):
		# The set (and the chain it came from) the next decision is made on, for the verdicts file
		# (harness/verdict_log.py); the search itself is the authors'.
		self.own, out = None, super().entity_search(*a, **k)
		self.searched = {"root": k.get("topic_entity"), "chain": k.get("relation_chain"), "found": out}
		return out

	def validate(self, *a, **k):
		return self._at("filter", super().validate, *a, **k)

	def generate_directly(self, *a, **k):
		return self._at("generate_directly", super().generate_directly, *a, **k)


def depth_for(search: dict, benchmark: str) -> int:
	"""The depth for a question set: its family's (the name before the first "-": "cwq" for cwq-test), else the default."""
	family = benchmark.lower().split("-", 1)[0]
	return search.get("depth_by_set", {}).get(family, search["depth"])


def run_cor(sample: QuestionSample, args: argparse.Namespace):
	kg = getattr(args, "kg_name", "")
	if kg not in KGS:
		raise NotImplementedError(f"[cor] kg_name={kg!r}: CoR runs here on {', '.join(KGS)}; not on MetaQA: CoR writes "
		                          "SPARQL itself and MetaQA is served from Neo4j (baselines/cor/UPSTREAM.md)")
	qm = QueryMetrics(question=sample.question)
	# Their runner passes 1,000 relation candidates into relation_prune (the agent's own default is 500); width and
	# depth from the settings table, as are the temperatures and max tokens.
	framework = getattr(args, "framework", "") or "cor_native"
	d = decoding(framework, args.engine)
	search = settings.system(framework)
	depth = depth_for(search, getattr(args, "dataset_benchmark", "") or sample.benchmark or "")
	agent = _Agent(model_name=args.engine, relation_width=search["relation_width"], depth=depth, sample_relation_threshold=1000,
	               temperature_exploration=d["temperature_exploration"], temperature_reasoning=d["temperature_reasoning"],
	               max_token=d["max_tokens"], backend=_backend(kg))
	agent.llm_api = _HarnessLLM(args.engine, d["max_tokens"], qm, top_p=d["top_p"], top_k=d["top_k"])
	topic = dict(sample.topic_entity or {})
	entities = [Entity(uri, label) for uri, label in topic.items()]
	print(f"\n  Question : {sample.question}\n  Topic    : {topic}")
	decisions: list = []
	if framework == "cor_estimator":              # the Stopping Estimator decides
		out = agent.answer(sample.question, entities, estimator=estimator_judge(decisions, framework, agent))
	elif framework == "cor_gold":                 # ... or the gold, over the same executed sets
		graph_oracle = getattr(args, "stop_oracle_callback", None)
		out = agent.answer(sample.question, entities,
		                   graph_oracle=None if graph_oracle is None else GoldLogged(graph_oracle, framework, agent))
		if graph_oracle is not None:
			decisions = graph_oracle.log
	else:
		out = agent.answer(sample.question, entities)
	names = [r["name"] if isinstance(r, dict) else str(r) for r in out.get("results", [])]
	items = [n for n in names if n]
	answer = "; ".join(items)
	print(f"  CoR ended by {out.get('action')}: {answer[:200]}")
	qm.report()
	summary = qm.summary()
	summary["cor_depth"] = depth
	summary["cor_action"] = out.get("action")   # Stop / Constraint / generate_directly / estimator_stop / ERROR
	summary["cor_chain"] = out.get("reasoning_chains") if isinstance(out.get("reasoning_chains"), str) else ""
	summary["stop_decisions"] = decisions
	return items, summary   # a list: the harness scores its items


# ── The estimator and gold arms ──────────────────────────────────────────────────────────────────────────────────────
import urllib.request as _urlreq

from baselines import estimator as _estimator


def chain_query(root_entity, relation_chain) -> str:
	"""The executed chain from the topic, in KORA's query dialect (the estimator's server renders it as the
	estimator was trained): ?v1 ... ?ans."""
	node, triples = str(root_entity.id), []
	for k, relation in enumerate(relation_chain, start=1):
		nxt = "?ans" if k == len(relation_chain) else f"?v{k}"
		r = str(relation.id)
		triples.append(f"{node} {r} {nxt}" if relation.left else f"{nxt} {r} {node}")
		node = nxt
	return "SELECT ?ans WHERE { " + " . ".join(triples) + " }"


def _candidate(ids, names, query, chain) -> dict:
	"""A CoR decision's candidate, for the verdicts file: the executed set (entity ids, then literal values as
	themselves; names likewise), the text the gold scores (names joined by ", ", as agent.py passes them) and the
	chain."""
	return {"answer": ", ".join(names), "names": names, "ids": ids, "n": len(ids), "query": query,
	        "relations": [f"{r.id}({'head' if r.left else 'tail'})" for r in chain or []]}


def estimator_judge(log: list, arm: str = "cor_estimator", agent=None):
	"""The Stopping Estimator over CoR's executed chain: the question, the chain from the topic (KORA's query dialect,
	so the server renders it as the estimator was trained) and the whole returned set (ids, and literal values as
	themselves). Every decision is also written to the question's verdicts file (harness/verdict_log.py)."""
	url, thr = _estimator.url(), _estimator.threshold()

	def judge(question, root_entity, relation_chain, entity_ids, literal_candidates) -> bool:
		query = chain_query(root_entity, relation_chain)
		ids = [str(i) for i in entity_ids] + [str(v) for v in literal_candidates]
		body = {"states": [""], "questions": [question],
		        "extras": [{"ids": ids, "n": len(ids), "query": query, "chain": []}]}
		p, error = None, None
		for _ in range(3):
			try:
				req = _urlreq.Request(url + "/score", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
				with _urlreq.urlopen(req, timeout=120) as r:
					p = float(json.load(r)["p"][0])
				error = None
				break
			except Exception as e:
				error = f"{type(e).__name__}: {str(e)[:300]}"
				continue
		accept = p is not None and p >= thr
		log.append({"chain": body["extras"][0]["query"], "n": len(ids), "p": p, "accept": accept})
		try:   # the names are the agent's (its latest entity_search); a failed write never changes the decision
			found = (getattr(agent, "searched", None) or {}).get("found") or ([], [], [])
			names = [str(n) for n in found[1]] + [str(v) for v in literal_candidates]
			verdict_log.judged(arm, [_candidate(ids, names, query, relation_chain)], [p], 0 if accept else None,
			                   judge_url=url, threshold=thr, error=error, depth=len(relation_chain), step=len(log),
			                   own_verdict=getattr(agent, "own", None))
		except Exception:
			pass
		return accept
	return judge


class GoldLogged:
	"""The gold's stop callback for cor_gold (harness/stop_oracle.StopOracle), unchanged, and each verdict it gives
	written to the question's verdicts file (harness/verdict_log.py), with the set it was given."""

	def __init__(self, oracle, arm: str, agent):
		self.oracle, self.arm, self.agent = oracle, arm, agent

	def __getattr__(self, name):
		return getattr(self.oracle, name)

	def right(self, step, judge_stop: bool, answer: str, path=None) -> bool:
		ok = self.oracle.right(step, judge_stop, answer, path=path)
		try:
			searched = getattr(self.agent, "searched", None) or {}
			entity_ids, entity_names, literals = searched.get("found") or ([], [], [])
			chain = searched.get("chain") or []
			ids = [str(i) for i in entity_ids] + [str(v) for v in literals]
			query = chain_query(searched["root"], chain) if searched.get("root") is not None else None
			candidate = {**_candidate(ids, [str(n) for n in entity_names] + [str(v) for v in literals], query, chain),
			             "answer": answer, "path": path}
			verdict_log.golded(self.arm, [candidate], [ok], self.oracle.log[-1:], 0 if ok else None,
			                   depth=len(chain), step=step, own_verdict=getattr(self.agent, "own", None))
		except Exception:
			pass
		return ok
