"""Think-on-Graph (Sun et al., ICLR 2024; github.com/IDEA-FinAI/ToG @ 7ccbb92), the authors' code vendored unchanged in
upstream/ (UPSTREAM.md). One question is the `for data in tqdm(datas):` body of their main_freebase.py, read from that
file and run for that question with their functions and prompts.

The setting (the paper's value where it states one, the code's where it is silent):
  * the paper's (§3.1.3, "Experiment details"): width N = 3, depth Dmax = 3, exploration temperature 0.4,
    reasoning temperature 0, max_length 256, pruning by the LLM; the incoming-relation search as §2.1.2 writes it,
    (?, r, e), where the released code's argument order finds nothing;
  * the code's, where the paper is silent: 5 entities kept, sampled at random, when a relation has 20 or more
    (--num_retain_entity);
  * the code's procedure where the paper describes another: §2.1.3 prompts a separate Generate call once the
    evaluation is positive; their code takes the answer from the evaluation reply, and that is what runs.
Temperatures and max_length come from harness/settings.py; on gpt-oss the cap is the model's allowance, since its
reasoning counts against it.

What the harness changes (marked "harness:" below): their run_llm is answered by the harness's client with their
system message, each call counted under the function that made it; the graph is Freebase at the harness's endpoint
or MetaQA through graphs.py; their save_2_jsonl is captured, and an answer saved before their loop crashes is kept
(their loop goes on after some half_stop saves and can then fail); the answer is read with extract_content from
their eval/utils.py (their eval.py reads only the first {...}, which is "Yes" for every stopped question).
"""
from __future__ import annotations

import argparse
import ast
import os
import random
import re
import sys

from baselines.official_llm import chat
from baselines.official_loader import load
from baselines.official_sampling import sampling
from baselines.tog import graphs
from harness.runner import QueryMetrics

HERE = os.path.dirname(os.path.abspath(__file__))
UPSTREAM = os.path.join(HERE, "upstream")
_m = load(UPSTREAM, ["prompt_list", "utils", "freebase_func"])
fb, ut = _m["freebase_func"], _m["utils"]

#: Their main_freebase.py defaults; the temperatures and max_length come from the settings table (args_for).
AUTHORS = dict(width=3, depth=3, num_retain_entity=5, prune_tools="llm", remove_unnecessary_rel=True,
               opeani_api_keys="")
SYSTEM = "You are an AI assistant that helps people find information."     # their run_llm's system message
_state = {"qm": None, "saved": [], "site": "", "sampling": {}}


def _run_llm(prompt, temperature, max_tokens, opeani_api_keys, engine="gpt-3.5-turbo"):
    # harness: their run_llm's call, answered by the harness's server and counted
    site = sys._getframe(1).f_code.co_name
    # harness: with the table's top_p / top_k (their call sent none: the API's defaults)
    text, metrics = chat(prompt, SYSTEM, engine, temperature, max_tokens, **_state["sampling"])
    if _state["qm"] is not None:
        _state["qm"].add(metrics, call_site=site)
    _state["site"] = site
    return text


def triples(chains) -> int:
    """The triples a chain list puts in their answer prompts (one line per depth, its triples joined)."""
    return 1 if isinstance(chains, tuple) else sum(triples(c) for c in chains) if isinstance(chains, list) else 0


def _save_2_jsonl(question, answer, cluster_chain_of_entities, file_name):
    # harness: captured, with the call that wrote it and the triples it was written over (counted now: their loop
    # can go on appending to the same list after a half_stop)
    _state["saved"].append({"results": answer, "reasoning_chains": cluster_chain_of_entities,
                            "route": _state["site"], "triples": triples(cluster_chain_of_entities)})


for _mod in (fb, ut):
    _mod.run_llm, _mod.save_2_jsonl = _run_llm, _save_2_jsonl


def _their_loop() -> str:
    """The `for data in tqdm(datas):` loop of the authors' main_freebase.py, as written, over one question."""
    src = open(os.path.join(UPSTREAM, "main_freebase.py")).read()
    tree = ast.parse(src)
    main = next(n for n in tree.body if isinstance(n, ast.If))
    loop = next(n for n in main.body if isinstance(n, ast.For))
    text = ast.get_source_segment(src, loop)
    head, rest = text.split("\n", 1)
    assert head.strip() == "for data in tqdm(datas):", head
    import textwrap
    # blank lines first, so a traceback names the line of their main_freebase.py
    return "\n" * (loop.lineno - 1) + "for data in __datas:\n" + textwrap.indent(textwrap.dedent(rest), "    ")


LOOP = compile(_their_loop(), os.path.join(UPSTREAM, "main_freebase.py"), "exec")


def extract_content(s: str) -> str:
    """The authors' eval/utils.py extract_content (ToG repo), verbatim."""
    matches = re.findall(r'\{(.*?)\}', s)
    if len(matches) >= 2 and matches[0].lower() == 'yes':
        return matches[1]
    elif len(matches) >= 1:
        return matches[0]
    else:
        return 'NULL'


def answer_of(reply: str) -> str:
    found = extract_content(reply or "")
    return (reply or "").strip() if found == "NULL" else found.strip()


def args_for(args: argparse.Namespace) -> argparse.Namespace:
    from harness.settings import decoding
    d = decoding(getattr(args, "framework", "") or "tog_native", args.engine)     # harness: the settings table
    a = dict(AUTHORS, temperature_exploration=d["temperature_exploration"],
             temperature_reasoning=d["temperature_reasoning"], max_length=d["max_tokens"])
    return argparse.Namespace(dataset=getattr(args, "dataset_benchmark", "") or "harness", LLM_type=args.engine, **a)


def run_tog(sample, args: argparse.Namespace):
    from harness.settings import system
    kg = getattr(args, "kg_name", "") or "freebase"
    framework = getattr(args, "framework", "") or "tog_native"
    # The head order is the settings table's (harness/settings.py): the paper's.
    graphs.HEAD_ORDER = system(framework).get("head_order", "paper")
    graphs.install(fb, kg)
    graphs.reset()
    qm = QueryMetrics(question=sample.question)
    _state.update(qm=qm, saved=[], site="", sampling=sampling(framework, args.engine),
                  produced={e: None for e in (sample.topic_entity or {})}, decisions=[], groups={},
                  names=dict(sample.topic_entity or {}))
    random.seed(f"tog:{sample.qid}")                     # harness: their random.sample, reproducible per question
    data = {"question": sample.question, "topic_entity": dict(sample.topic_entity or {})}
    ns = dict(vars(fb))
    ns.update(__datas=[data], question_string="question", args=args_for(args), random=random)
    if framework in STOP_ARMS:                         # the Stopping Estimator, or the gold, decides
        stop_install(ns, framework, getattr(args, "stop_oracle_callback", None))
    crash = None
    try:
        exec(LOOP, ns)
    except Exception as e:       # harness: a crash after their loop saved an answer keeps that answer
        if not _state["saved"]:
            raise
        crash = f"{type(e).__name__}: {str(e)[:300]}"
        print(f"  ToG (official): their loop failed after saving an answer ({crash}); the saved answer is kept")
    reply = _state["saved"][0]["results"] if _state["saved"] else ""
    answer = answer_of(reply)
    summary = qm.summary()
    summary.update({"graph_queries": graphs.QUERIES[0], "graph_rejected": graphs.REJECTED[0],
                    "tog_reply": (reply or "")[:2000], "tog_saved": len(_state["saved"]),
                    "tog_depth": len(_state["saved"][0]["reasoning_chains"]) if _state["saved"] else 0,
                    "answer_route": _state["saved"][0]["route"] if _state["saved"] else "",
                    "answer_triples": _state["saved"][0]["triples"] if _state["saved"] else 0,
                    "tog_crash_after_save": crash, "stop_decisions": _state.get("decisions", [])})
    print(f"  ToG (official) {kg}: {summary.get('num_llm_calls')} calls -> {answer[:200]}")
    return answer, summary


# ── The estimator and gold arms ──────────────────────────────────────────────────────────────────────────────────────
#: tog_estimator: ToG's evaluation call (`reasoning`, once per depth: "are these triples enough to answer") replaced by
#: the Stopping Estimator served at ESTIMATOR_URL (its cut: baselines/estimator.threshold); tog_gold: the gold decides,
#: over the same candidates. A candidate is one (topic, relation, direction) group of the current depth that holds a
#: kept entity, with every entity ToG's search returned for it (after its own 20 -> 5 sampling, before its 3-entity
#: prune), and the hops that led to the topic as its path. Accepted: ToG stops and answers with that group's names
#: (read from the graph). Everything else is ToG as published: its search, pruning and its fallbacks (half_stop, the
#: no-path answer), which still answer when nothing was accepted. The groups are read where the loop records them
#: (update_history), since their lists can be misaligned after entity_score drops an unnamed entity. Every decision,
#: each candidate accepted or not, is written to the question's verdicts file (harness/verdict_log.py).
import json
import urllib.request

from baselines import estimator
from harness import verdict_log

STOP_ARMS = ("tog_estimator", "tog_gold")


def _named(name) -> bool:
    return bool(name) and name not in ("UnName_Entity", "[FINISH]")


def stop_install(ns, framework, oracle):
    """Hook the loop for a stop arm: record each searched group (update_history) and each name the loop read
    (id2entity_name_or_type, unchanged calls), and replace reasoning(question, cluster_chain_of_entities, args)."""
    names, groups = _state["names"], _state["groups"]
    lookup = fb.id2entity_name_or_type                   # set per question by graphs.install

    def recorded(entity_id):
        name = lookup(entity_id)
        names[entity_id] = name
        return name
    fb.id2entity_name_or_type = ns["id2entity_name_or_type"] = recorded
    update_history = ns["update_history"]

    def history(entity_candidates, entity, scores, entity_candidates_id, *rest):
        ids = [i for i in entity_candidates_id if i != "[FINISH_ID]"]
        if ids:
            group = groups.setdefault(ns["depth"], {}).setdefault((entity["entity"], entity["relation"], bool(entity["head"])), [])
            group.extend([i for i in dict.fromkeys(ids) if i not in group])
        return update_history(entity_candidates, entity, scores, entity_candidates_id, *rest)
    ns["update_history"] = history
    ns["reasoning"] = stop_reasoning(ns, framework, oracle)


def stop_reasoning(ns, framework, oracle):
    thr = estimator.threshold() if framework == "tog_estimator" else None
    names, produced = _state["names"], _state["produced"]

    def name_of(i):
        if i not in names:
            names[i] = fb.id2entity_name_or_type(i)
        return names[i]

    def reasoning(question, cluster_chain_of_entities, args):
        depth, kept = ns["depth"], set(ns.get("entities_id") or [])
        found = []
        for (topic, rel, head), ids in (_state["groups"].get(depth) or {}).items():
            if not kept.intersection(ids):
                continue
            for i in ids:                                 # the next depth's topics: where each entity came from
                produced.setdefault(i, (topic, rel, head))
            named = [i for i in ids if _named(name_of(i))]
            if not named:
                continue
            hops, node, seen = [(topic, rel, head, None)], topic, set()
            while produced.get(node) is not None and node not in seen and len(hops) < 5:   # a question's topics: None
                seen.add(node)
                parent, p_rel, p_head = produced[node]
                hops.insert(0, (parent, p_rel, p_head, node))
                node = parent
            var = {}

            def term(i):
                """A node in the path's query: ?ans; a named entity as its id (quoted if it has spaces: MetaQA);
                an unnamed one as one variable wherever it occurs."""
                if i is None:
                    return "?ans"
                if not _named(name_of(i)):
                    return var.setdefault(i, f"?v{len(var) + 1}")
                k = str(i)
                return k if re.fullmatch(r"[\w.:\-]+", k) else '"' + k.replace('"', '') + '"'
            triples = [f"{term(a)} {r} {term(b)}" if h else f"{term(b)} {r} {term(a)}"
                       for a, r, h, b in hops]
            found.append({"topic": topic, "relation": rel, "head": head, "ids": named, "names": [names[i] for i in named],
                          "query": "SELECT ?ans WHERE { " + " . ".join(triples) + " }"})
        best, record = None, {"depth": depth, "groups": len(found)}
        # harness: every decision is also written to the question's verdicts file (harness/verdict_log.py)
        chosen, step = None, len(_state["decisions"]) + 1
        candidates = [{"answer": "; ".join(g["names"]), "names": g["names"], "ids": g["ids"], "n": len(g["ids"]),
                       "query": g["query"], "topic": g["topic"], "relation": g["relation"], "head": g["head"]}
                      for g in found]
        if framework == "tog_estimator":
            p, error = [None] * len(found), None
            if found:
                body = {"states": [""] * len(found), "questions": [question] * len(found),
                        "extras": [{"ids": [str(i) for i in g["ids"]], "n": len(g["ids"]), "query": g["query"], "chain": []}
                                   for g in found]}
                req = urllib.request.Request(estimator.url() + "/score", data=json.dumps(body).encode(),
                                             headers={"Content-Type": "application/json"})
                for attempt in range(3):
                    try:
                        with urllib.request.urlopen(req, timeout=120) as r:
                            got = json.load(r)["p"]
                        if len(got) == len(found):
                            p, error = [float(x) for x in got], None
                            break
                        error = f"the server returned {len(got)} scores for {len(found)} states"
                    except Exception as e:
                        error = f"{type(e).__name__}: {str(e)[:300]}"
                ok = [(s, k) for k, s in enumerate(p) if s is not None]
                record.update(p=[None if s is None else round(s, 4) for s in p], failed=sum(s is None for s in p))
                if ok and max(ok)[0] >= thr:
                    chosen = max(ok)[1]
                    best = found[chosen]
            verdict_log.judged(framework, candidates, p, chosen, judge_url=estimator.url(),
                               threshold=thr, error=error, depth=depth, step=step)
        elif oracle is not None:
            right, entries = [], []
            if found:
                right = [oracle.right(depth, False, "; ".join(g["names"]), g["query"]) for g in found]
                entries = oracle.log[-len(found):]
                f1 = [e["f1"] for e in entries]
                take = [k for k in range(len(found)) if right[k]]
                if take:
                    chosen = max(take, key=lambda k: f1[k])
                    best = found[chosen]
            verdict_log.golded(framework, candidates, right, entries, chosen, depth=depth, step=step)
        else:
            verdict_log.golded(framework, candidates, [], [], None, depth=depth, step=step,
                               error="no gold: the gold arm has no stop callback for this question")
        record.update(stop=best is not None, answer=(best or {}).get("names", [])[:30],
                      queries=[g["query"] for g in found])
        _state["decisions"].append(record)
        _state["site"] = "reasoning"                      # the answer's route: the evaluation step's
        if best is None:
            return False, "{No}"
        return True, "{Yes}. {" + "; ".join(best["names"]) + "}"
    return reasoning
