"""Plan-on-Graph (Chen et al., NeurIPS 2024; github.com/liyichen-cly/PoG @ 6d1627e), the authors' code vendored
unchanged in upstream/ (UPSTREAM.md). One question is the `for data in tqdm(datas):` body of their main_freebase.py,
read from that file and run for that question with their functions and prompts: subobjectives, relation pruning,
entity search, condition pruning with memory, reflection (adding entities back up to 5 times), reasoning, half_stop
and the no-path answer.

The setting: depth 4, exploration and reasoning temperature 0.3, max_length 1,024 (the paper's App. E; their script
defaults to 4,096), the msmarco-distilbert-base-tas-b sentence model. Temperatures and max_length come from
harness/settings.py; on gpt-oss the cap is the model's allowance (4,096).

What the harness changes (marked "harness:" below):
  * the model: their run_llm is answered by the harness's client with their system message and their
    (text, token counts) shape; each call is counted under the function that made it;
  * the graph: Freebase at the harness's endpoint; MetaQA through graphs.py;
  * one line of their loop: `if entity.startswith("m."):` reads `if is_entity(entity):`, the same test on Freebase
    and "an entity node" on MetaQA;
  * their per-question memory folder ('../mem/<dataset>/<model>/<question>') lives in a temporary folder;
  * the result: their save_2_jsonl is captured. Their loop's bare `except: continue` turns any error into no saved
    line (the authors rerun such questions). A graph that could not answer is raised again after their loop; any
    other swallowed exception with no answer saved becomes the row's error ("pog_unanswered: <type>: <text>"), seen
    with sys.monitoring, their code unchanged;
  * the clock: their bare except also catches the harness's timeout. While their loop runs, the alarm handler is
    wrapped to remember it; once it has fired, every model call raises it again, and it is raised after their loop,
    so the row is a timeout;
  * the answer: the saved reply read as their eval/eval.py reads it: the JSON from the first "{", its A.Answer or
    Answer; else the last "Answer": "..." or the "Answer": [...] list; else the whole reply.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import random
import contextlib
import re
import signal
import sys
import tempfile
import threading
import textwrap

from baselines.official_llm import chat
from baselines.official_loader import load
from baselines.official_sampling import sampling
from baselines.pog import graphs
from harness.runner import QueryMetrics

HERE = os.path.dirname(os.path.abspath(__file__))
UPSTREAM = os.path.join(HERE, "upstream")
_m = load(UPSTREAM, ["prompt_list", "utils", "freebase_func"])
fb, ut = _m["freebase_func"], _m["utils"]

#: The paper's setting (section 4.1): depth 4; temperature 0.3 and "the maximum token length for generation is 1024"
#: are in the settings table (harness/settings.DECODING), read by args_for.
AUTHORS = dict(depth=4, remove_unnecessary_rel=True, opeani_api_keys="")
SENTENCE_MODEL = os.getenv("POG_SENTENCE_MODEL", "sentence-transformers/msmarco-distilbert-base-tas-b")
SYSTEM = "You are an AI assistant that helps people find information."     # their run_llm's system message
LINE_THEIRS, LINE_OURS = 'if entity.startswith("m."):', "if is_entity(entity):"
_state = {"qm": None, "saved": [], "model": None, "site": "", "timeout": None, "sampling": {}}


def _run_llm(prompt, temperature, max_tokens, opeani_api_keys, engine="gpt-3.5-turbo", print_in=True, print_out=True):
    # harness: their run_llm's call, answered by the harness's server and counted; their (text, tokens) shape
    if _state["timeout"] is not None:      # the question's time is up (see _remember_timeout)
        raise _state["timeout"]
    site = sys._getframe(1).f_code.co_name
    # harness: with the table's top_p / top_k (their call sent none: the API's defaults)
    text, metrics = chat(prompt, SYSTEM, engine, temperature, max_tokens, **_state["sampling"])
    if _state["qm"] is not None:
        _state["qm"].add(metrics, call_site=site)
    _state["site"] = site
    return text, {"total": metrics.total_tokens, "input": metrics.prompt_tokens, "output": metrics.completion_tokens}


def _save_2_jsonl(question, question_string, answer, cluster_chain_of_entities, call_num, all_t, start_time, file_name):
    # harness: captured, with the route its answer took: the call that wrote it (every save follows the call whose
    # reply it saves) and the triples it was written over
    _state["saved"].append({"results": answer, "reasoning_chains": cluster_chain_of_entities, "call_num": call_num,
                            "route": _state["site"], "triples": triples(cluster_chain_of_entities)})


def triples(chains) -> int:
    """The triples a chain list puts in their generate_answer prompt."""
    return 1 if isinstance(chains, tuple) else sum(triples(c) for c in chains) if isinstance(chains, list) else 0


for _mod in (fb, ut):
    _mod.run_llm, _mod.save_2_jsonl = _run_llm, _save_2_jsonl


def _their_loop() -> str:
    src = open(os.path.join(UPSTREAM, "main_freebase.py")).read()
    tree = ast.parse(src)
    main = next(n for n in tree.body if isinstance(n, ast.If))
    loop = next(n for n in main.body if isinstance(n, ast.For))
    text = ast.get_source_segment(src, loop)
    head, rest = text.split("\n", 1)
    assert head.strip() == "for data in tqdm(datas):", head
    assert rest.count(LINE_THEIRS) == 1, "the one line the harness reads differently"
    rest = rest.replace(LINE_THEIRS, LINE_OURS)
    # blank lines first, so a traceback names the line of their main_freebase.py
    return "\n" * (loop.lineno - 1) + "for data in __datas:\n" + textwrap.indent(textwrap.dedent(rest), "    ")


LOOP = compile(_their_loop(), os.path.join(UPSTREAM, "main_freebase.py"), "exec")


def answer_of(reply) -> str | list:
    """The answer as the authors' eval/eval.py reads a saved reply."""
    text = reply if isinstance(reply, str) else str(reply or "")
    start = text.find("{")
    if start == -1:
        return text.strip()
    try:
        obj = json.loads(text[start:])
        found = obj["A"]["Answer"] if "A" in obj else obj["Answer"]
    except Exception:
        many = list(re.finditer(r'"Answer":\s*["\']([^"\']+)["\']', text[start:]))
        if many:
            found = many[-1].group(1)
        else:
            listed = re.search(r'"Answer":\s*(\[[^\]]+\])', text[start:])
            try:
                found = json.loads(listed.group(1)) if listed else text
            except Exception:
                found = text
    if isinstance(found, list):
        return [str(x) for x in found]   # a list: the harness scores its items
    return str(found).strip()


def _sentence_model():
    if _state["model"] is None:
        from sentence_transformers import SentenceTransformer
        _state["model"] = SentenceTransformer(SENTENCE_MODEL, device="cpu")
    return _state["model"]


def args_for(args: argparse.Namespace) -> argparse.Namespace:
    from harness.settings import decoding
    d = decoding(getattr(args, "framework", "") or "pog_native", args.engine)     # harness: the settings table
    a = dict(AUTHORS, temperature_exploration=d["temperature_exploration"],
             temperature_reasoning=d["temperature_reasoning"], max_length=d["max_tokens"])
    return argparse.Namespace(dataset=re.sub(r"[^A-Za-z0-9_-]", "_", getattr(args, "dataset_benchmark", "") or "harness"),
                              LLM_type=args.engine, **a)


@contextlib.contextmanager
def _swallowed(code):
    """harness: the exceptions handled inside their loop's own frame (code), i.e. by its bare `except:` (its only
    handler), seen with sys.monitoring's EXCEPTION_HANDLED event; their code is unchanged. Yields the list."""
    caught: list = []
    mon = getattr(sys, "monitoring", None)
    tool = next((t for t in (3, 4) if mon is not None and mon.get_tool(t) is None), None)
    if tool is None:                     # no sys.monitoring (Python < 3.12) or no free tool id: the row says so
        yield caught
        return
    events = mon.events.EXCEPTION_HANDLED
    mon.use_tool_id(tool, "pog adapter")
    mon.register_callback(tool, events, lambda c, offset, exc: caught.append(exc) if c is code else None)
    mon.set_events(tool, events)
    try:
        yield caught
    finally:
        mon.set_events(tool, 0)
        mon.register_callback(tool, events, None)
        mon.free_tool_id(tool)


@contextlib.contextmanager
def _remember_timeout():
    """harness: the harness's SIGALRM handler (harness/runner.py), wrapped to keep the QuestionTimeoutError it raises."""
    previous = signal.getsignal(signal.SIGALRM)
    if not callable(previous) or threading.current_thread() is not threading.main_thread():
        yield
        return

    def on_alarm(signum, frame):
        try:
            previous(signum, frame)
        except BaseException as e:
            _state["timeout"] = e
            raise
    signal.signal(signal.SIGALRM, on_alarm)
    try:
        yield
    finally:
        signal.signal(signal.SIGALRM, previous)


def run_pog(sample, args: argparse.Namespace):
    kg = getattr(args, "kg_name", "") or "freebase"
    graphs.install(fb, kg)
    graphs.tog_graphs.reset()
    qm = QueryMetrics(question=sample.question)
    _state.update(qm=qm, saved=[], site="", timeout=None,
                  sampling=sampling(getattr(args, "framework", "") or "pog_native", args.engine),
                  produced={e: None for e in (sample.topic_entity or {})}, decisions=[])
    random.seed(f"pog:{sample.qid}")                     # harness: their random.sample, reproducible per question
    data = {"question": sample.question, "topic_entity": dict(sample.topic_entity or {})}
    ns = dict(vars(fb))
    ns.update(__datas=[data], question_string="question", args=args_for(args), model=_sentence_model(),
              part_q=False, q_set=set(), random=random, time=__import__("time"), os=os, pprint=__import__("pprint"),
              is_entity=graphs.is_entity)
    framework = getattr(args, "framework", "") or "pog_native"
    if framework in STOP_ARMS:                         # the Stopping Estimator, or the gold, decides
        ns["reasoning"] = stop_reasoning(framework, getattr(args, "stop_oracle_callback", None), ns)
    here = os.getcwd()
    with tempfile.TemporaryDirectory(prefix="pog-") as tmp:
        os.makedirs(os.path.join(tmp, "run"))
        os.chdir(os.path.join(tmp, "run"))               # harness: their '../mem/...' folder, inside a temp folder
        try:
            with _remember_timeout(), _swallowed(LOOP) as swallowed:
                exec(LOOP, ns)
        finally:
            os.chdir(here)
    if _state["timeout"] is not None:              # harness: their bare except swallowed the timeout
        raise _state["timeout"]
    if graphs.tog_graphs.FAILURE[0] is not None:     # harness: their bare except swallowed a graph outage
        raise graphs.tog_graphs.FAILURE[0]
    reply = _state["saved"][0]["results"] if _state["saved"] else ""
    answer = answer_of(reply)
    summary = qm.summary()
    summary.update({"graph_queries": graphs.QUERIES[0], "graph_rejected": graphs.tog_graphs.REJECTED[0],
                    "pog_reply": str(reply or "")[:2000],
                    "pog_saved": len(_state["saved"]),
                    "answer_route": _state["saved"][0]["route"] if _state["saved"] else "",
                    "answer_triples": _state["saved"][0]["triples"] if _state["saved"] else 0,
                    "stop_decisions": _state.get("decisions", [])})
    if not _state["saved"]:                         # harness: their except swallowed a crash: an error, rerun
        crash = swallowed[-1] if swallowed else None
        what = f"{type(crash).__name__}: {str(crash)[:300]}" if crash is not None else \
            "their loop ended without a saved answer (the exception was not seen)"
        summary.update({"error": f"pog_unanswered: {what}", "pog_swallowed": what})
    print(f"  PoG (official) {kg}: {summary.get('num_llm_calls')} calls -> {str(answer)[:200]}")
    return answer, summary


# ── The estimator and gold arms ──────────────────────────────────────────────────────────────────────────────────────
#: pog_estimator: PoG's evaluation call (`reasoning`, once per depth: "is this sufficient, and what is the answer")
#: replaced by the Stopping Estimator served at ESTIMATOR_URL (its cut: baselines/estimator.threshold); pog_gold: the
#: gold decides, over the same candidates. A candidate is one kept group of the current depth, (topic, head/tail,
#: relation) -> its entities, with the hops that led to the topic as its path. Accepted: PoG stops and answers with that
#: group's names (read from the graph). Everything else is PoG as published: planning, pruning, memory, reflection and
#: its two fallbacks (half_stop, the no-path answer), which still answer when nothing was accepted. Every decision, each
#: candidate accepted or not, is written to the question's verdicts file (harness/verdict_log.py).
import urllib.request

from baselines import estimator
from harness import verdict_log

STOP_ARMS = ("pog_estimator", "pog_gold")


def candidate_groups(ent_rel_ent_dict, entid_name, produced):
    """Every kept (topic, head/tail, relation) group of the current depth with a named member: its ids, names and the
    query of its path (the hops that led to the topic, then this hop) in KORA's query dialect, the estimator's input. Every
    member, named or not, is recorded as reached through its group (the next depth's paths); a question's topic
    entities are the roots (None), so a path never loops back through them."""
    named = lambda i: bool(entid_name.get(i)) and entid_name[i] != "UnName_Entity"
    out = []
    for topic, ht in sorted(ent_rel_ent_dict.items()):
        for h_t, rels in sorted(ht.items()):
            for rel, ids in sorted(rels.items()):
                ids = list(dict.fromkeys(ids))
                for i in ids:
                    produced.setdefault(i, (topic, h_t, rel))
                ids = [i for i in ids if named(i)]
                if not ids:
                    continue
                hops, node, seen = [(topic, h_t, rel, None)], topic, set()
                while produced.get(node) is not None and node not in seen and len(hops) < 5:
                    seen.add(node)
                    parent, p_ht, p_rel = produced[node]
                    hops.insert(0, (parent, p_ht, p_rel, node))
                    node = parent
                var = {}

                def term(i):
                    """?ans; a named entity as its id (quoted if it has spaces: MetaQA); an unnamed one as one
                    variable wherever it occurs."""
                    if i is None:
                        return "?ans"
                    if not named(i):
                        return var.setdefault(i, f"?v{len(var) + 1}")
                    k = str(i)
                    return k if re.fullmatch(r"[\w.:\-]+", k) else '"' + k.replace('"', '') + '"'
                triples = [f"{term(a)} {r} {term(b)}" if p == "head" else f"{term(b)} {r} {term(a)}"
                           for a, p, r, b in hops]
                out.append({"topic": topic, "h_t": h_t, "relation": rel, "ids": ids,
                            "names": [entid_name[i] for i in ids],
                            "query": "SELECT ?ans WHERE { " + " . ".join(triples) + " }"})
    return out


def estimator_scores(question, groups, errors: list | None = None):
    """P(the candidate answers the question) for every group, from the estimator's server; None where it failed (the last
    error after the retries is appended to `errors`, for the verdicts file)."""
    body = {"states": [""] * len(groups), "questions": [question] * len(groups),
            "extras": [{"ids": [str(i) for i in g["ids"]], "n": len(g["ids"]), "query": g["query"], "chain": []}
                       for g in groups]}
    req = urllib.request.Request(estimator.url() + "/score", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    error = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                p = json.load(r)["p"]
            if len(p) == len(groups):
                return [float(x) for x in p]
            error = f"the server returned {len(p)} scores for {len(groups)} states"
        except Exception as e:
            error = f"{type(e).__name__}: {str(e)[:300]}"
    if errors is not None:
        errors.append(error)
    return [None] * len(groups)


def stop_reasoning(framework, oracle, ns=None):
    """PoG's reasoning(question, subquestions, ent_rel_ent_dict, entid_name, chains, mem_path, args), replaced: the
    estimator (or the gold) over the current depth's groups. Returns PoG's (reply, answer, sufficient, tokens). `ns`, their
    loop's namespace, gives the verdicts file the loop's depth."""
    thr = estimator.threshold() if framework == "pog_estimator" else None

    def reasoning(question, subquestions, ent_rel_ent_dict, entid_name, cluster_chain_of_entities, q_mem_f_path, args):
        produced = _state.setdefault("produced", {})
        groups = candidate_groups(ent_rel_ent_dict, entid_name, produced)
        best, record = None, {"depth": len(_state["decisions"]) + 1, "groups": len(groups)}
        # harness: every decision is also written to the question's verdicts file (harness/verdict_log.py)
        chosen, step, depth = None, record["depth"], (ns or {}).get("depth")
        candidates = [{"answer": "; ".join(g["names"]), "names": g["names"], "ids": g["ids"], "n": len(g["ids"]),
                       "query": g["query"], "topic": g["topic"], "h_t": g["h_t"], "relation": g["relation"]}
                      for g in groups]
        if framework == "pog_estimator":
            p, errors = [], []
            if groups:
                p = estimator_scores(question, groups, errors)
                ok = [(s, k) for k, s in enumerate(p) if s is not None]
                record.update(p=[None if s is None else round(s, 4) for s in p], failed=sum(s is None for s in p))
                if ok and max(ok)[0] >= thr:
                    chosen = max(ok)[1]
                    best = groups[chosen]
            verdict_log.judged(framework, candidates, p, chosen, judge_url=estimator.url(),
                               threshold=thr, error=errors[-1] if errors else None, depth=depth, step=step)
        elif oracle is not None:
            right, entries = [], []
            if groups:
                right = [oracle.right(record["depth"], False, "; ".join(g["names"]), g["query"]) for g in groups]
                entries = oracle.log[-len(groups):]
                f1 = [e["f1"] for e in entries]
                take = [k for k in range(len(groups)) if right[k]]
                if take:
                    chosen = max(take, key=lambda k: f1[k])
                    best = groups[chosen]
            verdict_log.golded(framework, candidates, right, entries, chosen, depth=depth, step=step)
        else:
            verdict_log.golded(framework, candidates, [], [], None, depth=depth, step=step,
                               error="no gold: the gold arm has no stop callback for this question")
        record.update(stop=best is not None, answer=(best or {}).get("names", [])[:30], queries=[g["query"] for g in groups])
        _state["decisions"].append(record)
        reply = json.dumps({"A": {"Sufficient": "Yes" if best else "No", "Answer": best["names"] if best else None}})
        return reply, (best["names"] if best else "null"), ("Yes" if best else "No"), {"total": 0, "input": 0, "output": 0}
    return reasoning
