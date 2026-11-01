"""KB-BINDER (Li et al., ACL 2023): the authors' code in upstream/, on the harness's model and graph."""
from __future__ import annotations

import argparse
import ast
import contextlib
import json
import os
import re
import sys
import time

from SPARQLWrapper import SPARQLWrapper

from baselines.official_llm import chat, complete
from baselines.official_loader import load
from harness.neo4j_client import rejected as _is_rejected
from harness.runner import QueryMetrics

HERE = os.path.dirname(os.path.abspath(__file__))
UPSTREAM = os.path.join(HERE, "upstream")
METAQA_SRC = os.path.join(UPSTREAM, "metaqa_src")
DATA = os.getenv("KBB_DATA", os.path.join(HERE, "data"))
SPLITS = os.path.join(HERE, "..", "..", "data", "questions")
UNSEEN_PAIRS = os.path.join(HERE, "..", "..", "data", "unseen")

TRAIN = {"webqsp": "webqsp_0107.train.json", "grailqa": "grailqa_v1.0_train.json"}


def _table() -> dict:
    from harness.settings import system
    return system("kbbinder_official")


#: The paper's values (section 4.3), from the settings table: exemplars per Freebase set (WebQSP 100,
#: GrailQA 40) and on MetaQA (5; their 1-hop script retrieves 20), and the relation binder's m (10; their code keeps 7)
SHOTS = {d: _table()["shots"][d] for d in TRAIN}
METAQA_SHOTS = _table()["shots"]["metaqa"]
RELATION_CANDIDATES = _table()["relation_candidates"]
#: Every function of theirs that calls Completion.create, and the settings-table temperature it sends
#: (theirs: drafts 0.3, type calls 0)
STEP = {"ep_generator": "temperature_drafts", "type_generator": "temperature_type",
        "two_hop_type_generator": "temperature_type"}
QUERIES = [0]
REJECTED = [0]          # of those, the queries the graph rejected as malformed (they found nothing)
NO_ANSWER = [0]         # a reasoning model's samples with no answer (final channel): empty drafts
FAILURE: list = [None]  # the error of a graph that could not answer, in the question in flight
_state = {"qm": None, "engine": "", "deadline": float("inf")}
_cache: dict = {}


# ── the model ────────────────────────────────────────────────────────────────

class _Completion:
    """harness: their Codex call, answered by the harness's model."""

    @staticmethod
    def create(engine=None, prompt="", temperature=0, max_tokens=256, top_p=1, frequency_penalty=0,
               presence_penalty=0, stop=None, n=1):
        _check_clock()
        _check_graph()
        site = sys._getframe(1).f_code.co_name
        # harness: the step's temperature, the cap, top_p and top_k from the settings table, not the call's
        # own constants
        d = _state["decoding"]
        temperature, cap, top_p, top_k = d[STEP[site]], d["max_tokens"], d["top_p"], d["top_k"]
        stops = [stop] if isinstance(stop, str) else list(stop or [])
        if not _state["engine"].startswith("vertex:"):
            # harness: their call is a Codex text completion, so it is sent as one (n samples in one request, their
            # stop strings); sent as a chat message, the models replied to it and their parser skipped the drafts.
            # A reasoning model (gpt-oss) reasons in a completion too: the stop is not sent, since the server matches
            # it in the reasoning as well, and the draft is the answer (the final channel), cut at their stop.
            reasoning = _state["reasoning"]
            texts, metrics = complete(prompt, _state["engine"], temperature, cap, n=n, stop=[] if reasoning else stops,
                                      top_p=top_p, top_k=top_k)
            if reasoning:
                texts = [_final_channel(t, stops) for t in texts]
                NO_ANSWER[0] += texts.count(None)
                texts = [t or "" for t in texts]
            if _state["qm"] is not None:
                _state["qm"].add(metrics, call_site=site)
            _check_clock()
            return {"choices": [{"text": t} for t in texts]}
        # Vertex (local smoke tests only) has no completion endpoint: a chat call per sample
        # harness: a reasoning model's server matches stop strings against its reasoning too, so there the stop
        # cuts the answer only
        sent_stops = [] if _state["reasoning"] else stops
        choices = []
        for _ in range(n):
            text, metrics = chat(prompt, "", _state["engine"], temperature, cap, stop=sent_stops, top_p=top_p,
                                 top_k=top_k)
            if _state["qm"] is not None:
                _state["qm"].add(metrics, call_site=site)
            for s in stops:
                text = text.split(s)[0]
            choices.append({"text": text})
        return {"choices": choices}


class _OpenAI:
    Completion = _Completion
    api_key = None


#: gpt-oss's header of its answer, <|start|>assistant<|channel|>final<|message|>, as a completion's text holds it: the
#: server leaves the special tokens out of the text (vLLM's skip_special_tokens), so it reads "assistantfinal"
FINAL_CHANNEL = "assistantfinal"


def _final_channel(text: str, stops: list[str]) -> str | None:
    """harness: a reasoning model's completion (reasoning, then answer): the answer cut at their stop."""
    _, header, answer = text.rpartition(FINAL_CHANNEL)
    if not header:
        return None
    for s in stops:
        answer = answer.split(s)[0]
    return answer


def _configure(args: argparse.Namespace) -> None:
    """The question's model and its decoding (the settings table)."""
    from harness.settings import MODELS, decoding, model_of
    model = model_of(args.engine)
    _state.update(engine=args.engine,
                  decoding=decoding(getattr(args, "framework", "") or "kbbinder_official", args.engine),
                  reasoning=bool(model and MODELS[model]["reasoning_effort"]))


_configure(argparse.Namespace(engine=""))


def _check_clock():
    if time.time() >= _state["deadline"]:
        from harness.settings import QuestionTimeoutError
        raise QuestionTimeoutError(f"question_timeout_after_{_state['seconds']}s")


def _sleep(seconds):
    """harness: their `sleep`, which ends the question once its time is up or a graph could not answer."""
    _check_clock()
    _check_graph()
    time.sleep(seconds)


def _new_question() -> None:
    QUERIES[0], REJECTED[0], NO_ANSWER[0], FAILURE[0] = 0, 0, 0, None


def _check_graph():
    """harness: once a graph could not answer, their later queries, model calls and sleeps raise it."""
    if FAILURE[0] is not None:
        raise FAILURE[0]


class _Counted(SPARQLWrapper):
    """harness: their SPARQLWrapper, with the clock, the query count and the graph-error rule."""

    def query(self):
        _check_clock()
        _check_graph()
        QUERIES[0] += 1
        try:
            return super().query()
        except Exception as e:
            if _is_rejected(e):
                REJECTED[0] += 1
            elif FAILURE[0] is None:
                FAILURE[0] = e
            raise


# ── data the authors read ─────────────────────────────────────────────────────

def _held_out(dataset: str) -> set[str]:
    """Every question text of the harness's val and test splits of this dataset, its full test and its Unseen pairs."""
    out = set()
    for folder in os.listdir(SPLITS):
        if folder == dataset or folder.startswith(dataset + "-"):
            for split in ("val", "test", "test_full"):
                path = os.path.join(SPLITS, folder, f"{split}.json")
                if os.path.exists(path):
                    out |= {_plain(q["question"]) for q in json.load(open(path))["questions"]}
    for split in ("test", "validation"):           # both halves of each Unseen pair, as the paper's folders held them
        path = os.path.join(UNSEEN_PAIRS, dataset, f"{split}.json")
        if os.path.exists(path):
            out |= {_plain(p[half]["question"]) for p in json.load(open(path))["pairs"] for half in ("twin", "source")}
    return out


def _plain(question: str) -> str:
    return re.sub(r"\s+", " ", question.replace("[", "").replace("]", "")).strip().lower()


def _pool(dataset: str) -> str:
    """harness: their training file minus the harness's val and test questions, written once."""
    out = os.path.join(DATA, "cache", f"{dataset}_train_pool.json")
    if not os.path.exists(out):
        held = _held_out(dataset)
        rows = [r for r in json.load(open(os.path.join(DATA, TRAIN[dataset]))) if _plain(r["question"]) not in held]
        os.makedirs(os.path.dirname(out), exist_ok=True)
        json.dump(rows, open(out + ".tmp", "w"))
        os.replace(out + ".tmp", out)
    return out


def _workdir() -> str:
    """The folder their relative paths are read from: their indexes and data folders, linked."""
    work = os.path.join(DATA, "work")
    data = os.path.join(UPSTREAM, "data")        # their README: "put them under data/"; sparql_exe reads data/fb_roles
    if os.path.islink(data) and os.readlink(data) != DATA:
        os.remove(data)
    if not os.path.lexists(data):
        os.symlink(DATA, data)
    links = {os.path.join("contriever_fb_relation", "index_relation_fb"):
             os.path.join(UPSTREAM, "contriever_fb_relation", "index_relation_fb"),
             os.path.join("contriever_fb_relation", "freebase_contriever_index", "docid"):
             os.path.join(UPSTREAM, "contriever_fb_relation", "freebase_contriever_index", "docid"),
             os.path.join("contriever_fb_relation", "freebase_contriever_index", "index"):
             os.path.join(DATA, "contriever", "index"),
             os.path.join("data", "metaQA"): os.path.join(DATA, "metaQA"),
             os.path.join("data", "data", "metaQA"): os.path.join(DATA, "metaQA")}
    for rel, target in links.items():
        path = os.path.join(work, rel)
        if not os.path.lexists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            os.symlink(target, path)
    # their main() reads an evaluation file and a surface map; the harness passes one question at a time and links
    # with the gold topic entities, so these are an empty list and a one-line map (their BM25 needs one name)
    for name, text in (("empty.json", "[]"), ("placeholder_surface_map", "kbbinder harness placeholder\t0\tm.0\n")):
        path = os.path.join(work, name)
        if not os.path.exists(path):
            open(path, "w").write(text)
    return work


@contextlib.contextmanager
def _cwd(path: str):
    here = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(here)


def _main_block(path: str, kind):
    """Their file's main() body (Freebase) or `if __name__ == "__main__":` body (MetaQA), as statements."""
    src = open(path).read()
    tree = ast.parse(src)
    if kind == "main":
        return src, next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main").body
    return src, next(n for n in tree.body if isinstance(n, ast.If)).body


# ── the paper's values where their code hard-codes others ──────────────────────────────────────────

def _paper_m(src: str, path: str, m: int):
    """harness: their bound_to_existed, compiled from their file with the paper's m in place of its two 7s."""
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "bound_to_existed")
    sevens = [n for n in ast.walk(fn) if isinstance(n, ast.Constant) and n.value == 7]
    assert len(sevens) == 2, "their relation binder keeps 7 candidates, in two places (few_shot_kbqa.py:318, 327)"
    for n in sevens:
        n.value = m
    return compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), path, "exec")


def _paper_shots(src: str, path: str, shots: int):
    """harness: their MetaQA question-type call with the paper's shots as its get_top_n n; (name, code)."""
    def retrieval(node):
        return isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "get_top_n" and \
            getattr(node.func.value, "id", "") == "bm25_train_full"
    fns = [n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and any(map(retrieval, ast.walk(n)))]
    assert len(fns) == 1, "one function of their script retrieves the exemplars"
    calls = [c for c in ast.walk(fns[0]) if retrieval(c)]
    assert len(calls) == 1 and [k.arg for k in calls[0].keywords] == ["n"], "with one get_top_n(..., n=...) call"
    calls[0].keywords[0].value = ast.Constant(shots)
    return fns[0].name, compile(ast.fix_missing_locations(ast.Module(body=[fns[0]], type_ignores=[])), path, "exec")


# ── Freebase: few_shot_kbqa.py ───────────────────────────────────────────────

def _freebase(dataset: str):
    if dataset in _cache:
        return _cache[dataset]
    from SPARQLWrapper import JSON

    work = _workdir()
    with _cwd(work):
        m = load(UPSTREAM, ["utils", "sparql_exe", "few_shot_kbqa"])   # sparql_exe reads data/fb_roles at import
    fsk, sx = m["few_shot_kbqa"], m["sparql_exe"]
    fsk.openai, fsk.sleep = _OpenAI, _sleep
    path = os.path.join(UPSTREAM, "few_shot_kbqa.py")
    src, body = _main_block(path, "main")
    assert ast.get_source_segment(src, body[0]) == "args = parse_args()", "their main() starts by parsing args"
    assert ast.get_source_segment(src, body[-1]).startswith("all_combiner_evaluation("), "and ends by evaluating"
    # harness: their main() setup with the arguments of the paper's -R run; the surface map is replaced (see above)
    args = argparse.Namespace(shot_num=SHOTS[dataset], temperature=_state["decoding"]["temperature_drafts"],
                              api_key=None, engine="harness",
                              retrieval=True, train_data_path=_pool(dataset),
                              eva_data_path=os.path.join(work, "empty.json"),
                              fb_roles_path=os.path.join(DATA, "fb_roles"),
                              surface_map_path=os.path.join(work, "placeholder_surface_map"))
    ns = fsk.__dict__
    ns["args"] = args
    with _cwd(work):
        exec(compile(ast.Module(body=body[1:-1], type_ignores=[]), path, "exec"), ns)
    fixed = _name_exemplar_mids(ns["que_to_s_dict_train"], ns["question_to_mid_dict"])     # harness: exemplar MIDs their table leaves unnamed
    if any(fixed.values()):
        print(f"  [kbbinder_official] {dataset} exemplars naming a MID their entry leaves unnamed: named from "
              f"their file {fixed['named']}; left out (no name) {fixed['left_out']}")
    exec(_paper_m(src, path, RELATION_CANDIDATES), ns)      # harness: the paper's m
    # their all_combiner_evaluation, plus one statement: return the answer it computed
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "all_combiner_evaluation")
    fn.name = "_harness_all_combiner_evaluation"
    fn.body.append(ast.parse('return locals().get("answer"), locals().get("answer_to_grounded_dict") or {}').body[0])
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), path, "exec"), ns)
    sx.sparql = _Counted(os.getenv("SPARQL_ENDPOINT", "http://localhost:8890/sparql"))
    sx.sparql.setReturnFormat(JSON)
    _cache[dataset] = (fsk, sx, args, _Counted)
    return _cache[dataset]


def _name_exemplar_mids(que_to_s: dict, question_to_mid: dict) -> dict:
    """harness: name an exemplar MID their table lacks as their file names it elsewhere, else leave the exemplar out."""
    names = {}
    for table in question_to_mid.values():
        for mid, name in table.items():
            if name:
                names.setdefault(mid, name)
    named, left_out = {}, {}
    for que, s_expression in que_to_s.items():
        for seg in (s_expression or "").split():
            mid = seg.strip(")(")
            if not (seg.startswith("m.") or seg.startswith("g.")) or mid in question_to_mid[que]:
                continue
            if mid not in names:
                que_to_s[que] = ""
                left_out[que] = mid
                named.pop(que, None)
                break
            question_to_mid[que][mid] = names[mid]
            named[que] = mid
    return {"named": named, "left_out": left_out}


def _linking(topic_entity: dict) -> dict:
    """harness: the gold topic entities as their name -> {MID: score} table."""
    table = {}
    for mid, name in (topic_entity or {}).items():
        name = str(name or "").strip().lower()
        if name and name != str(mid).lower():
            table.setdefault(name, {})[mid] = 1.0
    return table


def run_freebase(sample, dataset: str):
    from rank_bm25 import BM25Okapi
    fsk, sx, args, counted = _freebase(dataset)
    args.temperature = _state["decoding"]["temperature_drafts"]     # harness: the settings table's, as _Completion sends
    sx.sparql = counted(os.getenv("SPARQL_ENDPOINT", sx.sparql.endpoint))
    sx.sparql.setReturnFormat("json")
    ns = fsk.__dict__
    name_to_id_dict = _linking(sample.topic_entity)
    all_fns = list(name_to_id_dict.keys())
    bm25_all_fns = BM25Okapi([fn.split() for fn in all_fns]) if all_fns else None
    data = {"id": sample.qid, "question": sample.question, "sparql_query": "", "s_expression": "", "answer": []}
    with _cwd(_workdir()):
        answer, grounded = ns["_harness_all_combiner_evaluation"](
            [data], ns["selected_quest_compare"], ns["selected_quest_compose"], ns["selected_quest"],
            ns["prompt_type"], ns["hsearcher"], ns["rela_corpus"], ns["relationships"], args.temperature,
            ns["que_to_s_dict_train"], ns["question_to_mid_dict"], args.api_key, args.engine, name_to_id_dict,
            bm25_all_fns, all_fns, ns["relationship_to_enti"], retrieval=args.retrieval, corpus=ns["corpus"],
            nlp_model=ns["nlp"], bm25_train_full=ns["bm25_train_full"], retrieve_number=args.shot_num)
    values = list(answer or [])
    names = []
    for v in values:
        try:
            names.append(sx.get_friendly_name(v) if re.match(r"^[mg]\.", v) else v)
        except (Exception, SystemExit):            # their helpers exit(0) when the endpoint is unreachable
            names.append(v)
    return names, {"kbb_values": values[:50], "kbb_logical_form": (grounded or {}).get(tuple(values), "")}


# ── MetaQA: metaqa_src/metaqa_{k}hop_retrieval.py ───────────────────────────

#: An entity's fact lines, by (graph, database, entity), read once per process: their enti_to_fact_dict is a dict
#: built once from kb.txt, so a 3-hop question through a high-degree node made one round trip per lookup
_FACTS: dict = {}


class _Facts:
    """harness: their enti_to_fact_dict (kb.txt lines by entity), read from the graph, each entity once per run."""

    def __getitem__(self, ent):
        from harness import neo4j_client
        _check_clock()
        key = (neo4j_client.NEO4J_URI, neo4j_client.NEO4J_DATABASE, ent)
        if key not in _FACTS:
            QUERIES[0] += 1
            rows = neo4j_client.run_cypher(
                "MATCH (n:Entity {id: $id})-[r]->(m:Entity) RETURN n.id AS s, type(r) AS r, m.id AS o "
                "UNION ALL MATCH (n:Entity {id: $id})<-[r]-(m:Entity) RETURN m.id AS s, type(r) AS r, "
                "n.id AS o", {"id": ent})
            _FACTS[key] = [f"{x['s']}|{x['r'].lower()}|{x['o']}" for x in rows]
        if not _FACTS[key]:
            raise KeyError(ent)
        return list(_FACTS[key])


def _metaqa(hop: int):
    if hop in _cache:
        return _cache[hop]
    name = f"metaqa_{hop}hop_retrieval"
    path = os.path.join(METAQA_SRC, name + ".py")
    work = _workdir()
    with _cwd(work):
        mod = load(METAQA_SRC, ["bm25_trial", name])[name]
    mod.openai, mod.sleep = _OpenAI, _sleep
    src, body = _main_block(path, "if")
    loop = next(n for n in body if isinstance(n, ast.For) and getattr(n.target, "id", "") == "ques_dict")
    test_var = loop.iter.id                # their loop over the test questions: for ques_dict in test_question_<k>hop
    # harness: their setup without reading kb.txt (the facts come from the graph)
    setup = [n for n in body[:body.index(loop)] if "kb.txt" not in ast.get_source_segment(src, n)]
    ns = mod.__dict__
    ns.setdefault("api_key", None)
    with _cwd(work):
        exec(compile(ast.Module(body=setup, type_ignores=[]), path, "exec"), ns)
    ns["enti_to_fact_dict"] = _Facts()
    _, shots = _paper_shots(src, path, METAQA_SHOTS)         # harness: the paper's 5 exemplars
    exec(shots, ns)
    ns.setdefault("ent", None)            # their 3-hop loop reads `ent` from the previous question when a draft fails
    # harness: the harness's val and test questions out of the exemplar pool, the pool rebuilt as their code builds it
    held = _held_out("metaqa")
    from rank_bm25 import BM25Okapi
    ns["corpus"] = [q for q in ns["corpus"] if _plain(q) not in held]
    ns["bm25_train_full"] = BM25Okapi([doc.split() for doc in ns["corpus"]])
    code = compile(ast.Module(body=[loop], type_ignores=[]), path, "exec")
    _cache[hop] = (mod, code, test_var)
    return _cache[hop]


def _bracket(question: str, name: str) -> str:
    """The paper's MetaQA input marks the topic entity: "who directed [Kismet]"."""
    i = question.lower().find(name.lower())
    if i < 0:
        return f"{question} [{name}]"
    return question[:i] + f"[{name}]" + question[i + len(name):]


def run_metaqa(sample):
    hop = int(re.search(r"(\d)hop", sample.qid).group(1))
    mod, code, test_var = _metaqa(hop)
    ent = next(iter(sample.topic_entity))
    ns = mod.__dict__
    ns[test_var] = [{"question": _bracket(sample.question, ent), "retrieved_ent": ent, "answer": []}]
    ns.pop("pred", None)
    ns.pop("set_pred", None)
    ns["total"], ns["correct"] = 0, 0
    exec(code, ns)
    pred = ns.get("set_pred")
    if pred is None:
        pred = set(ns.get("pred") or [])
    return sorted(pred), {"kbb_hop": hop, "kbb_question_type": ns.get("question_type")}


# ── the harness's entry point ────────────────────────────────────────────────

def _untimed(setup, key) -> float:
    """harness: their one-off setup with the alarm paused; returns its seconds (0 when it already ran)."""
    if key in _cache:
        return 0.0
    import signal
    left = signal.alarm(0)
    started = time.perf_counter()
    try:
        setup(key)
    finally:
        if left:
            signal.alarm(left)
    return time.perf_counter() - started


def run_kbbinder_official(sample, args: argparse.Namespace, output_path: str | None = None):
    kg = getattr(args, "kg_name", "") or "freebase"
    dataset = re.sub(r"-(val|test|train)$", "", getattr(args, "dataset_benchmark", "") or "")
    dataset = re.sub(r"-(cf|anon|unseen).*$", "", dataset)   # a copy's or an Unseen set's: its source benchmark's
    _configure(args)
    _new_question()                                  # before the setup: the last question's failure is not this one's
    setup_seconds = 0.0
    if kg == "metaqa":
        setup_seconds = _untimed(_metaqa, int(re.search(r"(\d)hop", sample.qid).group(1)))
    elif kg == "freebase" and dataset in SHOTS:
        setup_seconds = _untimed(_freebase, dataset)
    _state["setup_seconds"] = setup_seconds          # also readable when the question then times out (no summary)
    from harness.settings import untimed
    untimed["seconds"] += setup_seconds              # the harness takes it out of the question's time
    qm = QueryMetrics(question=sample.question)
    seconds = int(getattr(args, "question_timeout_seconds", 0) or 200)
    _state.update(qm=qm, seconds=seconds, deadline=time.time() + seconds)
    if kg == "metaqa":
        names, extra = run_metaqa(sample)
    elif kg == "freebase" and dataset in SHOTS:
        names, extra = run_freebase(sample, dataset)
    else:
        raise NotImplementedError(f"[kbbinder_official] {dataset or kg}: KB-BINDER's code covers GrailQA, WebQSP "
                                  "(and GraphQ) on Freebase and MetaQA")
    _check_graph()        # harness: a graph that could not answer, swallowed by their except, is the row's error
    _check_clock()        # harness: an answer after the question's time is a timeout, as for every system
    answer = "; ".join(names)
    summary = qm.summary()
    # harness: the one-off setup's seconds, outside the question's 200 s; 0 when it ran for an earlier question
    summary.update({"graph_queries": QUERIES[0], "graph_rejected": REJECTED[0], "setup_seconds": round(setup_seconds, 3),
                    "kbb_samples_without_answer": NO_ANSWER[0], **extra})   # a reasoning model's empty drafts
    print(f"  KB-BINDER (official) {dataset or kg}: {summary.get('num_llm_calls')} calls -> {answer[:200]}")
    return list(names), summary   # a list: the harness scores its items
