"""Search-on-Graph (Sun et al., arXiv:2510.08825), the authors' code from the OpenReview supplementary material,
vendored unchanged in upstream/ (UPSTREAM.md: source and checksums). One question is their process_single_question
(upstream/main.py): their system prompt and exemplars, their tool, their loop of up to 50 model turns (the code's;
the paper states none), their Qwen3 sampling (Instruct: temperature 0.7, top_p 0.8, top_k 20, presence_penalty 1.5,
max_tokens 20,480), their stop (a reply with no tool call is the answer).

What the harness changes (marked "harness:" below):
  * the five exemplars: their get_system_prompt samples num_examples - 5 of them, so NUM_EXAMPLES = 10 gives the
    paper's five; their order is seeded per question;
  * the model: the harness's server through the same OpenAI client call, every turn with the decoding of
    harness/settings.py (on gpt-oss, where their code sends none, the OpenAI defaults stated and sent); a request
    whose prompt plus 20,480 output tokens exceeds the server's context is resent with the room that is left;
  * the tool's row cap: the paper's p (1,000), applied to every tool result on its way into their table
    (_table_of_p_rows), after their code's own 2,000;
  * the graph: SPARQL_ENDPOINT for Freebase instead of their localhost:8890; MetaQA through graphs.py, with its tool
    spec and exemplars from extra.py (exemplars built from train questions, build_examples.py);
  * the gold never reaches their code (their readers get a placeholder for their exact-match printout);
  * the answer: the entities in curly braces in their final reply (their own extraction); the reply when it has none;
  * the clock: their tool calls run in worker threads inside `with ThreadPoolExecutor`, whose exit waits for a
    pending graph query; when the harness's timeout ends the question, it no longer waits;
  * graph errors (graphs.py): a query the graph rejects finds nothing and is counted; a graph that cannot answer is
    never handed to the model as a tool result (their loop would turn it into a message the model answers around):
    no further model turn is sent and the failure is raised, so the question is an error;
  * a tool call the model writes as text, as their exemplars do, is executed as the native call it stands for
    (_as_native_tool_call).

Their prompt lets the model answer from its own knowledge when the graph does not give enough; that is kept. An
answer written before any tool call (sog_tool_calls == 0) is the route without the graph.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import re
import time
from types import SimpleNamespace

from baselines.sog import upstream_path  # noqa: F401
import main as sog                                   # the authors' main.py
import prompt as sog_prompt
import few_shot_llm_tools_unified_helper as helper

from baselines.sog import extra, graphs
from baselines.official_sampling import openai_kwargs
from harness.llm import LLMCallMetrics, trace_chat
from harness.runner import QueryMetrics

NUM_EXAMPLES = 10            # harness: their get_system_prompt takes num_examples - 5 exemplars; 10 -> the paper's 5
THEIRS = {"cwq": "CWQ", "webqsp": "WebQSP", "grailqa": "GrailQA"}
PLACEHOLDER = "__scored_by_the_harness__"


# ── the authors' readers and prompt, extended to the graphs they did not run (installed once, at import) ──────
_their_system_prompt = sog.get_system_prompt
_their_question_text, _their_question_id, _their_gold = sog.extract_question_text, sog.get_question_id, sog.extract_gold_answers


def _system_prompt(dataset, num_examples=0):
    if dataset not in extra.DATASETS.values():
        return _their_system_prompt(dataset, num_examples)
    # harness: the body of their get_system_prompt, for a graph they did not run
    msg = sog_prompt.prompt["main"] + "" + extra.instructions(dataset)
    msg += ("\nBelow are examples of how to correctly make tool calls, interpret the tool call results, and produce "
            "the final answer(s).\n" + "\n\n".join(random.sample(extra.examples(dataset), k=num_examples - 5)) + "\n--")
    return {"role": "system", "content": msg}


def _question_text(dataset, q_item):
    return q_item.get("question", "") if dataset in extra.DATASETS.values() else _their_question_text(dataset, q_item)


def _question_id(dataset, q_item, idx):
    return q_item.get("id", idx) if dataset in extra.DATASETS.values() else _their_question_id(dataset, q_item, idx)


def _gold(dataset, q_item, idx):
    return [PLACEHOLDER] if dataset in extra.DATASETS.values() else _their_gold(dataset, q_item, idx)


sog.get_system_prompt, sog.extract_question_text = _system_prompt, _question_text
sog.get_question_id, sog.extract_gold_answers = _question_id, _gold


class _Executor(concurrent.futures.ThreadPoolExecutor):
    """harness: their ThreadPoolExecutor for the tool calls, except that when the harness's alarm (a BaseException,
    harness.settings.QuestionTimeoutError) leaves the block, it does not wait for the tool calls still running."""

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None and not issubclass(exc_type, Exception):
            self.shutdown(wait=False, cancel_futures=True)
            return False
        return super().__exit__(exc_type, exc, tb)


sog.concurrent = SimpleNamespace(futures=SimpleNamespace(ThreadPoolExecutor=_Executor,
                                                         as_completed=concurrent.futures.as_completed))


# ── the paper's p (the paper-vs-code rule) ─────────────────────────────────────────────────────────────────────────
def _settings() -> dict:
    from harness.settings import system
    return system("sog_native")


#: The paper's maximum result size (v1 §5.1, "the maximum result size p = 1000"), from the settings table
#: (harness/settings.SYSTEMS); their code keeps 2,000 rows (few_shot_llm_tools_unified_helper.py:319, top_k = 2000)
MAX_ROWS = _settings()["max_rows"]
_their_table = helper.convert_tool_result_to_table


def _table_of_p_rows(tool_content):
    """harness: their convert_tool_result_to_table, given at most the paper's p rows. Every tool result reaches the
    model through it, after their own cut at 2,000 (Freebase: their function; the other graphs: graphs.py), so the
    model sees the first p rows, as their code with top_k = p would show it."""
    content = json.loads(tool_content) if isinstance(tool_content, str) else tool_content
    rows = content["results"]["bindings"]
    if len(rows) > MAX_ROWS:
        content = {**content, "results": {**content["results"], "bindings": rows[:MAX_ROWS]}}
    return _their_table(content)


helper.convert_tool_result_to_table = _table_of_p_rows


# ── the model ──────────────────────────────────────────────────────────────────────────────────────────────────
_CONTEXT = re.compile(r"maximum context length is (\d+) tokens.*?(\d+) input tokens", re.S)


def _endpoint(engine: str) -> tuple[str, str]:
    return (os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE") or "http://localhost:8000/v1",
            os.getenv("OPENAI_API_KEY") or "EMPTY")


class _Client:
    """The `client.chat.completions.create` the authors call, answered by the harness's model server."""

    def __init__(self, engine: str, qm: QueryMetrics):
        from openai import OpenAI
        base, key = _endpoint(engine)
        self.engine, self.qm, self.provider = engine, qm, "local"
        self._openai = OpenAI(base_url=base, api_key=key, max_retries=2)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
        self.text_tool_calls = 0          # replies whose tool call was written as text and executed

    def create(self, **kw):
        kw.pop("timeout", None)                       # their 3,600 s per call; the harness's 200 s per question rules
        kw = openai_kwargs("sog_native", self.engine, kw)   # harness: the settings table's decoding, every model
        from harness.settings import reasoning_effort
        effort = reasoning_effort(self.engine)        # harness: every framework on a run reasons equally hard
        if effort:
            kw["reasoning_effort"] = effort
        started = time.time()
        try:
            try:
                completion = self._openai.chat.completions.create(**kw)
            except Exception as error:               # harness: the server's context is smaller than the paper's
                room = _CONTEXT.search(str(error))
                if not room or "max_tokens" not in kw:
                    raise
                left = int(room.group(1)) - int(room.group(2)) - 16
                if left < 64:
                    raise
                kw["max_tokens"] = left
                completion = self._openai.chat.completions.create(**kw)
        except Exception as error:
            self._count(kw, started, None, str(error)[:300])
            raise
        self._count(kw, started, getattr(completion, "usage", None), None, completion)
        if kw.get("tools") and _as_native_tool_call(completion, kw["tools"]):
            self.text_tool_calls += 1
        return completion

    def _count(self, kw, started, usage, error, completion=None):
        prompt = getattr(usage, "prompt_tokens", 0) or 0
        out = getattr(usage, "completion_tokens", 0) or 0
        metrics = LLMCallMetrics(engine=self.engine, provider=self.provider, latency_seconds=time.time() - started,
                                 prompt_tokens=prompt, completion_tokens=out, total_tokens=prompt + out,
                                 temperature=kw.get("temperature", 0.0) or 0.0, max_tokens=kw.get("max_tokens", 0) or 0,
                                 success=error is None, error=error)
        self.qm.add(metrics, call_site="sog_turn")
        trace_chat(self.engine, getattr(self._openai, "base_url", ""), kw.get("messages"), completion, metrics)   # harness: the trace


_TEXT_CALL = re.compile(r"Tool Call:\s*(\[.*?\])\s*$", re.M)


def _as_native_tool_call(completion, tools: list) -> bool:
    """harness: their exemplars write each tool call as text ("Tool Call: [{...}]", then
    "Suppose it returns:" an imagined table, prompt.py:78-113), and their loop ends on any reply without a native
    tool call (main.py:416-418). A reply with no native call that writes one the way the exemplars do has its FIRST
    written call executed as the native call it stands for; the rest of the reply (an imagined result and an answer
    from it) is dropped. No hint or extra turn is added: the loop then goes on exactly as after a native call. A
    reply with neither stays their final answer. Returns whether a call was converted (counted per row)."""
    msg = completion.choices[0].message
    if msg.tool_calls or not msg.content:
        return False
    found = _TEXT_CALL.search(msg.content)
    if not found:
        return False
    try:
        call = json.loads(found.group(1))[0]
        name, args = call["function"]["name"], call["function"]["arguments"]
    except (ValueError, LookupError, TypeError):
        return False
    if name not in {t["function"]["name"] for t in tools}:
        return False
    from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall, Function
    msg.tool_calls = [ChatCompletionMessageToolCall(
        id=f"text_call_{abs(hash(found.group(1))) % 10**8}", type="function",
        function=Function(name=name, arguments=args if isinstance(args, str) else json.dumps(args)))]
    msg.content = msg.content[:found.start()].rstrip()
    return True


def _stops_on_graph_failure(client):
    """harness: the client their loop calls, which sends no model turn once a graph could not answer (graphs.py):
    the failure is raised instead, so the model never reads it as a tool result and answers around it."""
    def create(**kw):
        if graphs.failure() is not None:
            raise graphs.failure()
        return client.chat.completions.create(**kw)
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


# ── one question ───────────────────────────────────────────────────────────────────────────────────────────────
def dataset_of(args: argparse.Namespace, sample) -> str:
    """The authors' dataset name for this question's benchmark (cwq-test -> CWQ; metaqa-test -> MetaQA;
    webqsp-unseen-real-test -> WebQSP: an Unseen set's twins and real sources are its source benchmark's questions)."""
    name = (getattr(args, "dataset_benchmark", "") or getattr(sample, "benchmark", "") or "").lower()
    name = re.sub(r"-(test|val|train)$", "", name)
    base = re.sub(r"-unseen(-real)?$", "", name)
    if base in THEIRS:
        return THEIRS[base]
    if base in extra.DATASETS:
        return extra.DATASETS[base]
    raise NotImplementedError(f"[sog] no SoG setup for benchmark {name!r}")


def question_text(sample) -> str:
    """The question as the authors' data writes it: the text, then {label: id} of its topic entities."""
    topic = {label: eid for eid, label in (sample.topic_entity or {}).items()}
    return f"{sample.question} {repr(topic)}" if topic else sample.question


def q_item(dataset: str, sample) -> dict:
    text, qid = question_text(sample), str(sample.qid)
    if dataset == "CWQ":
        return {"ID": qid, "question": text, "answers_processed": [{"x_label": {"value": PLACEHOLDER}}]}
    if dataset == "WebQSP":
        return {"QuestionId": qid, "ProcessedQuestion": text,
                "Parses": [{"AnswersProcessed": [{"x": {"type": "literal", "value": PLACEHOLDER}}]}]}
    if dataset == "GrailQA":
        return {"qid": qid, "question": text, "answer": [{"answer_type": "Value", "answer_argument": PLACEHOLDER}]}
    return {"id": qid, "question": text}


def answer_of(reply: str) -> str | list:
    braces = helper.extract_answers_inside_curly_braces(reply or "")
    return list(dict.fromkeys(b for b in braces if b)) if braces else (reply or "").strip()   # braces: a list


def run_sog(sample, args: argparse.Namespace):
    dataset = dataset_of(args, sample)
    kg = getattr(args, "kg_name", "") or ("freebase" if dataset in THEIRS.values() else "")
    qm = QueryMetrics(question=sample.question)
    graphs.reset()
    imports = sog.get_imports_for_dataset(dataset if dataset in THEIRS.values() else "CWQ", "markdown-short")
    imports["get_adjacent_fn"] = graphs.get_adjacent_fn(kg)          # harness: the harness's graph
    if dataset not in THEIRS.values():
        imports["tools"] = extra.tools(dataset)
    random.seed(f"sog:{sample.qid}")                                 # harness: the exemplars' order, per question
    inner = _Client(args.engine, qm)
    client = _stops_on_graph_failure(inner)
    out = sog.process_single_question(dataset, q_item(dataset, sample), 0, 1, client, imports["tools"], args.engine,
                                      False, NUM_EXAMPLES, True, imports, io_mode=False) or {}
    if graphs.failure() is not None:      # harness: a graph that could not answer ends the question as an error
        raise graphs.failure()
    reply = out.get("model_answer") or ""
    answer = answer_of(reply)
    history = out.get("conversation_history") or []
    turns = sum(1 for m in history if m.get("role") == "assistant")
    # the tool calls the model made: none means the answer is its first reply, written without the graph
    tool_calls = sum(len(m.get("tool_calls") or []) for m in history if m.get("role") == "assistant")
    summary = qm.summary()
    summary.update({"graph_queries": graphs.queries(), "graph_rejected": graphs.rejected(),
                    "sog_turns": turns, "sog_tool_calls": tool_calls, "sog_text_tool_calls": getattr(inner, "text_tool_calls", 0),
                    "sog_reply": reply[:2000],
                    "sog_exhausted": bool(history) and not reply, "sog_dataset": dataset})
    if out.get("error"):
        summary["error"] = out["error"]
    print(f"  SoG (official) {dataset}: {turns} turns, {graphs.queries()} graph calls -> {str(answer)[:200]}")
    return answer, summary
