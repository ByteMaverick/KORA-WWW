"""The support rule's inputs, validated (RecordError names qid and field), entity ids and name resolution."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from harness.dataset import UNSEEN
from harness.metrics.reasoning import label_key as normalize

FREEBASE_NS = "http://rdf.freebase.com/ns/"
MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
          "november", "december"]
#: Month names and three-letter abbreviations (and "sept"), each to its number (also read by support.as_date).
MONTH_NUMBER = {**{m: i + 1 for i, m in enumerate(MONTHS)}, **{m[:3]: i + 1 for i, m in enumerate(MONTHS)},
                "sept": 9}


class RecordError(ValueError):
    """A calls or answers line that breaks the contract."""


@dataclass(frozen=True)
class ResultItem:
    """One thing the graph returned: an entity (`id` set) or a literal (`value` set), never both."""
    id: Optional[str] = None
    label: Optional[str] = None
    type: Optional[str] = None
    value: Optional[str] = None
    prop: Optional[str] = None
    of: Optional[str] = None
    shown: Optional[bool] = None

    @property
    def is_value(self) -> bool:
        return self.value is not None


@dataclass(frozen=True)
class CallRecord:
    """One graph call: what it was about (`anchor`), what it asked (`query`) and what came back (`result`)."""
    qid: str
    seq: int
    anchor: tuple = ()
    query: str = ""
    result: tuple = ()
    source: Optional[str] = None
    client: Optional[str] = None
    endpoint: Optional[str] = None
    relation: Optional[str] = None
    relations: tuple = ()
    n_rows: Optional[int] = None
    truncated: Optional[bool] = None
    llm_call: Optional[int] = None
    unresolved: tuple = ()


@dataclass(frozen=True)
class AnswerRecord:
    """One question's answer items (with ids when given), question, topic entities, and gold (to sort only)."""
    qid: str
    question: str
    items: tuple
    item_ids: tuple
    topic: tuple
    topic_labels: dict = field(default_factory=dict)
    answer_text: str = ""
    gold: tuple = ()
    gold_ids: tuple = ()
    gold_query: Optional[str] = None
    error: Optional[str] = None


def canonical_id(x: Optional[str]) -> Optional[str]:
    """An id in one spelling (full IRI, ns: and bare are all "m.0abc"; a node URI its short id)."""
    if x is None:
        return None
    x = x.strip()
    if x.startswith("<") and x.endswith(">"):
        x = x[1:-1].strip()
    for prefix in (FREEBASE_NS, "ns:"):
        if x.startswith(prefix):
            x = x[len(prefix):]
    if x.startswith("http") and "/node/" in x:
        x = x.rsplit("/node/", 1)[1]
    return x


def _ids(xs) -> tuple:
    return tuple(canonical_id(x) for x in xs)


def _fail(where: str, qid, what: str) -> None:
    raise RecordError(f"{where}: qid={qid}: {what}")


def _str_list(d: dict, key: str, where: str, qid, nullable: bool = False, required: bool = True) -> tuple:
    """The list of strings at d[key] (None entries allowed when nullable), as a tuple."""
    if key not in d:
        if required:
            _fail(where, qid, f"field '{key}' is missing")
        return ()
    v = d[key]
    if v is None and not required:
        return ()
    if not isinstance(v, list) or not all(isinstance(x, str) or (nullable and x is None) for x in v):
        _fail(where, qid, f"field '{key}' must be a list of strings{' or nulls' if nullable else ''}")
    return tuple(v)


def _opt_str(d: dict, key: str, where: str, qid) -> Optional[str]:
    v = d.get(key)
    if v is not None and not isinstance(v, str):
        _fail(where, qid, f"field '{key}' must be a string or null")
    return v


def parse_result_item(d, where: str, qid) -> ResultItem:
    if not isinstance(d, dict):
        _fail(where, qid, "field 'result' entries must be objects")
    has_id, has_value = d.get("id") is not None, d.get("value") is not None
    if has_id == has_value:
        _fail(where, qid, "field 'result' entry must have exactly one of 'id' and 'value'")
    if has_id and (not isinstance(d["id"], str) or not d["id"]):
        _fail(where, qid, "field 'result.id' must be a non-empty string")
    value = d.get("value")
    if has_value:
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            _fail(where, qid, "field 'result.value' must be a string or a number")
        value = str(value)
    shown = d.get("shown")
    if shown is not None and not isinstance(shown, bool):
        _fail(where, qid, "field 'result.shown' must be true, false or null")
    return ResultItem(id=canonical_id(d.get("id")), label=_opt_str(d, "label", where, qid),
                      type=_opt_str(d, "type", where, qid), value=value, prop=_opt_str(d, "prop", where, qid),
                      of=canonical_id(_opt_str(d, "of", where, qid)), shown=shown)


def parse_call(d, where: str, qid: str) -> CallRecord:
    """One calls line of question `qid` (the line's qid must equal it)."""
    if not isinstance(d, dict):
        _fail(where, qid, "a calls line must be an object")
    if d.get("qid") != qid:
        _fail(where, qid, f"field 'qid' is {d.get('qid')!r}, the file is for {qid!r}")
    seq = d.get("seq")
    if not isinstance(seq, int) or isinstance(seq, bool):
        _fail(where, qid, "field 'seq' must be an integer")
    if not isinstance(d.get("query", ""), str):
        _fail(where, qid, "field 'query' must be a string")
    if not isinstance(d.get("result"), list):
        _fail(where, qid, "field 'result' must be a list")
    where = f"{where} seq={seq}"
    return CallRecord(
        qid=qid, seq=seq, anchor=_ids(_str_list(d, "anchor", where, qid)), query=d.get("query", ""),
        result=tuple(parse_result_item(x, where, qid) for x in d["result"]),
        source=_opt_str(d, "source", where, qid), client=_opt_str(d, "client", where, qid),
        endpoint=_opt_str(d, "endpoint", where, qid), relation=_opt_str(d, "relation", where, qid),
        relations=_str_list(d, "relations", where, qid, required=False), n_rows=d.get("n_rows"),
        truncated=d.get("truncated"), llm_call=d.get("llm_call"),
        unresolved=_str_list(d, "unresolved", where, qid, required=False))


_STATED = re.compile(r"answer(?: to (?:the|this) question)? is[:\s]+(.+?)\s*$", re.I | re.S)
_BOLD = re.compile(r"\*\*(.+?)\*\*")


def stated_answer(text: str) -> str:
    """A prose answer (over 12 words) cut to what it states: after its last "answer is", else bold spans."""
    if len(text.split()) <= 12:
        return text
    m = _STATED.search(text)
    if m:
        return m.group(1).replace("**", "").strip().rstrip(".")
    bold = _BOLD.findall(text)
    return "; ".join(b.strip().rstrip(".") for b in bold) if bold else text


KG_OF_SET = {"webqsp": "freebase", "cwq": "freebase", "grailqa": "freebase", "metaqa": "metaqa"}
# The Unseen sets run on the extended graph, whose clone ids keep the graph's id shape (m.x1).
KG_OF_SET.update({u: KG_OF_SET[u.split("-")[0]] for u in UNSEEN})
ID_SHAPE = {"freebase": re.compile(r"^[mg]\.[0-9a-z_]+$"),
            "metaqa": re.compile(r".")}                                            # a MetaQA node's id is its name
STAND_IN = "name:"


def short_id(x: str) -> str:
    """An id as the records write it: no Freebase namespace, a node URI as its short id."""
    x = x.strip().removeprefix(FREEBASE_NS)
    return x.rsplit("/node/", 1)[-1] if x.startswith("http") and "/node/" in x else x


def is_id(x: str, kg: str) -> bool:
    """Whether a string is an entity id of this graph."""
    return bool(ID_SHAPE[kg].match(short_id(x)))


def resolve_name(name: str, topic: dict, seen: dict) -> tuple:
    """(id, None) or (None, why): the topic entity of that name, else an earlier result's; never guessed."""
    n = normalize(name)
    ids = sorted({i for i, label in topic.items() if n and normalize(label or "") == n})
    if ids:                       # every candidate is a topic entity, so the step is rooted whichever one it was
        return ids[0], None
    if not ids:
        found = list(dict.fromkeys(seen.get(n, [])))
        ids = [i for i in found if not i.startswith(STAND_IN)] or found
    if len(ids) == 1:
        return ids[0], None
    return None, (f"ambiguous: {', '.join(ids[:5])}" if ids else "no topic entity or earlier result has this name")
