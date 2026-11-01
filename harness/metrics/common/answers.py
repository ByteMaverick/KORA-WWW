"""Each system's answer slot -> one answer set per question (README: reading the answers)."""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

from harness.metrics.groundedness.records import canonical_id
from harness.metrics.reasoning import stated, status

FREE = ("direct_llm",)
#: The literal no-answer reply ("No answer found."), lower case, punctuation dropped: never an answer item.
NO_ANSWER = {"no answer found"}
_SPLIT = re.compile(r"[,;\n]+")


def slot_of(system: str) -> str:
    if system in FREE:
        return "free"
    if system.startswith("sog"):
        return "braces"
    if system.startswith("tog"):
        return "tog"
    if system.startswith("pog"):
        return "field"
    return "list"


@dataclass
class Answer:
    items: list | None                 # None: not scored (the extractor has not run on this row)
    status: str                        # ok, missing, error, over_budget (harness/metrics/reasoning.status)
    how: str = ""                      # list, text, stated, extract, fallback, empty, not run, miss
    ids: list | None = None
    counts: dict = field(default_factory=dict)


def _one(text: str) -> list:
    return [text.strip()] if text and text.strip() else []


def _list(fw: dict, text: str) -> tuple:
    got = fw.get("answer_items")
    if isinstance(got, list):
        return [str(x) for x in got], "list"
    return _one(text), "text"


def fallback_split(text: str) -> list:
    return [x.strip() for x in _SPLIT.split(text or "") if x.strip()]


def _verbatim(item: str, text: str) -> bool:
    return bool(item.strip()) and item.strip() in text


def answer_set(system: str, row: dict | None, extract: dict | None = None) -> Answer:
    """The answer set of one question (module docstring). `extract` is the cached extractor entry for the row."""
    st = status(row)
    if st != "ok":
        return Answer([], st, "miss")
    fw = row.get("fw_metrics") or {}
    text = str(row.get("answer") or "")
    slot = slot_of(system)
    ids = None
    if slot == "list":
        items, how = _list(fw, text)
        raw = (fw.get("kora") or {}).get("answer_ids") if isinstance(fw.get("kora"), dict) else None
        if isinstance(raw, list) and len(raw) == len(items):
            ids = [None if x is None else canonical_id(str(x)) for x in raw]
    elif slot in ("braces", "field"):
        items, how = _list(fw, text)
    elif slot == "tog":
        if text.strip().lower() == "yes":
            reply = re.sub(r"\{\s*yes\s*\}", " ", str(fw.get("tog_reply") or ""), flags=re.I)
            items, how = _one(stated(reply)), "stated"
        else:
            items, how = _one(text), "text"
    else:
        if isinstance(fw.get("answer_items"), list):
            return Answer([str(x) for x in fw["answer_items"]], st, "list")
        if not text.strip():
            return Answer([], st, "empty")
        if " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split()) in NO_ANSWER:
            return Answer([], st, "no answer")        # the harness's literal no-answer reply, never an item
        if extract is None or extract.get("status") not in ("ok", "empty"):
            return Answer(None, st, "not run", counts={"extract error": int(extract is not None)})
        cand = [str(x) for x in extract.get("items") or []]
        kept = [x.strip() for x in cand if _verbatim(x, text)]
        counts = {"extract items dropped (not verbatim)": len(cand) - len(kept),
                  "extract items rejected (cache)": len(extract.get("rejected") or [])}
        if kept:
            return Answer(kept, st, "extract", counts=counts)
        if not cand and not extract.get("rejected"):
            return Answer([], st, "no answer", counts=counts)        # the extractor read no answer in the response
        return Answer(fallback_split(text), st, "fallback", counts=counts)
    return Answer(items, st, how, ids)


def read_cache(path: str) -> tuple:
    """({qid: the last entry}, counts): unparsed lines skipped and counted, prompt hashes counted."""
    out, bad = {}, 0
    if not os.path.exists(path):
        return out, {"cache file": 0}
    for line in open(path, encoding="utf-8", errors="replace"):
        if not line.strip():
            continue
        try:
            e = json.loads(line)
        except ValueError:
            bad += 1
            continue
        if isinstance(e, dict) and e.get("qid") is not None:
            out[str(e["qid"])] = e
        else:
            bad += 1
    hashes = {e.get("prompt_hash") for e in out.values()}
    return out, {"cache file": 1, "unparsed cache lines": bad, "prompt hashes": len(hashes)}
