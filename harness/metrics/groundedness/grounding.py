"""Graph-trace support on the same answer sets as correctness; no model, no gold (README: graph-trace support)."""
from __future__ import annotations

import ast
import dataclasses
import os
import re
from collections import Counter

from harness.metrics.groundedness import hook, support
from harness.metrics.groundedness.records import KG_OF_SET, AnswerRecord, canonical_id, is_id, resolve_name
from harness.metrics.reasoning import contains, label_key, match, normalize
from harness.metrics.reasoning import harness_golds as golds_of

NO_BODY_KGS = {"metaqa"}
#: Systems that make no graph calls: no graph log is expected of them.
NO_GRAPH_LOG = {"direct_llm"}
SLACK = 5.0
RETURNED = "returned (logged)"


def needs_graph_log(system: str) -> bool:
    return system not in NO_GRAPH_LOG


def safe_qid(qid: str) -> str:
    """As in harness/llm._trace_call."""
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in (qid or "no-qid"))


def log_path(cell_dir: str, shard, qid: str) -> str:
    return os.path.join(cell_dir, "traces", f"s{shard}", f"{safe_qid(qid)}.graph.jsonl")


def last_attempt(lines: list, wall) -> tuple:
    """(the calls of the row's own attempt, the number dropped): ts >= the last ts - wall_time_s - SLACK."""
    stamped = [l.get("ts") for l in lines if isinstance(l.get("ts"), (int, float))]
    if wall is None or not stamped:
        return lines, 0
    since = max(stamped) - float(wall) - SLACK
    kept = [l for l in lines if not isinstance(l.get("ts"), (int, float)) or l["ts"] >= since]
    return kept, len(lines) - len(kept)


def _lit(v, kind):
    """A split field that may be written as a Python literal string."""
    if isinstance(v, str):
        try:
            v = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            v = [v] if kind is list else {}
    return kind(v) if isinstance(v, (list, tuple, dict)) else kind()


def topic_of(q: dict) -> dict:
    """{topic id: name} of a split question."""
    return {str(k): str(v) for k, v in _lit(q.get("topic_entity"), dict).items()}


def is_kora(system: str) -> bool:
    return system.startswith("kora")


def why_not(system: str, set_name: str, cell) -> str | None:
    metaqa = KG_OF_SET[set_name] in NO_BODY_KGS
    if metaqa and is_kora(system):
        return "not measurable: neo4j logs no rows"
    return None


_NAME_LITERAL = re.compile(hook._LABEL_PREDICATE + r'\s+"((?:[^"\\]|\\.)*)"')
_LABEL_VAR = re.compile(r"(?:\?\w+|<[^<>\s]+>|\bns:[\w.]+)\s+" + hook._LABEL_PREDICATE + r"\s+\?(\w+)")
_FILTER_EQ = re.compile(r'\?(\w+)\s*\)*\s*=\s*"((?:[^"\\]|\\.)*)"')


def query_names(query: str) -> list:
    """The names a query writes for an entity: label-predicate strings, or a FILTER on a label variable."""
    names = _NAME_LITERAL.findall(query)
    label_vars = set(_LABEL_VAR.findall(query))
    names += [n for v, n in _FILTER_EQ.findall(query) if v in label_vars]
    return list(dict.fromkeys(n for n in names if n.strip()))


def name_anchored(calls: list, topic_names: dict) -> tuple:
    """(the calls with name anchoring applied, how many it anchored): calls whose every name resolves."""
    seen, out, n = {}, [], 0
    for c in sorted(calls, key=lambda c: c.seq):
        names = query_names(c.query) if not c.anchor else []
        got = [resolve_name(x, topic_names, seen)[0] for x in names]
        if names and all(got):
            c = dataclasses.replace(c, anchor=tuple(dict.fromkeys(got)))
            n += 1
        out.append(c)
        for r in c.result:
            if not r.is_value and r.label:
                seen.setdefault(label_key(r.label), []).append(r.id)
    return out, n


def _kept(recs: list) -> list:
    return [dataclasses.replace(r, query="") for r in recs if r.result]


def calls_of(system: str, row: dict, shard, cell_dir: str, kg: str, counts: Counter, topic_names=None) -> tuple:
    """(a row's final-attempt calls records, the same name-anchored or None); empty results dropped."""
    lines, bad = hook.read_lines(log_path(cell_dir, shard, str(row["qid"])))
    counts["repeated seq"] += hook.repeated_seqs(lines)
    counts["unparsed log lines"] += bad
    lines, dropped = last_attempt(lines, row.get("wall_time_s"))
    counts["calls dropped (earlier attempts)"] += dropped
    queried = (row.get("fw_metrics") or {}).get("graph_queries") or 0
    counts["graph rows without a log"] += bool(needs_graph_log(system) and queried and not lines)
    recs = hook.records_of(lines, kg)
    named, n = name_anchored(recs, topic_names) if topic_names is not None else (recs, 0)
    counts["calls anchored by name (4a)"] += n
    return _kept(recs), (_kept(named) if n else None)


def grounded_hit(question: str, items: list, golds: list, gold_ids: list, topic, calls) -> float:
    named = [g for g in golds if any(match(i, g) for i in items)]
    if not named:
        return 0.0
    verdicts = support.item_verdicts(question, named, [None] * len(named), topic, calls)
    rooted = {r.id for c in support.rooted_records(calls, topic) for r in c.result if not r.is_value}
    by_id = len(gold_ids) == 1 and bool(set(gold_ids) & rooted)
    return float(by_id or any(v == support.STRICT for v, _ in verdicts))


def near_misses(question_items: list, answer: AnswerRecord, calls: list) -> int:
    """Items neither strict, yes/no nor type-rejected that match a rooted result either way (a diagnostic)."""
    cand = [i["item"] for i in question_items if not (i["strict"] or i["yes_no"] or i["type_rejected"])]
    if not cand:
        return 0
    rooted = support.rooted_records(support.with_labels(calls), answer.topic)
    texts = {(r.value if r.is_value else r.label) for c in rooted for r in c.result} - {None, ""}
    texts = [(t, normalize(t)) for t in texts]
    out = 0
    for it in cand:
        n = normalize(it)
        out += any(contains(n, tn) or contains(tn, n) for t, tn in texts)
    return out


def cell_support(system: str, set_name: str, cell, scores: dict, name_anchor: bool = True) -> dict:
    """The support rate and its companions of one cell; `scores` is reasoning.cell_scores of the same cell."""
    why = why_not(system, set_name, cell)
    if why:
        return {"support": None, "why": why, "counts": {}}
    kg = KG_OF_SET[set_name]
    answers, calls, alt, gh, counts, kinds = {}, {}, {}, [], Counter(), Counter()
    for qid, q in cell.questions.items():
        row, source, shard, cell_dir = cell.rows[qid]
        p = scores["per"][qid]
        if not p["scored"]:
            counts["not scored yet"] += 1
            continue
        topic_names = {canonical_id(t): n for t, n in topic_of(q).items()}
        topic = tuple(topic_names)
        items = list(p["items"])
        raw = p["ids"] if p["ids"] is not None else [None] * len(items)
        ids = [i if i is not None and is_id(i, kg) else None for i in raw]
        counts["item ids that are literals (matched as text)"] += sum(i is not None and j is None
                                                                       for i, j in zip(raw, ids))
        rec = AnswerRecord(qid=qid, question=str(q.get("question") or ""), items=tuple(items), item_ids=tuple(ids),
                           topic=topic, answer_text=str(row.get("answer") or ""),
                           error=None if p["status"] == "ok" else p["status"])
        if p["status"] != "ok":
            answers[qid] = rec
            gh.append(0.0)
            continue
        named = None
        if kg in NO_BODY_KGS:
            recs, kind = None, None
            counts["logged on neo4j: no result rows"] += 1
        else:
            recs, named = calls_of(system, row, shard, cell_dir, kg, counts, topic_names)
            kind = RETURNED
        if recs is None:
            counts["answered rows without evidence (left out)"] += 1
            continue
        kinds[kind] += 1
        answers[qid] = rec
        if named is not None:
            recs, alt[qid] = (named, recs) if name_anchor else (recs, named)
        if recs:
            calls[qid] = recs
        gold_ids = [canonical_id(str(x)) for x in q.get("answer_ids") or []]
        gh.append(grounded_hit(rec.question, items, golds_of(q), gold_ids, topic, recs))
    ev = support.evaluate_set(answers, calls, list(answers))
    qs = ev["questions"]
    other = []
    for x in qs:
        a = answers[x["qid"]]
        if x["qid"] in alt:
            v = support.item_verdicts(a.question, a.items, a.item_ids, a.topic, alt[x["qid"]])
            other.append(sum(k == support.STRICT for k, _ in v) / len(v))
        else:
            other.append(x["support"])
    n_items = sum(len(x["items"]) for x in qs)
    near = sum(near_misses(x["items"], answers[x["qid"]], calls.get(x["qid"], [])) for x in qs)
    guard = sum(i["type_rejected"] for x in qs for i in x["items"])
    non_yn = [[i for i in x["items"] if not i["yes_no"]] for x in qs]
    non_yn = [s for s in non_yn if s]
    h = ev["headline"]
    alt_rate = sum(other) / len(other) if other else None
    counts.update(answered=ev["counts"]["answered"], **{"answered without records": ev["counts"]["no_records"]})
    return {"support": h["support_rate"], "why": None, "answer_rate": ev["answer_rate"],
            "evidence": RETURNED if kinds else None, "evidence_rows": {k: v for k, v in kinds.items() if v},
            "name_anchor": name_anchor,
            "support_4a_on": h["support_rate"] if name_anchor else alt_rate,
            "support_4a_off": alt_rate if name_anchor else h["support_rate"],
            "items": n_items, "near_miss": near, "near_miss_share": near / n_items if n_items else None,
            "type_guard_items": guard, "type_guard_share": guard / n_items if n_items else None,
            "lenient_share": h["lenient_share"], "unsupported_share": h["unsupported_share"],
            "yes_no_share": h["yes_no_share"], "chance": ev["chance_rate"],
            "support_no_yes_no": sum(sum(i["strict"] for i in s) / len(s) for s in non_yn) / len(non_yn)
            if non_yn else None,
            "grounded_hit": sum(gh) / len(gh) if gh else None, "questions": qs,
            "complete": scores["complete"], "counts": {k: v for k, v in counts.items() if v}}
