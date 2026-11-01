"""One question's calls records, built from the graph meter's log, nothing guessed (README: calls records)."""
from __future__ import annotations

import json
import os
import re

from harness.metrics.groundedness.records import is_id, parse_call, short_id

_FB_ENTITY = re.compile(r"(?:\bns:|<http://rdf\.freebase\.com/ns/)([mg]\.[0-9a-z_]+)>?")
_NODE_IRI = re.compile(r"<(https?://[^<>\s]*/node/[^<>\s]+)>")
_FB_RELATION = re.compile(r"(?:\bns:|<http://rdf\.freebase\.com/ns/)([a-z][a-z0-9_]*(?:\.[a-z0-9_]+)+)>?")
_IRI_RELATION = re.compile(r"<https?://[^<>\s]*/relation/([^<>\s]+)>")
_LABEL_PREDICATE = (r"(?:\bns:type\.object\.name\b|<http://rdf\.freebase\.com/ns/type\.object\.name>|\brdfs:label\b|"
                    r"<http://www\.w3\.org/2000/01/rdf-schema#label>|\bschema:name\b|<https?://schema\.org/name>)")
_LABEL_TRIPLE = re.compile(r"(\?\w+|\bns:[mg]\.[0-9a-z_]+|<[^<>\s]+>)\s+" + _LABEL_PREDICATE + r"\s+\?(\w+)")
_ASK = re.compile(r"^\s*(?:(?:PREFIX|BASE)\b[^\n]*\n\s*)*ASK\b", re.I)
NOT_RELATIONS = {"type.object.name", "type.object.type"}
LABEL_VARS = ("name", "label")
LABEL_SUFFIXES = ("Label", "_label", "_name", "Name")


def query_text(query) -> str:
    """The query as text (a neo4j call's {"text", "parameters"} as JSON)."""
    return query if isinstance(query, str) else json.dumps(query, sort_keys=True)


def anchors(query: str, kg: str) -> list:
    """The entity ids written as constants in the query, in order of first appearance."""
    found = [(m.start(), m.group(1)) for m in _FB_ENTITY.finditer(query)]
    found += [(m.start(), short_id(m.group(1))) for m in _NODE_IRI.finditer(query)]
    return list(dict.fromkeys(i for _, i in sorted(found) if kg != "metaqa" and is_id(i, kg)))


def relation_of(query: str, kg: str):
    """The query's one relation constant, or None."""
    rels = {r for r in _FB_RELATION.findall(query) if not is_id(r, "freebase")} - NOT_RELATIONS
    rels |= set(_IRI_RELATION.findall(query))
    return rels.pop() if len(rels) == 1 else None


def label_subjects(query: str, kg: str) -> dict:
    """{literal variable: ("id", entity id) | ("var", variable)}: the subject each label read of the query names."""
    out = {}
    for subject, var in _LABEL_TRIPLE.findall(query):
        if subject.startswith("?"):
            out.setdefault(var, ("var", subject[1:]))
        else:
            ids = anchors(subject, kg)
            if ids:
                out.setdefault(var, ("id", ids[0]))
    return out


def _label_target(var: str, entities: dict, subjects: dict):
    """The entity this literal variable labels; None if none, False if it names one the call did not return."""
    if var in subjects:
        kind, x = subjects[var]
        return x if kind == "id" else entities.get(x, False)
    for suffix in LABEL_SUFFIXES:
        if var.endswith(suffix) and var[: -len(suffix)] in entities:
            return entities[var[: -len(suffix)]]
    if var in LABEL_VARS and len(set(entities.values())) == 1:
        return next(iter(entities.values()))
    return None


def _record(line: dict, kg: str, names: dict) -> dict:
    """One raw log line as a calls line; labels found go to `names` (id -> first label)."""
    query = query_text(line.get("query") or "")
    anchor = anchors(query, kg)
    relation = relation_of(query, kg)
    subjects = label_subjects(query, kg)
    out = {"qid": line.get("qid"), "seq": line.get("seq"), "source": "hook", "client": line.get("client"),
           "endpoint": line.get("endpoint"), "anchor": anchor, "query": query, "relation": relation, "result": [],
           "relations": [], "n_rows": None}
    try:
        body = json.loads(line.get("body") or "null") if line.get("error") is None else None
    except (TypeError, ValueError):
        body = None
    bindings = ((body or {}).get("results") or {}).get("bindings") if isinstance(body, dict) else None
    if not isinstance(bindings, list):
        return out
    if _ASK.match(query) or any(isinstance(r, dict) and "__ASK_RETVAL" in r for r in bindings):
        bindings = []
    out["n_rows"] = len(bindings)
    result, keys, relations = [], set(), []
    for row in bindings:
        if not isinstance(row, dict):
            continue
        entities, literals = {}, []
        for var, v in row.items():
            if not isinstance(v, dict) or v.get("value") is None:
                continue
            if v.get("type") == "uri":
                sid = short_id(str(v["value"]))
                if kg != "metaqa" and is_id(sid, kg):
                    entities[var] = sid
                else:
                    relations.append(sid.rsplit("/", 1)[-1])
            else:
                literals.append((var, str(v["value"])))
        one = set(entities.values())
        owner = next(iter(one)) if len(one) == 1 else (anchor[0] if len(anchor) == 1 else None)
        items = [{"id": e} for e in entities.values()]
        for var, value in literals:
            target = _label_target(var, entities, subjects)
            if target:
                names.setdefault(target, value)
            else:
                items.append({"value": value, "prop": relation, "of": owner})
        for it in items:
            k = it.get("id") or ("v", it["value"], it["prop"], it["of"])
            if k not in keys:
                keys.add(k)
                result.append(it)
    out["result"], out["relations"] = result, list(dict.fromkeys(relations))
    return out


def records_of(lines: list, kg: str) -> list:
    """The question's CallRecords, by seq, from its log lines; a refused line raises RecordError."""
    names: dict = {}
    raw = [_record(l, kg, names) for l in sorted(lines, key=lambda l: l.get("seq") or 0)]
    for r in raw:
        for it in r["result"]:
            if "id" in it and it["id"] in names:
                it["label"] = names[it["id"]]
    return [parse_call(r, f"graph log seq={r['seq']}", r["qid"]) for r in raw]


def read_lines(path: str) -> tuple:
    """(the log lines that parse to a call, how many do not); a cut or foreign line is skipped, counted."""
    if not os.path.exists(path):
        return [], 0
    good, bad = [], 0
    for line in open(path, encoding="utf-8", errors="replace"):
        if not line.strip():
            continue
        try:
            d = json.loads(line)
        except ValueError:
            bad += 1
            continue
        seq = d.get("seq") if isinstance(d, dict) else None
        if not isinstance(seq, int) or isinstance(seq, bool) or not isinstance(d.get("qid"), str):
            bad += 1
            continue
        good.append(d)
    return good, bad


def repeated_seqs(lines: list) -> int:
    """How many lines repeat a seq: two processes wrote one question's log (counted, never fatal)."""
    return len(lines) - len({l.get("seq") for l in lines})


def load(path: str, kg: str) -> list:
    """records_of the log file at `path` (read_lines: cut lines skipped; [] when the file does not exist)."""
    return records_of(read_lines(path)[0], kg)
