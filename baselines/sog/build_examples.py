"""
sog/build_examples.py
----------------------
SoG's five exemplars for the graph the authors did not run (MetaQA). The paper builds "five diverse exemplars" per
dataset by hand (upstream/prompt.py); here each one is a TRAIN question walked along its gold relation path with the
real tool, written in the authors' exemplar style (topic entity line, Tool Call, "Suppose it returns:" with a few
rows of the actual table, "We see ...", "Final answer: {..}"). Test questions are never read. Run once against a live
graph; the output (examples_extra.json) is committed.

    NEO4J_METAQA_URI=... python -m baselines.sog.build_examples --train TRAIN.json
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import random

from baselines.sog.graphs import get_adjacent_fn

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "examples_extra.json")
SEED = 20261006
SHOWN = 6          # rows shown per table, as the authors' exemplars show a handful
MAX_BRANCH = 3     # frontier nodes followed per hop


def _lit(v):
    return ast.literal_eval(v) if isinstance(v, str) and v[:1] in "[{(" else v


def _short_relation(rel: str) -> str:
    return rel.split(".", 1)[1] if rel.count(".") == 1 and not rel.isupper() else rel


def _parse(table: str) -> list[dict]:
    lines = [l for l in table.strip().splitlines() if "|" in l]
    if len(lines) < 2:
        return []
    head = [h.strip() for h in lines[0].split("|")]
    return [dict(zip(head, (c.strip() for c in l.split("|", len(head) - 1)))) for l in lines[2:]]


def _same(a: str, b: str) -> bool:
    """A gold relation and a graph property, with underscores read as spaces and case ignored."""
    return a.replace("_", " ").lower() == b.replace("_", " ").lower()


def _render(rows: list[dict], cols: list[str]) -> str:
    return "\n".join(["|".join(cols), "|".join("--" for _ in cols)] + ["|".join(r.get(c, "") for c in cols) for r in rows])


def _call(entity, direction, filt=None) -> str:
    arguments = {"entity": entity, "direction": direction}
    if filt:
        arguments["properties_to_filter_for"] = filt
    return ('Tool Call: [{"id": "some_unique_id", "type": "function", "function": {"name": '
            '"get_adjacent_relations_and_entities","arguments": ' + json.dumps(arguments) + "}}]")


def _name(row: dict) -> str:
    return f"{row['value']} ({row['valueLabel']})" if row.get("valueLabel") else row["value"]


#: MetaQA: the words that name each relation. A gold path whose relation the question never names is a path that
#: happens to reach the answers (a writer who also directed), not the one the question asks for; it is skipped.
METAQA_WORDS = {"DIRECTED_BY": ("direct",), "WRITTEN_BY": ("writ", "wrote", "screenwriter"),
                "STARRED_ACTORS": ("act", "star", "appear"), "IN_LANGUAGE": ("language",),
                "HAS_GENRE": ("genre", "type"), "RELEASE_YEAR": ("year", "release", "when"),
                "HAS_TAGS": ("about", "tag", "describ", "topic"), "HAS_IMDB_RATING": ("rating", "rate"),
                "HAS_IMDB_VOTES": ("vote", "popular")}


def names_its_relations(q: dict) -> bool:
    text = q["question"].lower()
    return all(any(w in text for w in METAQA_WORDS.get(h["relation"], ("",))) for h in _lit(q["gold"])["hops"])


def walk(q: dict, tool) -> str | None:
    """The exemplar text for one train question, or None if its gold path does not reproduce its answers."""
    topic = _lit(q["topic_entity"])
    gold = _lit(q["gold"])
    answers = [str(a) for a in _lit(q["answers"])]
    (eid, label), = list(topic.items())[:1]
    hops = gold["hops"]
    text = [f"--\nQuestion: {q['question']} {repr({label: eid})}"]
    frontier = [(eid, label)]
    for i, hop in enumerate(hops):
        rel, direction = _short_relation(hop["relation"]), "outgoing" if hop["direction"] == "out" else "incoming"
        nxt = []
        for node, node_label in frontier[:MAX_BRANCH]:
            if i == 0:
                text.append(f"'{node}' ('{node_label}') is the topic entity of the question. Look for {direction} edges "
                            f"from '{node}' and look for relations and entities related to {rel.replace('_', ' ').lower()}. "
                            f"Any linked object entities are potential candidates.")
            else:
                text.append(f"Get {direction} relations and entities from {node}"
                            f"{f' ({node_label})' if node_label and node_label != node else ''} to find its "
                            f"{rel.replace('_', ' ').lower()}:")
            text.append(_call(node, direction))
            table = tool(node, direction)
            header = next((l for l in table.splitlines() if "|" in l), "")
            if "value" not in header:                          # the high-degree case: properties only
                text.append("Suppose it returns:\n" + table.strip())
                listed = [r.get("property", "") for r in _parse(table)]
                rel = next((p for p in listed if _same(p, rel)), rel)
                text.append(f"The result lists only properties. The property {rel} is relevant, so filter for it:")
                text.append(_call(node, direction, [rel]))
                table = tool(node, direction, [rel])
            rows = _parse(table)
            hit = [r for r in rows if _same(r.get("property", ""), rel)]
            rel = hit[0]["property"] if hit else rel
            if not hit:
                return None
            other = [r for r in rows if not _same(r.get("property", ""), rel)][: max(0, SHOWN - len(hit[:3]))]
            shown = (hit[:3] + other)[:SHOWN]
            random.Random(SEED).shuffle(shown)
            text.append("Suppose it returns:\n" + _render(shown, ["property", "propertyLabel", "value", "valueLabel"]))
            text.append(f"We see the property {rel} ({hit[0].get('propertyLabel', rel)}), which points to "
                        + ", ".join(_name(r) for r in hit[:3]) + ".")
            nxt += [(r["value"], r.get("valueLabel", "")) for r in hit]
        frontier = nxt
    reached = {(l or v).lower() for v, l in frontier}
    if gold.get("projection") == "count":
        final = "{" + str(len(answers) if not answers[0].isdigit() else answers[0]) + "}"
    else:
        if not any(a.lower() in reached for a in answers):
            return None
        final = ", ".join("{" + a + "}" for a in answers if a.lower() in reached)
    text.append(f"Final answer: {final}")
    return "\n".join(text) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True, help='the MetaQA train questions ({"questions": [...]}, with gold paths)')
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--per-hop", type=int, default=2, dest="per_hop", help="at most this many exemplars per hop count")
    p.add_argument("--exclude", default="", help="train ids whose gold path contradicts the question (read by hand)")
    a = p.parse_args(argv)
    train = json.load(open(a.train))["questions"]
    tool = get_adjacent_fn("metaqa")
    excluded = set(filter(None, a.exclude.split(",")))
    pool = [q for q in train if q["id"] not in excluded and q.get("gold") and q.get("topic_entity")
            and len(_lit(q["topic_entity"])) == 1
            and _lit(q["gold"]).get("projection") in ("identity", "count") and not _lit(q["gold"]).get("constraints")
            and 1 <= len(_lit(q["gold"])["hops"]) <= 3 and len(_lit(q["answers"])) <= 4
            and names_its_relations(q)]
    random.Random(SEED).shuffle(pool)
    chosen, shapes = [], set()
    for q in pool:
        g = _lit(q["gold"])
        shape = (len(g["hops"]), g.get("projection"), g["hops"][0]["relation"], q.get("question_type"))
        if shape in shapes or sum(1 for h in shapes if h[0] == shape[0]) >= a.per_hop:
            continue
        try:
            ex = walk(q, tool)
        except Exception as e:  # a graph error on one train question: take the next
            print(f"  skip {q['id']}: {e}")
            continue
        if ex:
            chosen.append({"id": q["id"], "text": ex})
            shapes.add(shape)
            print(f"  kept {q['id']} {shape}")
        if len(chosen) == a.n:
            break
    out = json.load(open(OUT)) if os.path.exists(OUT) else {}
    out["metaqa"] = chosen
    json.dump(out, open(OUT, "w"), indent=1, ensure_ascii=False)
    print(f"metaqa: {len(chosen)} exemplars -> {OUT}")
    return 0 if len(chosen) == a.n else 1


if __name__ == "__main__":
    raise SystemExit(main())
