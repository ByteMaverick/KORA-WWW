"""One training row: the path rendered and labelled by the gold, and the checks of a sampled set."""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from dataclasses import dataclass, field

#: The label rule, the gold stop's: the candidate covers at least half the gold and at least half of it is gold.
RECALL_MIN = 0.5
PRECISION_MIN = 0.5
#: Names written into the candidate line; the count stays the true count.
NAMES_SHOWN = 30
QUESTIONS = {"sufficient": {"type": "noul", "instructions": "Does the evidence answer the question, with the candidate "
             "as the answer, and is the candidate the kind and number of things the question asks for?"}}


@dataclass(frozen=True)
class Node:
    id: str
    name: str | None    # None: an unnamed node (a Freebase mediator)

    @property
    def named(self) -> bool:
        return self.name is not None


@dataclass
class Hop:
    """One step: from `subjects` through `relation` (out: subject -> object; in: object -> subject) to `objects`."""
    subjects: list[Node]
    relation: str
    direction: str          # "out" | "in"
    objects: list[Node]


@dataclass(frozen=True)
class Constraint:
    """A restriction the path applied to the set at `at`, written as its own line: set | `relation` | `value`."""
    at: int
    relation: str
    value: str


@dataclass
class Path:
    anchors: list[Node]
    hops: list[Hop]
    source: str             # spine | enum | truncate | extend | sibling | drop_constraint | wrong_value | same_name | random
    score: float = 0.0      # the ranker's score of the relation chain, for selection
    extra: dict = field(default_factory=dict)
    constraints: list[Constraint] = field(default_factory=list)

    @property
    def candidate(self) -> list[Node]:
        return self.hops[-1].objects if self.hops else []

    @property
    def n_hops(self) -> int:
        """Hops between named nodes: a mediator hop does not count."""
        return sum(1 for h in self.hops if any(o.named for o in h.objects)) or len(self.hops)

    def key(self) -> tuple:
        return (tuple((h.relation, h.direction) for h in self.hops), tuple(sorted(n.id for n in self.candidate)))


def _display(nodes: list[Node], unnamed_index: int) -> str:
    """Names, deduplicated, at most NAMES_SHOWN of them with the rest counted; an empty set is (none); unnamed is ?x<k>."""
    if not nodes:
        return "(none)"
    named = list(dict.fromkeys(n.name for n in nodes if n.named))
    if named:
        if len(named) > NAMES_SHOWN:
            return ", ".join(named[:NAMES_SHOWN]) + f", ... (+{len(named) - NAMES_SHOWN} more)"
        return ", ".join(named)
    return f"?x{unnamed_index}"


#: Names a constraint line shows of the set it restricts; the set itself is listed on its hop line.
CONSTRAINT_SUBJECT_NAMES = 3


def _short(nodes: list[Node], full: str) -> str:
    """A set as the subject of a constraint line: ?x<k> as on its hop line, else its first names and a count."""
    if full.startswith("?x") or not nodes:
        return full
    named = list(dict.fromkeys(n.name for n in nodes if n.named))
    if len(named) > CONSTRAINT_SUBJECT_NAMES:
        return ", ".join(named[:CONSTRAINT_SUBJECT_NAMES]) + f" (+{len(named) - CONSTRAINT_SUBJECT_NAMES} more)"
    return ", ".join(named) or full


def evidence_of(path: Path) -> list[str]:
    """'subject | relation | object' per hop, an unnamed set as ?x<k>; each constraint after the hop it restricts."""
    lines, k, prev = [], 0, _display(path.anchors, 0)
    if not path.anchors or not any(a.named for a in path.anchors):
        k += 1; prev = f"?x{k}"
    sets, shown = [path.anchors], [prev]
    for hop in path.hops:
        if any(o.named for o in hop.objects) or not hop.objects:
            obj = _display(hop.objects, 0)
        else:
            k += 1; obj = f"?x{k}"
        lines.append(f"{prev} | {hop.relation} | {obj}" if hop.direction == "out" else f"{obj} | {hop.relation} | {prev}")
        sets.append(hop.objects); shown.append(obj)
        prev = obj
    if not path.constraints:
        return lines
    out = []
    for at in range(len(sets)):
        if at:
            out.append(lines[at - 1])
        for c in path.constraints:
            if min(max(c.at, 0), len(sets) - 1) == at:
                out.append(f"{_short(sets[at], shown[at])} | {c.relation} | {c.value}")
    return out


def relations_of(path: Path) -> list[str]:
    """The relations the path walked, in order (the lucky-label rule compares them with the gold path's)."""
    return [h.relation for h in path.hops]


def directions_of(path: Path) -> list[str]:
    return [">" if h.direction == "out" else "<" for h in path.hops]


def candidate_names(path: Path) -> list[str]:
    return list(dict.fromkeys(n.name for n in path.candidate if n.named))


def label_of(candidate: list[Node], gold_names: list[str], gold_ids: list[str] | None = None) -> tuple[int, float, float]:
    """(label, recall, precision): an item matches the gold by id when ids are known, else by case-folded name."""
    gids = set(gold_ids or [])
    gnames = {g.strip().casefold() for g in gold_names if g and g.strip()}
    if not candidate or not (gids or gnames):
        return 0, 0.0, 0.0
    hit_ids: set[str] = set(); hit_names: set[str] = set(); matched = 0
    for n in candidate:
        ok = False
        if n.id in gids:
            hit_ids.add(n.id); ok = True
        if n.named and n.name.strip().casefold() in gnames:
            hit_names.add(n.name.strip().casefold()); ok = True
        matched += ok
    gold_total = max(len(gids), len(gnames))
    found = max(len(hit_ids), len(hit_names))
    recall = found / gold_total if gold_total else 0.0
    precision = matched / len(candidate)
    return int(recall >= RECALL_MIN and precision >= PRECISION_MIN), recall, precision


@dataclass
class Row:
    qid: str
    bench: str
    question: str
    evidence: list[str]
    candidate: list[str]
    n_candidate: int
    label: int
    source: str
    hops: int
    recall: float
    precision: float
    score: float = 0.0
    relations: list[str] = field(default_factory=list)
    directions: list[str] = field(default_factory=list)
    constraints: int = 0        # constraint lines shown
    constraint_lines: list[str] = field(default_factory=list)   # "relation = value" per line, for the look-alike check
    lucky: bool = False         # a yes whose path shares no relation with the gold path: not trained on
    # What kind of thing the path starts from and the candidate is, from the graph (any system can read it):
    # Freebase's notable type. Empty when the graph has none.
    anchor_ids: list[str] = field(default_factory=list)
    anchor_names: list[str] = field(default_factory=list)
    candidate_ids: list[str] = field(default_factory=list)
    anchor_kind: str = ""
    candidate_kind: str = ""

    def text(self) -> str:
        lines = list(self.evidence)
        if self.anchor_kind and self.anchor_names:
            lines = [f"{', '.join(self.anchor_names[:3])} | kind | {self.anchor_kind}"] + lines
        ev = "\n".join(f"  {line}" for line in lines) or "  (nothing retrieved)"
        cand = "; ".join(self.candidate[:NAMES_SHOWN]) or "(empty)"
        kind = f"; kind: {self.candidate_kind}" if self.candidate_kind else ""
        return (f"QUESTION\n{self.question}\n\nEVIDENCE (facts retrieved so far, as subject | relation | object)\n{ev}"
                f"\n\nCANDIDATE ANSWER ({self.n_candidate} items{kind})\n{cand}")

    def record(self) -> dict:
        return {"state": self.text(), "questions": QUESTIONS, "expected": {"sufficient": bool(self.label)}}

    def meta(self) -> dict:
        return {k: getattr(self, k) for k in ("qid", "bench", "question", "evidence", "candidate", "n_candidate", "label",
                                              "source", "hops", "recall", "precision", "score", "relations",
                                              "directions", "constraints", "constraint_lines", "lucky",
                                              "anchor_kind", "candidate_kind")}


def row_of(path: Path, qid: str, bench: str, question: str, gold_names: list[str], gold_ids: list[str] | None) -> Row:
    label, recall, precision = label_of(path.candidate, gold_names, gold_ids)
    return Row(qid=qid, bench=bench, question=question, evidence=evidence_of(path), candidate=candidate_names(path),
               n_candidate=len({n.id for n in path.candidate}), label=label, source=path.source, hops=path.n_hops,
               recall=round(recall, 4), precision=round(precision, 4), score=round(path.score, 4),
               relations=relations_of(path), directions=directions_of(path), constraints=len(path.constraints),
               constraint_lines=[f"{c.relation} = {c.value}" for c in path.constraints],
               anchor_ids=[a.id for a in path.anchors], anchor_names=[a.name for a in path.anchors if a.named],
               candidate_ids=list(dict.fromkeys(n.id for n in path.candidate))[:NAMES_SHOWN])


def majority_kind(ids: list[str], kinds: dict[str, str]) -> str:
    """The most common kind among the items that have one; ties keep the first seen."""
    counts: dict[str, int] = {}
    for i in ids:
        k = kinds.get(i)
        if k:
            counts[k] = counts.get(k, 0) + 1
    return max(counts, key=counts.get) if counts else ""


def add_kinds(rows: list[Row], kinds_of) -> None:
    """Fill each row's anchor and candidate kind from `kinds_of(ids) -> {id: kind}`, one call per question."""
    ids = list(dict.fromkeys(i for r in rows for i in r.anchor_ids + r.candidate_ids))
    if not ids:
        return
    kinds = kinds_of(ids)
    for r in rows:
        r.anchor_kind = majority_kind(r.anchor_ids, kinds)
        r.candidate_kind = majority_kind(r.candidate_ids, kinds)


def write_rows(rows: list[Row], rows_path: str, meta_path: str) -> None:
    with open(rows_path, "a", encoding="utf-8") as fl, open(meta_path, "a", encoding="utf-8") as fm:
        for r in rows:
            fl.write(json.dumps(r.record(), ensure_ascii=False) + "\n")
            fm.write(json.dumps(r.meta(), ensure_ascii=False) + "\n")


def ids_and_texts(path: str) -> tuple[set[str], set[str]]:
    """Ids and texts of a split-shaped file or a list of questions; a bare list of ids gives ids only."""
    d = json.load(open(path))
    qs = d["questions"] if isinstance(d, dict) else d
    if qs and not isinstance(qs[0], dict):
        return ({str(i) for i in qs}, set())
    return ({str(q["id"]) for q in qs}, {q["question"].strip().casefold() for q in qs})


def anchors_text(r: dict) -> str:
    """The starting entities as the first hop line writes them (`Barack Obama, Michelle Obama`)."""
    rels, dirs = r.get("relations") or [], r.get("directions") or []
    if not rels:
        return ""
    for line in r.get("evidence") or []:
        parts = line.split(" | ")
        if len(parts) == 3 and parts[1] == rels[0]:
            return parts[0] if (dirs[:1] or [">"])[0] == ">" else parts[2]
    return ""


def structure(r: dict) -> tuple:
    """What a row says apart from the sets it reaches: its start, relations with directions, constraint lines."""
    return (anchors_text(r), r.get("anchor_kind", ""), r.get("candidate_kind", ""), tuple(r.get("relations") or []),
            tuple(r.get("directions") or []), tuple(sorted(r.get("constraint_lines") or [])))


def look_alikes(rows: list[dict]) -> tuple[int, int]:
    """(pairs, questions): yes / no rows of one question that differ only by their entity lists."""
    by: dict[tuple, dict[tuple, set[int]]] = collections.defaultdict(lambda: collections.defaultdict(set))
    for r in rows:
        if r.get("lucky"):
            continue
        by[(r["bench"], r["qid"])][structure(r)].add(r["label"])
    pairs = qs = 0
    for groups in by.values():
        n = sum(1 for labels in groups.values() if labels == {0, 1})
        pairs += n; qs += bool(n)
    return pairs, qs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m estimator.sampler.rows")
    ap.add_argument("meta"); ap.add_argument("--holdout", nargs="*", default=[]); ap.add_argument("--rows")
    ap.add_argument("--types", help="a questions file with question_type per id: prints the spine-positive rate per type")
    a = ap.parse_args(argv)
    rows = [json.loads(l) for l in open(a.meta)]
    held_ids: set[str] = set(); held_texts: set[str] = set()
    for h in a.holdout:
        i, t = ids_and_texts(h); held_ids |= i; held_texts |= t
    leaks = [r for r in rows if r["qid"] in held_ids or r["question"].strip().casefold() in held_texts]
    print(f"rows {len(rows)}, questions {len({r['qid'] for r in rows})}, held-out files {len(a.holdout)}, LEAKS {len(leaks)}")
    spine = [r for r in rows if r["source"] == "spine"]
    bad = [r for r in spine if r["label"] != 1]
    print(f"spine rows {len(spine)}, not positive {len(bad)} ({100 * len(bad) / max(len(spine), 1):.1f}%)")
    for r in bad[:5]:
        print("   spine not positive:", r["qid"], "|", r["question"][:80], "| recall", r["recall"], "precision", r["precision"])
    if a.types:
        d = json.load(open(a.types)); qs = d["questions"] if isinstance(d, dict) else d
        kind = {str(q["id"]): q.get("question_type") or q.get("type") or "?" for q in qs}
        per = collections.defaultdict(lambda: [0, 0])
        for r in spine:
            t = kind.get(r["qid"], "?"); per[t][0] += 1; per[t][1] += r["label"]
        print("spine positive by type: " + ", ".join(f"{t} {p}/{n}" for t, (n, p) in sorted(per.items())))
    by = collections.defaultdict(lambda: [0, 0, 0.0])
    for r in rows:
        b = by[r["source"]]; b[0] += 1; b[1] += r["label"]; b[2] += r["recall"]
    print("source            rows    pos   mean recall")
    for s, (n, p, rc) in sorted(by.items(), key=lambda kv: -kv[1][0]):
        print(f"{s:16s} {n:6d} {p:6d}   {rc / n:.2f}")
    pos = sum(r["label"] for r in rows)
    print(f"positive share {100 * pos / max(len(rows), 1):.1f}%")
    pairs, questions = look_alikes(rows)
    print(f"LOOK-ALIKE yes/no pairs (same relations and constraint lines, opposite labels): {pairs} in {questions} questions")
    lucky = sum(1 for r in rows if r.get("lucky"))
    print(f"lucky yes rows (no gold relation on the path; kept for the floor, not trained on): {lucky} "
          f"({100 * lucky / max(pos, 1):.1f}% of yes rows)")
    single = lambda lab: [r for r in rows if r["label"] == lab and not r.get("lucky")]
    for lab in (1, 0):
        rs = single(lab)
        print(f"{'yes' if lab else 'no '} rows with one item: {100 * sum(r['n_candidate'] == 1 for r in rs) / max(len(rs), 1):.0f}%")
    if a.rows and os.path.exists(a.rows):
        lens = [len(json.loads(l)["state"].split()) for l in open(a.rows)]
        print(f"rendered length: over 512 words {100 * sum(x > 512 for x in lens) / len(lens):.1f}%, "
              f"over 2048 {100 * sum(x > 2048 for x in lens) / len(lens):.2f}%")
    return 1 if leaks else 0



if __name__ == "__main__":
    sys.exit(main())
