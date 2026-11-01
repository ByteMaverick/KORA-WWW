"""The checks: the eight-check suite and the build's run of it, the held-out filter, the pollution check, the trim."""
from __future__ import annotations

import collections
import itertools
import json
import random
import re
import shutil
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Mapping

import requests

from . import prompts
from .graph import (_NOT_BOILERPLATE, LABEL_BATCH, NS, Freebase, LiteralRange, _cached, _chunks, _values, boilerplate,
                    bulk_load, collides_remote, refuse_source)
from .names import BACKOFF, kind_of, normalise
from .twin import KINDS, _inside, _parses, _parts, _template, expose, is_id, parse

if TYPE_CHECKING:
    from .main import Build


# step 7: the eight checks, run on the extended copy
SAMPLE = 50                    # clones per model-based check
ATTRIBUTES = 4                 # attributes asked per clone
NEIGHBOURS = 30                # facts shown per clone
HUMAN_READ = 100               # twins in human_read.jsonl
PADDABLE = 1000                # a type's q1 from which a clone below it only warns
SURFACE_GAP = 0.25             # a surface share this far from the real one is listed
SURFACE_RATES = ("digit", "year", "title", "lower", "mixed", "none")
HARDNESS = ("out_degree", "relations", "two_hop", "in_degree")
HARDNESS_SAMPLE = 200          # compared clones per run whose two-hop reach and in-degree are measured (seeded)
ROLES = ("topic", "intermediate", "answer")
SMALL_WORDS = frozenset("a an and as at by de del der di du for in la le of on or the to van von".split())
SEEDED = ("nationality", "gender", "ethnicity", "language", "religion", "country")   # names are seeded with these
DATE_PAIRS = (("people.person.date_of_birth", "people.deceased_person.date_of_death"),
              ("time.event.start_date", "time.event.end_date"))
CHECKS = ("real_unchanged", "planted_executes", "names_unknown", "not_renamed_source", "readable",
          "structure_matches", "prior_empty", "difficulty_matched")
HISTOGRAMS = {"shape": lambda r: r["gold"]["shape"], "hops": lambda r: r["gold"]["shape"].split("|")[0],
              "constraints": lambda r: len(r["gold"]["constraints"]), "answer_size": lambda r: len(r["answer_ids"])}
_NOT_BOILERPLATE_Q = _NOT_BOILERPLATE.replace("STR(?p)", "STR(?q)")
_YEAR = re.compile(r"(?<!\d)(?:1\d|20)\d\d(?!\d)")
_CLONE_SUBJECT = re.compile(rf"<{re.escape(NS)}m\.x\w+>")         # a clone's IRI (``fresh_ids``)
_REFUSED = (RuntimeError, requests.RequestException)    # a partial reply, the row cap, HTTP: a failure, not a crash
SubCheck = tuple[bool | None, dict, list[dict]]   # a pass of None: advisory


def _entry(passed: bool | None, summary: dict | None = None, failures: list | None = None, **extra) -> dict:
    return {"pass": passed, "summary": summary or {}, "failures": failures or [], **extra}


def _combined(parts: dict[str, SubCheck | None]) -> dict:
    """One check from its sub-checks (None: skipped)."""
    ran = {name: part for name, part in parts.items() if part is not None}
    summary = {name: part[1] if part else "skipped" for name, part in parts.items()}
    if not ran:
        return _entry(None, summary, skipped=True)
    failures = [{"check": name, **f} for name, part in ran.items() for f in part[2]]
    if not (decided := [part[0] for part in ran.values() if part[0] is not None]):
        return _entry(None, summary, failures, advisory=True)
    return _entry(all(decided), summary, failures)


def _sample(ids: list[str], rng: random.Random) -> list[str]:
    return sorted(rng.sample(sorted(ids), min(SAMPLE, len(ids))))


def _says(reply: str, text: str) -> bool:
    name = normalise(text)
    return bool(name) and f" {name} " in f" {normalise(reply)} "     # normalised: words between single spaces


def _seeded(attribute: str) -> bool:
    """An attribute a name was seeded with: a word of its key, made singular, is in ``SEEDED``."""
    return any(re.sub("ies$", "y", word).removesuffix("s") in SEEDED for word in attribute.split())


def _execute(fb: Freebase, record: dict) -> tuple[set[str], set[str]]:
    """The record's answers and its gold's on ``fb``: as ids, or as names when the record holds no ids."""
    query = record["gold"]["query"]
    var = parse(query).select
    got = {row[var] for row in fb.select(query) if var in row}
    if record["answer_ids"]:
        return {str(a) for a in record["answer_ids"]}, got
    labels = fb.labels(sorted(v for v in got if is_id(v)))
    return {str(a) for a in record["answers"]}, {labels.get(v, v) for v in got}


def _mismatches(fb: Freebase, records: list[dict], sets: bool = False) -> list[dict]:
    """A failure per record whose gold, run on ``fb``, does not return its answers."""
    out = []
    for record in records:
        try:
            expected, executed = _execute(fb, record)
        except _REFUSED as e:
            out.append({"id": record["id"], "error": str(e)})
            continue
        if expected != executed:
            out.append({"id": record["id"], "expected": len(expected), "executed": len(executed),
                        **({"extra": sorted(executed - expected), "missing": sorted(expected - executed)}
                           if sets else {})})
    return out


def _edges(fb: Freebase, id: str) -> list[tuple[str, str, str]]:
    """The node's edges as ``(relation, other end, "out" | "in")``, boilerplate left out, sorted."""
    rows = {"out": fb.select(f"SELECT ?p ?o WHERE {{ ns:{id} ?p ?o }}"),
            "in": fb.select(f"SELECT ?p ?o WHERE {{ ?o ?p ns:{id} }}")}
    return sorted((r["?p"], r["?o"], d) for d, found in rows.items() for r in found if not boilerplate(r["?p"]))


def _out_per_relation(fb: Freebase, id: str) -> collections.Counter:
    """The node's out-edges per relation, boilerplate filtered on the server: one grouped COUNT."""
    rows = fb.select(f"SELECT ?p (COUNT(*) AS ?n) WHERE {{ ns:{id} ?p ?o FILTER({_NOT_BOILERPLATE}) }} GROUP BY ?p")
    return collections.Counter({row["?p"]: int(row["?n"]) for row in rows})


def _in_degree(fb: Freebase, id: str) -> int:
    found = fb.select(f"SELECT (COUNT(*) AS ?n) WHERE {{ ?o ?p ns:{id} FILTER({_NOT_BOILERPLATE}) }}")
    return int(found[0]["?n"]) if found else 0


def _two_hop(fb: Freebase, id: str) -> int:
    """The distinct nodes within two outgoing hops, no boilerplate relation walked."""
    found = fb.select(f"SELECT (COUNT(DISTINCT ?n) AS ?c) WHERE {{ {{ ns:{id} ?p ?n FILTER({_NOT_BOILERPLATE}) }} "
                      f"UNION {{ ns:{id} ?p ?m . ?m ?q ?n FILTER({_NOT_BOILERPLATE} && {_NOT_BOILERPLATE_Q}) }} "
                      f"FILTER(isIRI(?n) && ?n != ns:{id}) }}")
    return int(found[0]["?c"]) if found else 0


def real_unchanged(ext: Freebase, twins: list[dict], sources: list[dict]) -> dict:
    failures = _mismatches(ext, sources)
    polluted = {f["id"] for f in failures}
    victims = [t["id"] for t in twins if t["twin_id"] in polluted]
    return _entry(not failures, {"sources": len(sources), "polluted": len(polluted), "victim_twin_ids": victims},
                  failures)


def planted_executes(ext: Freebase, twins: list[dict]) -> dict:
    failures, reached = _mismatches(ext, twins, sets=True), {}
    for t in twins:
        for c in t["clone_ids"]:
            if c not in reached:
                reached[c] = bool(ext.select(f"SELECT ?p ?o WHERE {{ ns:{c} ?p ?o }} LIMIT 1"))
            if not reached[c]:
                failures.append({"id": t["id"], "unreachable": c})
    return _entry(not failures, {"twins": len(twins), "clones": len(reached),
                                 "unreachable": sum(not r for r in reached.values())}, failures)


def _collisions(twins: list[dict], collides: Callable[[str], bool]) -> SubCheck:
    hit: dict[str, bool] = {}
    failures = []
    for t in twins:
        clones = set(t["clone_ids"])
        new = [name for id, name in t["topic_entity"].items() if id in clones]
        new += [name for id, name in zip(t["answer_ids"], t["answers"]) if id in clones]
        for name in dict.fromkeys(new):
            if name not in hit:
                hit[name] = collides(name)
            if hit[name]:
                failures.append({"id": t["id"], "name": name})
    return not failures, {"names": len(hit), "collisions": sum(hit.values())}, failures


def _attribute_test(ext: Freebase, clones: list[str], judge: Callable[[str], str], attrs: dict[str, dict],
                    clone_types: dict[str, str], rng: random.Random) -> SubCheck | None:
    asked = {c: [(a, v) for a, v in attrs.get(c, {}).items() if v and not _seeded(a)][:ATTRIBUTES] for c in clones}
    named = ext.labels([c for c in clones if asked[c]])
    sample = _sample(list(named), rng)
    if not sample:
        return None
    template = prompts.ATTRIBUTES
    known = []
    for c in sample:
        kind = kind_of(clone_types[c]) if c in clone_types else "entity"
        reply = judge(template.format(name=named[c], kind=kind, attributes=", ".join(a for a, _ in asked[c]))) or ""
        matched = [value for _, value in asked[c] if any(_says(reply, part) for part in value.split(", "))]
        if normalise(reply) != "unknown" and matched:
            known.append({"clone": c, "name": named[c], "matched": matched})
    return None, {"advisory": True, "sample": len(sample), "known": len(known), "known_share": len(known) / len(sample),
                  "known_clones": known}, []


def _source_labels(real: Freebase, mapping: dict[str, str]) -> dict[str, list[str]]:
    sources = sorted(mapping)
    labels, aliases = real.labels(sources), real.aliases(sources)
    return {mapping[s]: [labels[s], *sorted(aliases.get(s, ()))] for s in sources if s in labels}


def _facts(ext: Freebase, edges: list[tuple[str, str, str]]) -> list[str]:
    """``relation: label`` lines (``relation (inverse): label`` for an in-edge); literals as they are."""
    labels = ext.labels(sorted({o for _, o, _ in edges if is_id(o)}))
    return [f"{p}{' (inverse)' if d == 'in' else ''}: {labels.get(o, o)}"
            for p, o, d in edges if not is_id(o) or o in labels][:NEIGHBOURS]


def not_renamed_source(ext: Freebase, clones: list[str], judge: Callable[[str], str], labels: dict[str, list[str]],
                       rng: random.Random) -> dict:
    sample = _sample([c for c in clones if labels.get(c)], rng)
    if not sample:
        return _entry(None, {"reason": "no clone has a source label"}, skipped=True)
    template = prompts.GUESS_SOURCE
    failures, no_facts = [], 0
    for c in sample:
        facts = _facts(ext, _edges(ext, c))
        if not facts:
            no_facts += 1
            continue
        reply = (judge(template.format(facts="\n".join(facts))) or "").strip()
        if any(_says(reply, name) for name in labels[c]):
            failures.append({"clone": c, "source": labels[c][0], "guess": reply})
    return _entry(not failures, {"sample": len(sample), "asked": len(sample) - no_facts, "no_facts": no_facts,
                                 "identified": len(failures)}, failures)


def readable(twins: list[dict], sources: list[dict], out: Path) -> dict:
    question = {s["id"]: s["question"] for s in sources}
    by_shape: dict[str, list[dict]] = collections.defaultdict(list)
    for t in twins:
        if not t.get("flags"):
            by_shape[t["gold"]["shape"]].append(t)
    rounds = itertools.zip_longest(*(by_shape[shape] for shape in sorted(by_shape)))
    flagged = [t for t in twins if t.get("flags")]
    chosen = (flagged + [t for group in rounds for t in group if t is not None])[:HUMAN_READ]
    with (out / "human_read.jsonl").open("w", encoding="utf-8") as f:
        for t in chosen:
            f.write(json.dumps({**t, "source_question": question.get(t["twin_id"])}, ensure_ascii=False) + "\n")
    return _entry(None, {"written": len(chosen), "flagged": min(len(flagged), len(chosen)),
                         "file": "human_read.jsonl"}, pending="human read")


def _histograms(twins: list[dict], sources: list[dict]) -> SubCheck:
    by_id = {s["id"]: s for s in sources}
    paired = [by_id[t["twin_id"]] for t in twins if t["twin_id"] in by_id]
    summary, failures = {}, []
    for name, key in HISTOGRAMS.items():
        sides = {side: dict(sorted(collections.Counter(str(key(r)) for r in records).items()))
                 for side, records in (("sources", paired), ("twins", twins))}
        summary[name] = sides
        if sides["sources"] != sides["twins"]:
            failures.append({"histogram": name, **sides})
    return not failures, summary, failures


def _roles(twins: list[dict]) -> dict[str, str]:
    """Each clone's role in its twin: ``topic``, else ``answer``, else ``intermediate``."""
    roles: dict[str, str] = {}
    for t in twins:
        for c in t["clone_ids"]:
            role = "topic" if c in t["topic_entity"] else "answer" if c in t["answer_ids"] else "intermediate"
            roles[c] = min(roles.get(c, role), role, key=ROLES.index)
    return roles


def _hardness(measured: dict[str, dict[str, list[int]]], roles: dict[str, str]) -> dict:
    """Per role: the clones measured and each measure's mean on the sources and on the clones."""
    out = {}
    for role in ROLES:
        group = [m for c, m in sorted(measured.items()) if roles.get(c, "intermediate") == role]
        if group:
            out[role] = {"clones": len(group), **{
                name: {side: round(statistics.fmean(m[name][i] for m in values), 2)
                       for i, side in enumerate(("source", "clone"))}
                for name in HARDNESS if (values := [m for m in group if name in m])}}
    return out


def _degrees(clones: list[str], real: Freebase, ext: Freebase, clone_sources: dict[str, str], roles: dict[str, str],
             unpadded: dict[str, dict[str, int]] | None = None, rng: random.Random | None = None) -> SubCheck:
    """Each clone's out-degree per relation against its source's less what padding left unfilled."""
    failures, errors, measured = [], [], {}
    compared = [c for c in clones if c in clone_sources]
    sampled = set((rng or random.Random(0)).sample(sorted(compared), min(HARDNESS_SAMPLE, len(compared))))
    for c in compared:
        s = clone_sources[c]
        try:
            source, got = _out_per_relation(real, s), _out_per_relation(ext, c)
        except _REFUSED as e:
            failures.append({"clone": c, "source": s, "error": str(e)})
            continue
        want = source - collections.Counter((unpadded or {}).get(c, {}))
        if want != got:
            failures.append({"clone": c, "source": s, "relations": {
                p: [want[p], got[p]] for p in sorted(set(want) | set(got)) if want[p] != got[p]}})
        measured[c] = {"out_degree": [sum(source.values()), sum(got.values())], "relations": [len(source), len(got)]}
        for name, read in (("two_hop", _two_hop), ("in_degree", _in_degree)) if c in sampled else ():
            try:
                measured[c][name] = [read(real, s), read(ext, c)]
            except _REFUSED as e:
                errors.append({"clone": c, "measure": name, "error": str(e)})
    summary = {"exact": {"compared": len(compared), "unequal": len(failures), "no_source": len(clones) - len(compared),
                         "unpadded": sum(n for c in compared for n in (unpadded or {}).get(c, {}).values()),
                         "errors": errors},
               "hardness_sample": len(sampled), "hardness": _hardness(measured, roles)}
    return (not failures) if compared else None, summary, failures


def case(text: str) -> str:
    """``none``, ``lower``, ``title`` or ``mixed``."""
    words = [w for w in text.split() if any(ch.isalpha() for ch in w)]
    if not words:
        return "none"
    if not any(ch.isupper() for ch in text):
        return "lower"
    initials = [next(ch for ch in w if ch.isalpha()) for w in words]
    small = [normalise(w) in SMALL_WORDS for w in words]
    return "title" if initials[0].isupper() and all(i.isupper() or sm for i, sm in zip(initials[1:], small[1:])) \
        else "mixed"


def question_type(question: str) -> str:
    """``when``, ``how many``, ``who``, ``where``, else ``what``: the question's first words."""
    words = normalise(question).split()
    if words[:2] in (["how", "many"], ["how", "much"]):
        return "how many"
    first = words[0] if words else ""
    return {"when": "when", "who": "who", "whom": "who", "whose": "who", "where": "where"}.get(first, "what")


def _surface(texts: list[str]) -> dict:
    """The strings' surface: how many, the share with a digit, with a year, in each ``case``, the mean token count."""
    n = len(texts)
    cases = collections.Counter(map(case, texts))
    shares = {"digit": sum(any(ch.isdigit() for ch in t) for t in texts),
              "year": sum(bool(_YEAR.search(t)) for t in texts),
              **{c: cases[c] for c in ("title", "lower", "mixed", "none")}}
    return {"n": n, **{k: round(v / n, 3) for k, v in shares.items()},
            "tokens": round(statistics.fmean(len(t.split()) for t in texts), 2)}


def _answer_strings(record: dict) -> list[str]:
    """The record's answers as text: ``answers``, else its literal ``answer_ids``."""
    return [str(a) for a in record["answers"]] or [str(a) for a in record["answer_ids"] if not is_id(str(a))]


def _surface_audit(twins: list[dict], sources: list[dict], names: dict[str, str], labels: dict[str, list[str]],
                   clone_types: dict[str, str]) -> SubCheck | None:
    """Advisory: the clones' names and answers against their sources', per kind and question type."""
    groups: dict[str, dict[str, dict[str, list[str]]]] = {"kinds": {}, "question_types": {}}
    for c in sorted(names):
        if labels.get(c):
            sides = groups["kinds"].setdefault(kind_of(clone_types[c]) if c in clone_types else "entity",
                                               {"real": [], "clones": []})
            sides["real"].append(labels[c][0])
            sides["clones"].append(names[c])
    by_id = {s["id"]: s for s in sources}
    for t in twins:
        if (source := by_id.get(t["twin_id"])) is not None:
            sides = groups["question_types"].setdefault(question_type(source["question"]), {"real": [], "clones": []})
            sides["real"] += _answer_strings(source)
            sides["clones"] += _answer_strings(t)
    summary, failures = {}, []
    for group, keyed in groups.items():
        summary[group] = {}
        for key, sides in sorted(keyed.items()):
            real, clones = (_surface(sides[side]) if sides[side] else None for side in ("real", "clones"))
            summary[group][key] = {"real": real, "clones": clones}
            if real and clones:
                failures += [{"group": group, "key": key, "rate": rate, "real": real[rate], "clones": clones[rate],
                              "n": [real["n"], clones["n"]]}
                             for rate in SURFACE_RATES if abs(real[rate] - clones[rate]) > SURFACE_GAP]
    if not any(summary.values()):
        return None
    return None, {"advisory": True, "gap": SURFACE_GAP, **summary}, failures


def _direction_rule(inserts: list[str]) -> SubCheck:
    """The direction rule: no insert line has a real subject."""
    real = [line.strip() for line in inserts if line.strip() and not _CLONE_SUBJECT.fullmatch(line.split(" ", 1)[0])]
    return not real, {"lines": sum(bool(line.strip()) for line in inserts), "real_subject_lines": len(real)}, [
        {"line": line} for line in real[:20]]


def _literal_ranges(clone_literals: list[tuple[str, str, str]], ranges: dict[str, LiteralRange]) -> SubCheck:
    failures, checked = [], 0
    for c, p, value in clone_literals:
        if (bounds := ranges.get(p)) is None:
            continue
        checked += 1
        if not (_parses(value, KINDS[bounds.kind]) and _inside(value, bounds)):
            failures.append({"clone": c, "property": p, "value": value, "range": [bounds.lo, bounds.hi]})
    return not failures, {"checked": checked, "out_of_range": len(failures)}, failures


def _before(a: str, b: str) -> bool:
    """``a`` on or before ``b``, on their common precision ("1941" is not after "1941-02-01")."""
    pa, pb = _parts(a), _parts(b)
    n = min(len(pa), len(pb))
    return pa[:n] <= pb[:n]


def _date_pairs(clone_literals: list[tuple[str, str, str]]) -> SubCheck:
    dated = {p for pair in DATE_PAIRS for p in pair}
    values: dict[tuple[str, str], list[str]] = collections.defaultdict(list)
    for c, p, value in clone_literals:
        if p in dated and _parses(value, "date"):
            values[c, p].append(value)
    failures, checked = [], 0
    for c in sorted({c for c, _ in values}):
        for first, second in DATE_PAIRS:
            pairs = [[a, b] for a in values.get((c, first), []) for b in values.get((c, second), [])]
            checked += bool(pairs)
            if bad := [pair for pair in pairs if not _before(*pair)]:
                failures.append({"clone": c, "first": first, "second": second, "values": bad[0]})
    return not failures, {"checked": checked, "inconsistent": len(failures)}, failures


def run_suite(real: Freebase, ext: Freebase, twins: list[dict], sources: list[dict], mapping: dict[str, str],
              judge: Callable[[str], str] | None, out: Path, collides: Callable[[str], bool] | None = None,
              ranges: dict[str, LiteralRange] | None = None,
              attrs: dict[str, dict] | None = None, source_labels: dict[str, list[str]] | None = None,
              clone_types: dict[str, str] | None = None, clone_literals: list[tuple[str, str, str]] | None = None,
              rng: random.Random | None = None, clone_sources: dict[str, str] | None = None,
              unpadded: dict[str, dict[str, int]] | None = None, inserts: list[str] | None = None) -> dict:
    """Run the checks, write ``<out>/report.json`` (and ``human_read.jsonl``), return the report."""
    rng, out = rng or random.Random(0), Path(out)
    seed3, seed4 = rng.getrandbits(64), rng.getrandbits(64)       # checks 3 and 4, drawn whether they run or not
    seed6 = rng.getrandbits(64)                                    # check 6's hardness sample
    out.mkdir(parents=True, exist_ok=True)
    clones = sorted({c for t in twins for c in t["clone_ids"]})
    labels = source_labels if source_labels is not None else _source_labels(real, mapping)
    report = {
        "real_unchanged": real_unchanged(ext, twins, sources),
        "planted_executes": planted_executes(ext, twins),
        "names_unknown": _combined({
            "collisions": _collisions(twins, collides) if collides else None,
            "attribute_test": _attribute_test(ext, clones, judge, attrs, clone_types or {}, random.Random(seed3))
            if judge and attrs else None}),
        "not_renamed_source": not_renamed_source(ext, clones, judge, labels, random.Random(seed4))
        if judge else _entry(None, {"reason": "no judge"}, skipped=True),
        "readable": readable(twins, sources, out),
        "structure_matches": _combined({
            "histograms": _histograms(twins, sources),
            "degrees": _degrees(clones, real, ext, clone_sources, _roles(twins), unpadded, random.Random(seed6))
            if clone_sources else None,
            "direction_rule": _direction_rule(inserts) if inserts is not None else None,
            "literals": _literal_ranges(clone_literals, ranges) if ranges and clone_literals else None,
            "date_pairs": _date_pairs(clone_literals) if clone_literals else None,
            "surface": _surface_audit(twins, sources, ext.labels(clones), labels, clone_types or {})}),
        **{check: _entry(None, pending="harness run") for check in ("prior_empty", "difficulty_matched")}}
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


# step 8: the held-out filter
CLOSED_BOOK = ("Answer the question from your own knowledge, with the entity name or value only, no explanation. "
               "If you do not know, answer 'unknown'.\nQuestion: {question}")


def judge_twin(twin: dict, labels: dict[str, list[str]], facts_of: Callable[[str], list[str]],
               call: Callable[[str], str], template: str) -> dict:
    """One twin's verdict: answered closed-book, or a topic clone's source named from its facts."""
    reply = (call(CLOSED_BOOK.format(question=twin["question"])) or "").strip()
    answered = any(_says(reply, a) for a in twin.get("answers", []) if isinstance(a, str))
    identified = []
    guesses = {}
    for clone in twin.get("topic_entity", {}):
        if not clone.startswith("m.x"):              # a topic kept real (a hub): nothing was renamed
            continue
        facts = facts_of(clone)
        if not facts or not labels.get(clone):
            continue
        guess = (call(template.format(facts="\n".join(facts))) or "").strip()
        guesses[clone] = guess
        if any(_says(guess, name) for name in labels[clone]):
            identified.append(clone)
    return {"id": twin["id"], "closed_book": reply, "answered": answered, "guesses": guesses,
            "identified": identified, "fails": answered or bool(identified)}


def source_labels(mappings: list[dict]) -> dict[str, list[str]]:
    """clone -> the source's label(s), from the merge's mapping.json files."""
    labels: dict[str, list[str]] = {}
    for m in mappings:
        for clone, label in m.get("labels", {}).items():
            labels[clone] = list(label) if isinstance(label, list) else [label]
    return labels


def run(twins: list[dict], labels: dict[str, list[str]], ext: Freebase, call: Callable[[str], str], out: Path,
        threads: int = 16) -> dict:
    template = prompts.GUESS_SOURCE
    def facts_of(clone: str) -> list[str]:
        try:
            return _facts(ext, _edges(ext, clone))
        except (RuntimeError, requests.RequestException):   # past the row cap, or a query the server refuses
            return []
    with ThreadPoolExecutor(threads) as pool:
        verdicts = list(pool.map(lambda t: judge_twin(t, labels, facts_of, call, template), twins))
    out.mkdir(parents=True, exist_ok=True)
    with (out / "heldout.jsonl").open("w", encoding="utf-8") as f:
        for v in verdicts:
            f.write(json.dumps(v, ensure_ascii=False) + "\n")
    failing = [v["id"] for v in verdicts if v["fails"]]
    (out / "heldout_excluded.txt").write_text("".join(i + "\n" for i in failing), encoding="utf-8")
    return {"twins": len(verdicts), "answered": sum(v["answered"] for v in verdicts),
            "identified": sum(bool(v["identified"]) for v in verdicts), "excluded": len(failing)}


# the cross-split pollution check
SPLIT_FILES = ("test", "val", "train")


def _split_file(d: Path) -> Path:
    """The one split file of a checked directory."""
    found = [d / f"{name}.json" for name in SPLIT_FILES if (d / f"{name}.json").is_file()]
    if len(found) != 1:
        raise ValueError(f"{d}: {len(found)} split files (test.json, val.json, train.json): need exactly one")
    return found[0]


def rule_out(checked: Path) -> dict:
    """The rule-outs of the module docstring, from every ``<checked>/<set>/<split>/`` holding a report.json."""
    dirs = sorted(d for d in checked.glob("*/*") if (d / "report.json").is_file())
    owner = {c: (d.parent.name, d.name, t["id"]) for d in dirs for t in _json(_split_file(d))["questions"]
             for c in t["clone_ids"]}
    out: dict = {"ruled_out": collections.defaultdict(dict), "unowned": {}, "unattributed": {}}
    for d in dirs:
        summary = _json(d / "report.json")["real_unchanged"]["summary"]
        where = f"{d.parent.name}/{d.name}"
        if summary.get("unattributed"):
            out["unattributed"][where] = summary["unattributed"]
        for source, clones in sorted(summary.get("foreign", {}).items()):
            for c in clones:
                if c not in owner:
                    out["unowned"].setdefault(where, {}).setdefault(source, []).append(c)
                    continue
                set_, split, twin = owner[c]
                out["ruled_out"][f"{set_}/{split}"].setdefault(twin, [])
                if f"{where}:{source}" not in out["ruled_out"][f"{set_}/{split}"][twin]:
                    out["ruled_out"][f"{set_}/{split}"][twin].append(f"{where}:{source}")
    out["ruled_out"] = dict(out["ruled_out"])
    out["sets"] = sorted({d.parent.name for d in dirs})
    return out


def failing(root: Path) -> dict[str, list[str]]:
    """``<set>/<split>`` -> the checks among 1 and 2 its report.json fails, for every report under ``root``."""
    out = {}
    for path in sorted(root.glob("*/*/report.json")):
        report = _json(path)
        if failed := [c for c in ("real_unchanged", "planted_executes") if report[c]["pass"] is not True]:
            out[f"{path.parent.parent.name}/{path.parent.name}"] = failed
    return out


# the split files every step writes
_DATA = Path(__file__).resolve().parent.parent / "data"
SPLITS = _DATA / "questions"
GRAPH = "freebase-unseen"
CLONE_KEYS = ("names", "types", "attrs", "labels", "unpadded")   # mapping.json's keys by clone id
CLONE_IRI = re.compile(rf"<{re.escape(NS)}(m\.x\w+)>")


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=1, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def write_split(path: Path, set: str, split: str, source: str, questions: list[dict]) -> None:
    """A split file of twins: the dataset ``<set>-unseen``, the graph ``freebase-unseen``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(path, {"dataset": f"{set}-unseen", "graph": GRAPH, "split": split, "source": source,
                       "questions": questions})


def write_meta(path: Path, meta: list[dict]) -> None:
    path.write_text("".join(json.dumps(m, ensure_ascii=False) + "\n" for m in meta), encoding="utf-8")


# the build's meta lines, the mapping it writes
META = ("shape", "clone_ids", "answer_class", "flags")    # a meta line's fields besides twin_id and dropped_reason


def _saved(b: Build) -> dict:
    """mapping.json: source -> clones, clone -> name, type, attributes, labels, unpadded edges, literals."""
    sources = collections.defaultdict(list)
    for s, c in ((s, c) for clone in b.clones for s, c in clone.mapping.items()):
        sources[s].append(c)
    literals = [[c, p, b.literals.get((c, p, old), old)] for clone in b.clones for c, p, old in clone.literals]
    unpadded: dict = collections.defaultdict(lambda: collections.defaultdict(dict))
    for clone in b.clones:
        for (c, r, reason), n in sorted(clone.unpadded.items()):
            unpadded[c][r][reason] = n
    return {"sources": sources, "names": b.names, "types": b.types, "attrs": b.attrs, "labels": b.labels,
            "literals": literals, "unpadded": unpadded}


# clones.nt loaded into the extended copy
def _held(ext: Freebase, ids: list[str]) -> str | None:
    """The first of ``ids`` the extended replica already holds (an earlier build's clone), or None."""
    for batch in _chunks(ids, LABEL_BATCH):
        if rows := ext.select(f"SELECT ?s WHERE {{ {_values(batch)} ?s ?p ?o }} LIMIT 1"):
            return rows[0]["?s"]
    return None


def _load(root: Path, host: str, port: int, nt: Path, name: str) -> None:
    """``bulk_load`` of a copy of ``nt`` in the replica's ``load/`` folder, named per build."""
    refuse_source(root)
    if not Path(nt).resolve().is_relative_to(Path(root).resolve()):
        (Path(root) / "load").mkdir(exist_ok=True)
        nt = Path(shutil.copy(nt, Path(root) / "load" / name))
    bulk_load(root, host, port, nt)


def _load_and_check(b: Build, args, ext: Freebase, judge, loader, questions: list[dict], next_id: Path,
                    seed: int) -> dict:
    """Load the clones unless the replica holds one of their ids already, advance ``next_id``, run the suite."""
    if held := _held(ext, [c for clone in b.clones for c in clone.mapping.values()]):
        raise RuntimeError(f"{args.ext} already holds {held}, a clone of an earlier build: check {next_id} or "
                           "give --id-start")
    loader(args.replica_root, args.isql_host, args.isql_port, args.out / "clones.nt",
           f"clones-{args.out.name}-{b.start}.nt")
    if next_id:
        next_id.parent.mkdir(parents=True, exist_ok=True)
        next_id.write_text(f"{b.next_id}\n", encoding="utf-8")
    return _check(b.real, ext, b.twins, questions, _saved(b), judge, args.out, seed, b.ranges,
                  (args.out / "clones.nt").read_text(encoding="utf-8").splitlines())


# step 7, validated: the suite, its polluters, the twins it rules out
POLLUTED, MISMATCH = "polluted real question", "planted gold mismatch"     # the suite's checks 1 and 2 rule a twin out
PADDED = "padding satisfied the gold"   # check 2: a node or literal padding added is among the wrong answers
CLONE_ID = re.compile(r"m\.x\w+")


def chat_judge(url: str, model: str, sleep: Callable[[float], None] = time.sleep) -> Callable[[str], str]:
    """The suite's judge: one chat completion at temperature 0, retried on 429 / 5xx as ``openai_call``."""
    url = url if url.rstrip("/").endswith("/chat/completions") else url.rstrip("/") + "/chat/completions"

    def judge(prompt: str) -> str:
        body = {"model": model, "temperature": 0, "max_tokens": 64, "messages": [{"role": "user", "content": prompt}]}
        for wait in (*BACKOFF, None):
            try:
                reply = requests.post(url, json=body, timeout=60)
            except requests.RequestException as error:
                failure = error
            else:
                if reply.status_code != 429 and reply.status_code < 500:
                    reply.raise_for_status()                  # any other 4xx is not worth a retry
                    return reply.json()["choices"][0]["message"]["content"] or ""
                failure = requests.HTTPError(f"{reply.status_code} from {url}", response=reply)
            if wait is None:
                raise failure
            sleep(wait)
    return judge


def _touched(ext: Freebase, source: dict) -> set[str]:
    """The clone ids behind a source's changed answers on the replica."""
    expected, executed = _execute(ext, source)
    var = parse(source["gold"]["query"]).select
    rows = ext.select(expose(source["gold"]["query"]))
    clones = [{v for v in row.values() if CLONE_ID.fullmatch(v)} for row in rows]
    return (set().union(*(c for row, c in zip(rows, clones) if row.get(var) in executed - expected))
            or set().union(*clones))


def attribute(report: dict, ext: Freebase, twins: list[dict], sources: list[dict]) -> dict:
    """Check 1's polluters, added to its summary: the twins whose clones changed a real question's answers."""
    owner = {c: t["id"] for t in twins for c in t["clone_ids"]}
    by_id = {q["id"]: q for q in sources}
    polluters, foreign, unattributed = {}, {}, []
    for failure in report["real_unchanged"]["failures"]:
        try:
            clones = set() if "error" in failure else _touched(ext, by_id[failure["id"]])
        except (RuntimeError, requests.RequestException):
            clones = set()
        if found := sorted({owner[c] for c in clones if c in owner}):
            polluters[failure["id"]] = found
        if alien := sorted(c for c in clones if c not in owner):
            foreign[failure["id"]] = alien
        if not clones:
            unattributed.append(failure["id"])
    report["real_unchanged"]["summary"].update(
        polluters=polluters, polluter_twin_ids=sorted({t for found in polluters.values() for t in found}),
        ambiguous=[s for s, found in polluters.items() if len(found) > 1], foreign=foreign, unattributed=unattributed)
    return report


def excluded(report: dict, padded: dict[str, set[str]] | None = None) -> dict[str, str]:
    """The twins the suite rules out, record id -> drop reason (``POLLUTED``, ``PADDED``, ``MISMATCH``)."""
    summary = report["real_unchanged"]["summary"]
    if "polluter_twin_ids" not in summary:
        raise ValueError("check 1 has no polluter attribution (a v0 report): re-run build --suite-only")
    reasons = dict.fromkeys(summary["polluter_twin_ids"], POLLUTED)
    for failure in report["planted_executes"]["failures"]:
        wrong = {*failure.get("extra", ()), *failure.get("missing", ())}
        reasons.setdefault(failure["id"], PADDED if wrong & (padded or {}).get(failure["id"], set()) else MISMATCH)
    return reasons


def _written(splits: Path, files: Path, split: str) -> tuple[list[dict], list[dict], dict, list[str]]:
    """What a build wrote: the twins and the meta lines in ``splits``, the mapping and the insert lines in ``files``."""
    return (json.loads((splits / f"{split}.json").read_text(encoding="utf-8"))["questions"],
            [json.loads(line) for line in (splits / f"{split}_meta.jsonl").read_text(encoding="utf-8").splitlines()],
            json.loads((files / "mapping.json").read_text(encoding="utf-8")),
            (files / "clones.nt").read_text(encoding="utf-8").splitlines(keepends=True))


def _exclude(reasons: dict[str, str], twins: list[dict], meta: list[dict], saved: dict, lines: list[str],
             splits: Path, files: Path, dst: Path, split: str,
             always: bool = False) -> tuple[list[dict], list[dict], dict, list[str]]:
    """The written files without the twins ``reasons`` names, rewritten to ``dst`` when it names any."""
    gone = {t["twin_id"]: reasons[t["id"]] for t in twins if t["id"] in reasons}
    if not gone and not always:
        return twins, meta, saved, lines
    removed = {c for t in twins if t["twin_id"] in gone for c in t["clone_ids"]}
    twins = [t for t in twins if t["twin_id"] not in gone]
    meta = [{**m, **dict.fromkeys(META), "dropped_reason": gone[m["twin_id"]]} if m["twin_id"] in gone else m
            for m in meta]
    saved = {**saved, "sources": {s: kept for s, clones in saved["sources"].items()
                                  if (kept := [c for c in clones if c not in removed])},
             **{key: {c: v for c, v in saved.get(key, {}).items() if c not in removed} for key in CLONE_KEYS},
             "literals": [literal for literal in saved["literals"] if literal[0] not in removed]}
    lines = [line for line in lines if not removed & set(CLONE_IRI.findall(line))]
    drops = json.loads((files / "drops.json").read_text(encoding="utf-8"))
    for source, reason in gone.items():
        entry = drops.setdefault(reason, {"count": 0, "ids": []})
        entry["count"] += 1
        entry["ids"].append(source)
    split_file = json.loads((splits / f"{split}.json").read_text(encoding="utf-8"))
    dst.mkdir(parents=True, exist_ok=True)
    _write_json(dst / f"{split}.json", {**split_file, "questions": twins})
    write_meta(dst / f"{split}_meta.jsonl", meta)
    _write_json(dst / "drops.json", drops)
    _write_json(dst / "mapping.json", saved)
    (dst / "clones.nt").write_text("".join(lines), encoding="utf-8")
    return twins, meta, saved, lines


def _counts(twins: list[dict], meta: list[dict], saved: dict, lines: list[str]) -> dict:
    """build.json's counts, read off the written files."""
    dropped = collections.Counter(m["dropped_reason"] for m in meta if m["dropped_reason"])
    return {"read": len(meta), "built": len(meta) - sum(dropped.values()), "dropped": dict(dropped),
            "raw": len(twins), "mod_entities": len({_template(t["gold"]["query"]) for t in twins}),
            "clones": sum(map(len, saved["sources"].values())), "triples": len(lines)}


def _check(real: Freebase, ext: Freebase, twins: list[dict], questions: list[dict], saved: dict, judge, out: Path,
           seed: int, ranges: dict | None = None, inserts: list[str] | None = None) -> dict:
    """Run the suite over the twins and their sources, then attribute check 1's polluters."""
    ranges = {} if ranges is None else ranges
    errors = out / "errors.log"
    by_id = {q["id"]: q for q in questions}
    sources = [by_id[t["twin_id"]] for t in twins]
    types, literals = saved["types"], [tuple(literal) for literal in saved["literals"]]
    report = run_suite(real, ext, twins, sources,
                     {s: c for s, clones in saved["sources"].items() for c in clones}, judge, out,
                     collides=collides_remote(real),
                     ranges={p: r for p in {p for _, p, _ in literals}
                             if (r := _cached(ranges, p, real.literal_range, errors))},
                     attrs=saved["attrs"], source_labels=saved["labels"], clone_types=types, clone_literals=literals,
                     rng=random.Random(seed),
                     clone_sources={c: s for s, clones in saved["sources"].items() for c in clones},
                     unpadded={c: {r: sum(reasons.values()) for r, reasons in relations.items()}
                               for c, relations in saved.get("unpadded", {}).items()}, inserts=inserts)
    attribute(report, ext, twins, sources)
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def _print_checks(report: dict) -> None:
    print("[build] checks: " + ", ".join(f"{name} {check['pass']}" for name, check in report.items()), flush=True)
    check = report["real_unchanged"]["summary"]
    if check.get("polluters") or check.get("foreign") or check.get("unattributed"):
        print(f"[build]   check 1: {len(check['polluter_twin_ids'])} polluting twins ruled out, "
              f"{len(check['ambiguous'])} sources changed by several twins (all ruled out), {len(check['foreign'])} "
              f"by another build's clones, {len(check['unattributed'])} unattributed (none ruled out)", flush=True)


# the trim: the validated twins cut to the release
SIZES = {"test": 1000, "val": 200, "train": 2000}
SOURCES = ("test", "val", "train")             # the merged splits trim reads, when present
VAL_WAVE = "val-wave"                          # --val-from: train questions past the first wave, built on their own
HELD_OUT, HELD_OUT_BY = "held-out filter", "heldout"
CROSS_SPLIT, CROSS_SPLIT_BY = "polluted real question (cross-split)", "pollution"   # the pollution step's rule-outs
VAL_SOURCE = "train-extra"                     # val without --val-from: the train twins after the released train


class Unresolved(ValueError):
    """A report trim reads holds pollution nothing has ruled out: the release is refused."""


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def allocate(shares: Mapping[str, int], n: int) -> dict[str, int]:
    """``n`` slots over the shapes in proportion to ``shares``, by largest remainder."""
    total = sum(shares.values())
    if not total:
        return dict.fromkeys(shares, 0)
    quota = {s: c * n // total for s, c in shares.items()}
    by_remainder = sorted(shares, key=lambda s: (-(shares[s] * n % total), -shares[s], s))
    for s in by_remainder[:n - sum(quota.values())]:
        quota[s] += 1
    return quota


def stratum(shape: str) -> str:
    """A shape's stratum: its hops, operations and projection, without the answer type."""
    return "|".join(shape.split("|")[:3])


def stratify(shares: Mapping[str, int], twins: Mapping[str, int], n: int) -> tuple[dict[str, int], dict[str, int]]:
    """Each stratum's request and allotment, every shortfall given to strata with twins to spare."""
    requested = allocate(shares, n)
    allotted = {s: min(requested[s], twins.get(s, 0)) for s in shares}
    while (left := n - sum(allotted.values())) > 0:
        if not (room := {s: shares[s] for s in shares if twins.get(s, 0) > allotted[s]}):
            break
        for s, k in allocate(room, left).items():
            allotted[s] += min(k, twins[s] - allotted[s])
    return requested, allotted


def assign(twins: dict[str, list[dict]], sizes: dict[str, int],
           source_test: list[dict]) -> tuple[dict[tuple[str, str], str | None], dict[str, dict]]:
    """(source split, twin id) -> release split, and the test's allocation per stratum."""
    shares = collections.Counter(stratum(q["gold"]["shape"]) for q in source_test
                                 if (q.get("gold") or {}).get("shape"))
    by_stratum: dict[str, list[dict]] = collections.defaultdict(list)
    for t in twins.get("test", []):
        by_stratum[stratum(t["gold"]["shape"])].append(t)
    requested, allotted = stratify(shares, {s: len(ts) for s, ts in by_stratum.items()}, sizes["test"])
    allocation = {s: {"source": shares[s], "requested": requested[s], "allotted": allotted[s],
                      "short": max(requested[s] - allotted[s], 0), "twins": len(by_stratum[s])}
                  for s in sorted(shares, key=lambda s: (-shares[s], s))}
    release: dict[tuple[str, str], str | None] = {(s, t["id"]): None for s, ts in twins.items() for t in ts}
    release.update({("test", t["id"]): "test" for s, n in allotted.items() for t in by_stratum[s][:n]})
    in_test = {q["id"] for q in source_test}
    train = [t for t in twins.get("train", []) if t["twin_id"] not in in_test]
    release.update({("train", t["id"]): "train" for t in train[:sizes["train"]]})
    if VAL_WAVE in twins:                       # the val wave: its own train questions, never the first wave's
        taken = {t["id"] for t in train[:sizes["train"]]}
        wave = [t for t in twins[VAL_WAVE] if t["twin_id"] not in in_test and t["id"] not in taken]
        release.update({(VAL_WAVE, t["id"]): "val" for t in wave[:sizes["val"]]})
    else:
        release.update({("train", t["id"]): "val" for t in train[sizes["train"]:sizes["train"] + sizes["val"]]})
    return release, allocation


def _reports(dirs: dict[str, Path], polluted: set[str] | None) -> None:
    """``Unresolved`` when a report holds check-1 pollution no step resolved."""
    for d in dirs.values():
        if not (d / "report.json").is_file():
            continue
        summary = _json(d / "report.json")["real_unchanged"]["summary"]
        if summary.get("unattributed"):
            raise Unresolved(f"{d}/report.json: unattributed pollution of {summary['unattributed']}: the release is "
                             "blocked until it is resolved")
        if summary.get("foreign") and polluted is None:
            raise Unresolved(f"{d}/report.json: pollution by other splits' clones ({len(summary['foreign'])} "
                             "sources): run python -m unseen_builder.main pollution and pass its list as --polluted")


def trim(set_: str, merged: Path, out: Path, sizes: dict[str, int], splits: Path = SPLITS,
         exclude: set[str] = frozenset(), val_from: Path | None = None,
         polluted: set[str] | None = None) -> dict[str, int]:
    """Cut the merged twins to the release, written to ``out``; its counts."""
    unseen = splits / f"{set_}-unseen"
    inputs = {Path(merged).resolve(), unseen.resolve()} | ({Path(val_from).resolve()} if val_from else set())
    if Path(out).resolve() in inputs:
        raise ValueError(f"--out {out} is an input: trim never rewrites the merged files")
    files = {s: merged / s for s in SOURCES} | ({VAL_WAVE: Path(val_from)} if val_from else {})   # build files
    where = {s: merged / s if (merged / s / f"{s}.json").is_file() else unseen for s in SOURCES}  # split files
    where.update({VAL_WAVE: Path(val_from)} if val_from else {})
    name = {**{s: s for s in SOURCES}, VAL_WAVE: "train"}       # the val wave is a build of the train split
    parts = {s: _json(where[s] / f"{name[s]}.json") for s in files if (where[s] / f"{name[s]}.json").is_file()}
    if val_from and VAL_WAVE not in parts:
        raise ValueError(f"--val-from {val_from}: no train.json (the val wave is a build of the train split)")
    _reports({s: files[s] for s in parts}, polluted)
    meta = {(s, m["twin_id"]): m for s in parts for m in _jsonl(where[s] / f"{name[s]}_meta.jsonl")}
    source_test = _json(splits / set_ / "test.json")["questions"]
    exclusions = {**{i: (HELD_OUT_BY, HELD_OUT) for i in exclude},
                  **{i: (CROSS_SPLIT_BY, CROSS_SPLIT) for i in polluted or ()}}     # a polluter is recorded so
    held = {s: [t for t in p["questions"] if t["id"] in exclusions] for s, p in parts.items()}
    pools = {s: [t for t in p["questions"] if t["id"] not in exclusions] for s, p in parts.items()}
    release, allocation = assign(pools, sizes, source_test)
    released = {split: [(s, t) for s in parts for t in pools[s] if release[s, t["id"]] == split] for split in SIZES}

    out.mkdir(parents=True, exist_ok=True)
    triples = 0
    for split, items in released.items():
        if not items:
            continue
        [source] = {s for s, _ in items}                 # test reads test; train, train; val, train or the val wave
        qs = [t for _, t in items]
        kept = {c for t in qs for c in t["clone_ids"]}
        lines = [line for line in (files[source] / "clones.nt").read_text(encoding="utf-8").splitlines(keepends=True)
                 if kept & set(CLONE_IRI.findall(line))]
        drops = _json(files[source] / "drops.json")
        for t in held[source]:
            entry = drops.setdefault(exclusions[t["id"]][1], {"count": 0, "ids": []})
            entry["count"] += 1
            entry["ids"].append(t["twin_id"])
        d = out / split
        write_split(d / f"{split}.json", set_, split,
                    parts[source]["source"] + f"; released {len(qs)} of the validated twins", qs)
        write_meta(d / f"{split}_meta.jsonl", [meta[(source, t["twin_id"])] for t in qs])
        (d / "clones.nt").write_text("".join(lines), encoding="utf-8")
        _write_json(d / "mapping.json", _cut(_json(files[source] / "mapping.json"), kept))
        _write_json(d / "drops.json", drops)
        shutil.copyfile(files[source] / "build.json", d / "build.json")
        triples += len(lines)
    with (out / "raw.jsonl").open("w", encoding="utf-8") as f:
        for s, p in parts.items():
            for t in p["questions"]:
                by, reason = exclusions.get(t["id"], (None, None))
                f.write(json.dumps({**t, "meta": meta.get((s, t["twin_id"])), "source_split": s,
                                    "release_split": None if by else release[s, t["id"]],
                                    "excluded_by": by, "dropped_reason": reason}, ensure_ascii=False) + "\n")
    counts = {split: len(items) for split, items in released.items()}
    counts.update(surplus=sum(r is None for r in release.values()), excluded=sum(map(len, held.values())),
                  triples=triples)
    in_test = {q["id"] for q in source_test}
    _write_json(out / "trim.json", {
        "set": set_, "sizes": sizes, "released": {split: counts[split] for split in SIZES},
        "val_source": f"{VAL_WAVE} {val_from}" if val_from else VAL_SOURCE, "val_short": sizes["val"] - counts["val"],
        "test_stratum": "hops|operations|projection", "test_allocation": allocation,
        "test_short": {"strata": sum(a["short"] > 0 for a in allocation.values()),
                       "twins": sum(a["short"] for a in allocation.values()),
                       "unplaced": sizes["test"] - counts["test"]}, "surplus": counts["surplus"],
        "excluded": {s: dict(collections.Counter(exclusions[t["id"]][1] for t in ts)) for s, ts in held.items()},
        "excluded_unmatched": sorted(set(exclusions) - {t["id"] for p in parts.values() for t in p["questions"]}),
        "train_in_source_test": [t["id"] for t in pools.get("train", []) if t["twin_id"] in in_test],
        "triples": triples})
    return counts


def _cut(mapping: dict, kept: set[str]) -> dict:
    """The merge's mapping.json restricted to ``kept`` clones (the shape the merge step writes)."""
    cut = {"sources": {s: [c for c in clones if c in kept] for s, clones in mapping["sources"].items()},
           "literals": [literal for literal in mapping["literals"] if literal[0] in kept]}
    cut["sources"] = {s: clones for s, clones in cut["sources"].items() if clones}
    for key in CLONE_KEYS:
        cut[key] = {c: v for c, v in mapping.get(key, {}).items() if c in kept}
    return {**mapping, **cut}
