"""The deterministic support rate; the gold only splits items into correct and wrong (README: graph-trace support)."""
from __future__ import annotations

import dataclasses
import datetime
import random
import re
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Optional

from harness.metrics.groundedness.records import MONTH_NUMBER, AnswerRecord, ResultItem, canonical_id
from harness.metrics.reasoning import label_key as normalize
from harness.metrics.reasoning import mentions, score_text

STRICT, LENIENT, UNSUPPORTED = "strict", "lenient", "unsupported"

#: Wh-words and the class each asks for (None: no constraint). "whose" is left out: it is mostly relative ("the film
#: whose director ..."), where the answer is not a person.
WH_CLASSES = {"how many": "number", "who": "person_org", "whom": "person_org", "where": "location", "when": "date",
              "what": None, "which": None}
PREPOSITIONS = ("in", "for", "from", "at", "to", "by", "with", "of", "on", "during")   # "In which city ..."
_WH = "|".join(WH_CLASSES)
_WH_AT_START = re.compile(rf"\W*(?:(?:{'|'.join(PREPOSITIONS)})\s+)?({_WH})\b")
#: Types a class rejects: equal to an entry, or starting with an entry that ends in ".". Freebase domains and types
#: first, then plain class names as other graphs write them.
DENY = {
    "location": (  # person and organization types never answer "where"
        "people.", "organization.", "business.", "royalty.", "sports.sports_team", "sports.pro_athlete",
        "music.artist", "music.group_member", "music.musical_group", "film.actor", "film.director", "film.producer",
        "film.writer", "tv.tv_actor", "book.author", "fictional_universe.fictional_character", "award.award_winner",
        "human", "person", "organization", "company", "band", "team", "actor", "director", "writer", "character"),
    "person_org": (  # location types never answer "who"
        "location.", "geography.", "travel.",
        "location", "place", "country", "city", "town", "state", "region", "continent", "island"),
}
UNTYPED = {"entity"}                                       # MetaQA's single node label carries no class

#: The one unit word a number may carry ("97 minute"); a closed list, so "21 Savage" is not a number.
UNITS = {
    "second", "seconds", "sec", "secs", "minute", "minutes", "min", "mins", "hour", "hours", "hr", "hrs", "day",
    "days", "week", "weeks", "month", "months", "year", "years", "yr", "yrs",
    "metre", "metres", "meter", "meters", "m", "kilometre", "kilometres", "kilometer", "kilometers", "km",
    "centimetre", "centimetres", "centimeter", "centimeters", "cm", "millimetre", "millimetres", "millimeter",
    "millimeters", "mm", "mile", "miles", "mi", "foot", "feet", "ft", "inch", "inches", "yard", "yards",
    "gram", "grams", "g", "kilogram", "kilograms", "kg", "pound", "pounds", "lb", "lbs", "tonne", "tonnes", "ton",
    "tons", "percent", "%",
}
_NUMBER = r"([-+]?(?:\d{1,3}(?:,\d{3})+|\d{1,3}(?: \d{3})+|\d+)(?:\.\d+)?(?:e[-+]?\d+)?)"
_MONTH = r"(" + "|".join(sorted(MONTH_NUMBER, key=len, reverse=True)) + r")\.?"
_DAY = r"(\d{1,2})(?:st|nd|rd|th)?"


def question_type(question: str) -> Optional[str]:
    """The one operation the question's text selects, by the first rule that fires; None: membership only."""
    q = question.lower().strip()
    if re.search(r"\bhow long\b|\bhow many (years|months|days)\b", q):
        return "date_diff"
    if re.search(r"\bhow many\b|\bnumber of\b", q):
        return "count"
    if re.search(r"\baverage\b|\bthe mean\b", q):
        return "average"
    if re.search(r"\btotal\b|\bin all\b", q):
        return "sum"
    return None


def date_unit(question: str) -> str:
    """The unit a date difference is asked in: 'months', 'days', or 'years' (the default)."""
    m = re.search(r"\b(month|day)s?\b", question.lower())
    return m.group(1) + "s" if m else "years"


def as_number(s) -> Optional[Decimal]:
    """The number a string writes ("30,147", "-3", "3e7", one unit word allowed), or None."""
    m = re.fullmatch(_NUMBER + r"(?:\s*([a-z]+|%))?", str(s).strip().lower())
    if not m or (m.group(2) is not None and m.group(2) not in UNITS):
        return None
    return Decimal(m.group(1).replace(",", "").replace(" ", ""))


def as_date(s) -> Optional[tuple]:
    """(year,), (year, month) or (year, month, day) from an ISO or written date; None, also when impossible."""
    t = str(s).strip().lower()
    if m := re.fullmatch(r"(\d{4})(?:[- ](\d{1,2})(?:[- ](\d{1,2}))?)?(?:t[\d:.]+(?:z|[+-]\d{2}:?\d{2})?)?", t):
        parts = [int(x) for x in m.groups() if x is not None]
    elif m := re.fullmatch(rf"{_MONTH}\s+{_DAY},?\s+(\d{{4}})", t):
        parts = [int(m.group(3)), MONTH_NUMBER[m.group(1)], int(m.group(2))]
    elif m := re.fullmatch(rf"{_DAY}\s+(?:of\s+)?{_MONTH},?\s+(\d{{4}})", t):
        parts = [int(m.group(3)), MONTH_NUMBER[m.group(2)], int(m.group(1))]
    elif m := re.fullmatch(rf"{_MONTH},?\s+(\d{{4}})", t):
        parts = [int(m.group(2)), MONTH_NUMBER[m.group(1)]]
    else:
        return None
    try:
        datetime.date(parts[0], *(parts[1:] + [1, 1])[:2])
    except ValueError:
        return None
    return tuple(parts)


def yes_no(s) -> Optional[str]:
    t = str(s).strip().lower()
    return "yes" if t in ("yes", "true") else "no" if t in ("no", "false") else None


def same_number_rounded(answer: Decimal, computed: Decimal) -> bool:
    """A computed number rounded half up to the answer's decimal places equals the answer ("1e7" must be exact)."""
    exp = answer.as_tuple().exponent
    if exp > 0:
        return Decimal(computed) == answer
    try:
        return Decimal(computed).quantize(Decimal((0, (1,), exp)), rounding=ROUND_HALF_UP) == answer
    except InvalidOperation:
        return False


def total(values: dict) -> Optional[Decimal]:
    """The sum of {entity: raw value}, if there are any and all are numbers."""
    nums = [as_number(v) for v in values.values()]
    return sum(nums) if nums and None not in nums else None


def average(values: dict) -> Optional[Decimal]:
    s = total(values)
    return s / len(values) if s is not None else None


def date_diff(a: str, b: str, unit: str) -> Optional[int]:
    """|a - b| in whole years (year minus year), months or days; None if a date lacks the precision the unit needs."""
    da, db = as_date(a), as_date(b)
    if da is None or db is None:
        return None
    if unit == "years":
        return abs(da[0] - db[0])
    if unit == "months":
        return abs(da[0] * 12 + da[1] - db[0] * 12 - db[1]) if len(da) > 1 and len(db) > 1 else None
    if len(da) < 3 or len(db) < 3:
        return None
    return abs((datetime.date(*da) - datetime.date(*db)).days)


def rooted_records(calls, topic) -> list:
    """The rooted records, in seq order: every anchor a topic entity or an earlier rooted result."""
    reached, out = set(topic), []
    for c in sorted(calls, key=lambda c: c.seq):
        if c.anchor and all(a in reached for a in c.anchor):
            out.append(c)
            reached.update(r.id for r in c.result if not r.is_value)
    return out


def with_labels(calls) -> list:
    """The calls (by seq) with every unlabelled entity result given the first label any record gives its id."""
    calls = sorted(calls, key=lambda c: c.seq)
    names: dict = {}
    for c in calls:
        for r in c.result:
            if not r.is_value and r.label:
                names.setdefault(r.id, r.label)
    return [dataclasses.replace(c, result=tuple(
        dataclasses.replace(r, label=names.get(r.id)) if not r.is_value and not r.label else r for r in c.result))
        for c in calls]


def answer_class(question: str) -> Optional[str]:
    """The class of the wh-word at the start (after an optional preposition), else of the last wh-word."""
    q = question.lower()
    m = _WH_AT_START.match(q)
    if m:
        return WH_CLASSES[m.group(1)]
    last = list(re.finditer(rf"\b({_WH})\b", q))
    return WH_CLASSES[last[-1].group(1)] if last else None


def type_fits(r: ResultItem, cls: Optional[str]) -> bool:
    """The type guard: whether a result may stand for an answer of the class."""
    if cls is None:
        return True
    if r.is_value:
        if cls == "date":
            return as_date(r.value) is not None
        if cls == "number":
            return as_number(r.value) is not None
        return as_number(r.value) is None and as_date(r.value) is None
    t = (r.type or "").strip().lower()
    if not t or t in UNTYPED:
        return True
    if cls in ("date", "number"):
        return False
    return not any(t == p or (p.endswith(".") and t.startswith(p)) for p in DENY[cls])


def same_literal(item: str, value: str) -> bool:
    """Equal as numbers, else as dates (the answer may be coarser), else as normalized text."""
    a, v = as_number(item), as_number(value)
    if a is not None and v is not None:
        return a == v
    da, dv = as_date(item), as_date(value)
    if da is not None and dv is not None:
        return dv[: len(da)] == da
    if (a, v, da, dv) != (None, None, None, None):
        return False
    return normalize(item) != "" and normalize(item) == normalize(value)


def matches(item: str, item_id: Optional[str], r: ResultItem, cls: Optional[str]) -> bool:
    """An item with an id matches that entity only; else a literal or a normalized label, passing the type guard."""
    if item_id is not None:
        return not r.is_value and r.id == item_id
    if r.is_value:
        return same_literal(item, r.value) and type_fits(r, cls)
    if r.id == canonical_id(item):        # the answer printed the entity's id itself (an unnamed entity, a MetaQA node)
        return True
    return (r.label is not None and normalize(item) != "" and normalize(r.label) == normalize(item)
            and type_fits(r, cls))


def is_yes_no(item: str, item_id: Optional[str]) -> bool:
    return item_id is None and yes_no(item) is not None


def is_literal(item: str, item_id: Optional[str]) -> bool:
    """A number, a date or yes / no, rather than an entity."""
    return item_id is None and (as_number(item) is not None or as_date(item) is not None or yes_no(item) is not None)


def operation_outputs(question: str, topic, rooted) -> list:
    """The numbers the question's one operation computes over rooted results (count, sum, average, date_diff)."""
    op = question_type(question)
    sets = [ids for ids in (sorted({r.id for r in c.result if not r.is_value}) for c in
                            sorted(rooted, key=lambda c: c.seq)) if ids]
    last = sets[-1] if sets else []
    props: dict = {}
    for c in sorted(rooted, key=lambda c: c.seq):
        for r in c.result:
            if r.is_value and r.prop and r.of:
                props.setdefault(r.prop, {}).setdefault(r.of, r.value)
    nums: list = []
    if op == "count" and last:
        nums = [Decimal(len(set(last)))]
    elif op in ("sum", "average"):
        for p in sorted(props):
            x = (total if op == "sum" else average)({e: props[p][e] for e in last if e in props[p]})
            if x is not None:
                nums.append(x)
    elif op == "date_diff":
        unit = date_unit(question)
        dated = [(of, v) for p in sorted(props) for of, v in props[p].items() if as_date(v) is not None]
        for i, (of_a, a) in enumerate(dated):
            for of_b, b in dated[i + 1:]:
                if of_a == of_b or (of_a in topic and of_b in topic):
                    d = date_diff(a, b, unit)
                    if d is not None:
                        nums.append(Decimal(d))
    return nums


def _index(results) -> tuple:
    """Results by id, by normalized label, and the literals, so matching an item is a lookup, not a scan."""
    by_id, by_label, values = {}, {}, []
    for r in results:
        if r.is_value:
            values.append(r)
        else:
            by_id.setdefault(r.id, []).append(r)
            if r.label is not None:
                by_label.setdefault(normalize(r.label), []).append(r)
    return by_id, by_label, values


def _candidates(index, item: str, item_id) -> list:
    by_id, by_label, values = index
    if item_id is not None:
        return by_id.get(item_id, [])
    return by_id.get(canonical_id(item), []) + by_label.get(normalize(item), []) + values


def item_verdicts(question: str, items, item_ids, topic, calls) -> list:
    """(verdict, type_rejected) per item: unsupported, strict or lenient (README: graph-trace support); no gold."""
    calls = with_labels(calls)
    rooted = rooted_records(calls, topic)
    cls = answer_class(question)
    out_nums = operation_outputs(question, topic, rooted)
    rooted_index = _index([r for c in rooted for r in c.result])
    anchored_index = _index([r for c in calls if c.anchor for r in c.result])

    def strict_under(item, iid, guard):
        n = as_number(item) if iid is None else None
        return (any(matches(item, iid, r, guard) for r in _candidates(rooted_index, item, iid))
                or (n is not None and any(same_number_rounded(n, x) for x in out_nums)))

    out = []
    for item, iid in zip(items, item_ids):
        if is_yes_no(item, iid):
            out.append((UNSUPPORTED, False))
        elif strict_under(item, iid, cls):
            out.append((STRICT, False))
        else:
            lenient = any(matches(item, iid, r, cls) for r in _candidates(anchored_index, item, iid))
            out.append((LENIENT if lenient else UNSUPPORTED, strict_under(item, iid, None)))
    return out


def item_correct(item: str, item_id: Optional[str], answer: AnswerRecord) -> Optional[bool]:
    """Whether an item names a gold (whole-word rule or an equal id); None without gold. The gold's only use."""
    golds = [normalize(g) for g in (answer.gold or answer.gold_ids) if normalize(g)]
    if not golds and not answer.gold_ids:
        return None
    return (item_id is not None and item_id in answer.gold_ids) or any(mentions(normalize(item), g) for g in golds)


def result_size(calls) -> int:
    """|R|: distinct result items over all of a question's calls."""
    return len({("e", r.id) if not r.is_value else ("v", r.value, r.prop, r.of) for c in calls for r in c.result})


def grounded_hit(answer: AnswerRecord, calls) -> tuple:
    """(Hit, grounded Hit): a gold named in the answer, and also strict in the system's rooted records."""
    text = answer.answer_text or "; ".join(answer.items)
    hit = score_text(text, list(answer.gold))[0] if answer.gold else 0.0
    if not hit:
        return 0.0, 0.0
    named = [g for g in answer.gold if mentions(normalize(text), normalize(g))]
    verdicts = item_verdicts(answer.question, named, [None] * len(named), answer.topic, calls)
    rooted_ids = {r.id for c in rooted_records(calls, answer.topic) for r in c.result if not r.is_value}
    # gold ids are not listed in the same order as gold names, so an id counts if any gold id was returned rooted
    by_id = bool(set(answer.gold_ids) & rooted_ids) and len(answer.gold_ids) == 1
    return 1.0, float(by_id or any(v == STRICT for v, _ in verdicts))


def evaluate_question(answer: AnswerRecord, calls) -> dict:
    """Per-item verdicts and flags, the question's support (strict items / items), |R|, Hit and grounded Hit."""
    verdicts = item_verdicts(answer.question, answer.items, answer.item_ids, answer.topic, calls)
    items = [{"item": it, "item_id": iid, "verdict": v, "strict": v == STRICT, "found": v != UNSUPPORTED,
              "type_rejected": rejected, "literal": is_literal(it, iid), "yes_no": is_yes_no(it, iid),
              "correct": item_correct(it, iid, answer)}
             for it, iid, (v, rejected) in zip(answer.items, answer.item_ids, verdicts)]
    hit, grounded = grounded_hit(answer, calls)
    return {"qid": answer.qid, "items": items, "support": sum(i["strict"] for i in items) / len(items),
            "R": result_size(calls), "hit": hit, "grounded_hit": grounded}


def _mean(xs) -> Optional[float]:
    return sum(xs) / len(xs) if xs else None


def summarize(results) -> dict:
    """The set's rates, each a mean over questions: support, found, item shares, |R|, Hit, grounded Hit."""
    def share(keep):
        return _mean([sum(bool(keep(i)) for i in q["items"]) / len(q["items"]) for q in results])

    def rate(keep):
        per_q = [s for s in ([i for i in q["items"] if keep(i)] for q in results) if s]
        return {"support_rate": _mean([sum(i["strict"] for i in s) / len(s) for s in per_q]),
                "n_questions": len(per_q), "n_items": sum(len(s) for s in per_q)}

    n_items = lambda keep: sum(bool(keep(i)) for q in results for i in q["items"])
    return {"n_questions": len(results), "n_items": n_items(lambda i: True),
            "support_rate": _mean([q["support"] for q in results]),
            "found_rate": share(lambda i: i["found"]),
            "lenient_share": share(lambda i: i["verdict"] == LENIENT),
            "unsupported_share": share(lambda i: i["verdict"] == UNSUPPORTED),
            "yes_no_share": share(lambda i: i["yes_no"]), "yes_no_items": n_items(lambda i: i["yes_no"]),
            "type_rejected_share": share(lambda i: i["type_rejected"]),
            "type_rejected_items": n_items(lambda i: i["type_rejected"]),
            "literal": rate(lambda i: i["literal"]),
            "correct": rate(lambda i: i["correct"] is True),
            "wrong": rate(lambda i: i["correct"] is False),
            "items_without_gold": n_items(lambda i: i["correct"] is None),
            "R_mean": _mean([q["R"] for q in results]),
            "hit": _mean([q["hit"] for q in results]), "grounded_hit": _mean([q["grounded_hit"] for q in results])}


def shown_only(calls) -> list:
    """The calls without the results marked shown = false (unknown is kept)."""
    return [dataclasses.replace(c, result=tuple(r for r in c.result if r.shown is not False)) for c in calls]


def stratum(answer: AnswerRecord) -> str:
    """The null test's stratum: the question's operation, and whether every item is a literal."""
    kind = "literal" if all(is_literal(i, iid) for i, iid in zip(answer.items, answer.item_ids)) else "entity"
    return f"{question_type(answer.question) or 'membership'}|{kind}"


def null_test(answered, calls: dict, seed: int = 0) -> dict:
    """The chance rate: answers shuffled within each stratum (seeded), per stratum and overall."""
    rng = random.Random(seed)
    by = {a.qid: a for a in answered}
    groups: dict = {}
    for a in sorted(answered, key=lambda a: a.qid):
        groups.setdefault(stratum(a), []).append(a.qid)
    strata, results = {}, []
    for name in sorted(groups):
        order = groups[name]
        rng.shuffle(order)
        if len(order) < 2:
            strata[name] = {"n_questions": len(order), "support_rate": None, "found_rate": None}
            continue
        part = []
        for i, qid in enumerate(order):
            donor = by[order[(i + 1) % len(order)]]
            a = dataclasses.replace(by[qid], items=donor.items, item_ids=donor.item_ids, gold=(), gold_ids=())
            part.append(evaluate_question(a, calls.get(qid, [])))
        s = summarize(part)
        strata[name] = {"n_questions": len(order), "support_rate": s["support_rate"], "found_rate": s["found_rate"]}
        results += part
    s = summarize(results)
    return {"seed": seed, "overall": "question mean", "n_questions": len(results),
            "n_undefined": sum(v["n_questions"] for v in strata.values() if v["support_rate"] is None),
            "support_rate": s["support_rate"], "found_rate": s["found_rate"], "strata": strata}


def evaluate_set(answers: dict, calls: dict, qids, seed: int = 0) -> dict:
    """The support rate and its companions for one system on one set; every split question is counted once."""
    split = sorted(set(qids))
    counts = {"split": len(split), "answered": 0, "no_row": 0, "no_answer": 0, "error": 0, "no_records": 0,
              "no_rooted": 0, "answers_outside_split": len(set(answers) - set(split)),
              "records_outside_split": len(set(calls) - set(split))}
    answered = []
    for qid in split:
        a = answers.get(qid)
        if a is None:
            counts["no_row"] += 1
        elif a.error is not None:
            counts["error"] += 1
        elif not a.items:
            counts["no_answer"] += 1
        else:
            counts["answered"] += 1
            counts["no_records"] += qid not in calls
            counts["no_rooted"] += qid in calls and not rooted_records(calls[qid], a.topic)
            answered.append(a)
    results = [evaluate_question(a, calls.get(a.qid, [])) for a in answered]
    null = null_test(answered, calls, seed)
    return {"counts": counts, "answer_rate": counts["answered"] / len(split) if split else None,
            "no_rooted_share": counts["no_rooted"] / counts["answered"] if counts["answered"] else None,
            "headline": summarize(results), "chance_rate": null["support_rate"],
            "chance_rate_by_stratum": {k: v["support_rate"] for k, v in null["strata"].items()},
            "shown_only": summarize([evaluate_question(a, shown_only(calls.get(a.qid, []))) for a in answered]),
            "null_test": null, "questions": results}
