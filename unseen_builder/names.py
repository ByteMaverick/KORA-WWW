"""Step 5, names for the clones: real name parts recombined for people and places, a model for every other kind."""
from __future__ import annotations

import collections
import fcntl
import itertools
import json
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator

import requests

from .graph import Freebase
from .prompts import NAME as PROMPT, NAME_PHRASE as PHRASE_PROMPT, NAME_SERIES as SERIES_PROMPT
from .twin import _DATE, CLASS_RELATIONS, Clone, Triple

K = 5                          # candidates per batch: model names per call, recombination draws per redraw
ATTEMPTS = 5                   # batches per clone before NamingFailed
MAX_ATTRIBUTES = 6             # attribute lines in a prompt
OFFSET_SPAN = 100_000          # build_pools reads its labels from a seeded offset below this
BACKOFF = (2, 4, 8)            # seconds before each retry of a failed naming-model request
RECOMBINED = ("person", "place")
POOL_FILES = {"first": "first_names.txt", "last": "last_names.txt", "place": "place_parts.txt"}
STEM_LETTERS = 5               # a name's stem: its first letters, a leading year or ordinal dropped
STEM_USES = 3                  # names of one kind a build hands out on one stem
RECENT = 5                     # names of the kind the prompt is told to avoid: the last handed out
FORM_SAMPLE = 100              # labels read per type to find its form
FORM_SHARE = 0.3               # share of the sampled labels in one dated form that names the kind by that form
                               # (safe with the per-label gate; time.event sits near 0.53)
PHRASE_KINDS = ("symptom", "medical sign", "clinical sign", "finding", "side effect", "adverse effect",
                "adverse event", "phenotype")    # a kind ending in one of these is named by a clinical phrase
PLACEHOLDERS = ["John Doe", "Jane Doe", "John Smith", "Acme", "Foo", "Bar", "Lorem Ipsum", "Example", "Test",
                "Unknown", "N/A"]
ARTICLES = {"the", "a", "an"}
KIND_OF_TYPE = {"people.person": "person", "location.location": "place", "location.citytown": "place",
                "location.country": "place", "location.administrative_division": "place", "film.film": "film",
                "tv.tv_program": "TV program", "music.album": "album", "book.book": "book",
                "organization.organization": "organization", "business.employer": "organization",
                "sports.sports_team": "organization", "education.educational_institution": "school"}
PLACEMENT = {"year": "after its year", "ordinal": "after its edition number", "year_last": "before its year"}
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
_PUNCT = re.compile(r"[^\w\s]|_")
_BULLET = re.compile(r"^\s*(?:\d+\s*[.)]|[-*•])\s*")
_QUOTES = "\"'“”‘’`*"
_LETTER = re.compile(r"[^\W\d_]")
_SUFFIX = re.compile(r"Jr|Sr|[IVX]+")
_YEAR_FIRST = re.compile(r"(\d{4})(?=[\s,:/–-])")
_ORDINAL_FIRST = re.compile(r"(\d+)(?:st|nd|rd|th)\b", re.I)
_YEAR_LAST = re.compile(r"(,?\s+)(\d{4})$")
_VALUE = re.compile(r"^\s*(?:\d{4}(?:[–-]\d{2,4})?|\d+(?:st|nd|rd|th))\b[\s,:–-]*|,?\s+\d{4}\s*$", re.I)   # a value
_PHRASE = re.compile(rf"(?:^|\s)(?:{'|'.join(map(re.escape, PHRASE_KINDS))})s?$", re.I)


@dataclass
class Pools:
    first: list[str]
    last: list[str]
    place: list[str]


@dataclass(frozen=True)
class Form:
    """How a kind's labels carry a value: ``year``, ``ordinal`` or ``year_last``."""
    form: str
    values: tuple[int, ...]       # the years or ordinals the sampled labels carry, sorted
    sep: str = " "


class NamingFailed(Exception):
    """No candidate for the clone survived ``ATTEMPTS`` batches."""

    def __init__(self, clone_id: str):
        super().__init__(clone_id)
        self.clone_id = clone_id


def normalise(name: str) -> str:
    """Case-folded, punctuation as spaces, a leading article dropped, single spaces: what two names compare on."""
    words = _PUNCT.sub(" ", name.casefold()).split()
    if len(words) > 1 and words[0] in ARTICLES:
        words = words[1:]
    return " ".join(words)


def stem(name: str) -> str:
    """The first ``STEM_LETTERS`` letters of the normalised name, a leading year or ordinal dropped."""
    return "".join(ch for ch in _VALUE.sub("", normalise(name), count=1) if ch.isalpha())[:STEM_LETTERS]


class LabelIndex:
    """What a new name must not be: a placeholder, a name in names, a real label or alias, or taken."""

    def __init__(self, collides_remote: Callable[[str], bool], names: Iterable[str] = (),
                 shared: Path | None = None):
        self.remote = collides_remote                 # exact English type.object.name / common.topic.alias
        self.placeholders = {normalise(n) for n in PLACEHOLDERS}
        self.names = {normalise(n) for n in names}
        self.taken: set[str] = set()                  # normalised names assign has handed out
        self.stems: collections.Counter = collections.Counter()   # (kind, stem) -> names handed out on it, here
        self.recent: dict[str, list[str]] = collections.defaultdict(list)   # kind -> its names, in hand-out order
        self.shared = shared                          # one line per name handed out by any concurrent build
        self._seen = None
        self.refresh()

    def refresh(self) -> None:
        """Pull the names other builds appended to ``shared`` since the last look (a stat when nothing changed)."""
        if self.shared is None or not self.shared.is_file():
            return
        stat = self.shared.stat()
        if (stat.st_mtime_ns, stat.st_size) != self._seen:
            self._seen = (stat.st_mtime_ns, stat.st_size)
            self.taken.update(line for line in self.shared.read_text(encoding="utf-8").split("\n") if line)

    def take(self, name: str, kind: str | None = None) -> None:
        """Hand the name out: remembered, appended to ``shared``, its stem counted for ``kind``."""
        key = normalise(name)
        self.taken.add(key)
        if kind is not None:
            self.stems[kind, stem(name)] += 1
            self.recent[kind].append(name)
        if self.shared is not None:
            with self.shared.open("a", encoding="utf-8") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                f.write(key + "\n")

    def reason(self, name: str, kind: str | None = None) -> str | None:
        """``placeholder``, ``repeated stem``, ``collision``, or None."""
        key = normalise(name)
        if key in self.placeholders:
            return "placeholder"
        if kind is not None and (s := stem(name)) and self.stems[kind, s] >= STEM_USES:
            return "repeated stem"
        return "collision" if key in self.names or self.remote(name) else None

    def collides(self, name: str) -> bool:
        return self.reason(name) is not None


def _part(token: str) -> bool:
    """A usable name part: ASCII letters only, two or more, capitalised, not a suffix."""
    return (token.isascii() and token.isalpha() and len(token) >= 2 and token[0].isupper()
            and not _SUFFIX.fullmatch(token))


def build_pools(fb: Freebase, out_dir: Path, rng: random.Random, n: int = 5000) -> None:
    """The three name pools from ``n`` English labels per type, each file written whole."""
    def labels(type: str) -> list[str]:
        q = (f"SELECT ?n WHERE {{ ?s ns:type.object.type ns:{type} . ?s ns:type.object.name ?n "
             f'FILTER(lang(?n) = "en") }} LIMIT {n} OFFSET {rng.randrange(OFFSET_SPAN)}')
        return [row["?n"] for row in fb.select(q)]
    people = [words for words in map(str.split, labels("people.person")) if len(words) > 1]
    tokens = {"first": {w[0] for w in people}, "last": {w[-1] for w in people},
              "place": {t for label in labels("location.location") for t in label.split()}}
    out_dir.mkdir(parents=True, exist_ok=True)
    for pool, file in POOL_FILES.items():
        tmp = out_dir / f"{file}.{os.getpid()}.tmp"                  # one per writer: none writes into another's
        tmp.write_text("".join(t + "\n" for t in sorted(filter(_part, tokens[pool]))), encoding="utf-8")
        os.replace(tmp, out_dir / file)


def load_pools(dir: Path) -> Pools:
    return Pools(**{pool: (Path(dir) / file).read_text(encoding="utf-8").split() for pool, file in POOL_FILES.items()})


def recombine(kind: str, attrs: dict, rng: random.Random, pools: Pools) -> str:
    """``<First> <Last>`` for a person, one or two distinct place parts for a place; any other kind is the model's."""
    if kind == "person":
        parts = [rng.choice(pools.first), rng.choice(pools.last)]
    elif kind == "place":
        parts = rng.sample(pools.place, rng.randint(1, 2) if len(pools.place) > 1 else 1)
    else:
        raise ValueError(f"no recombination for {kind!r}")
    return " ".join(p[:1].upper() + p[1:] for p in parts)


def type_labels(fb: Freebase, type: str, n: int = FORM_SAMPLE) -> list[str]:
    """Up to ``n`` English labels of the type's nodes: what ``label_form`` reads."""
    q = (f"SELECT ?n WHERE {{ ?s ns:type.object.type ns:{type} . ?s ns:type.object.name ?n "
         f'FILTER(lang(?n) = "en") }} LIMIT {n}')
    return [row["?n"] for row in fb.select(q)]


def label_form(labels: Iterable[str]) -> Form | None:
    """The dated form at least ``FORM_SHARE`` of the labels have, else None."""
    labels = [label.strip() for label in labels if label and label.strip()]
    found: dict[str, list[int]] = {"year": [], "ordinal": [], "year_last": []}
    seps = collections.Counter()
    for label in labels:
        if m := _YEAR_FIRST.match(label):
            found["year"].append(int(m[1]))
        elif m := _ORDINAL_FIRST.match(label):
            found["ordinal"].append(int(m[1]))
        elif m := _YEAR_LAST.search(label):
            found["year_last"].append(int(m[2]))
            seps[m[1]] += 1
    form, values = max(found.items(), key=lambda item: len(item[1]))
    if not labels or len(values) < FORM_SHARE * len(labels):
        return None
    return Form(form, tuple(sorted(values)), seps.most_common(1)[0][0] if form == "year_last" else " ")


def phrase_kind(kind: str) -> bool:
    """A symptom-like kind (its last words one of ``PHRASE_KINDS``): named by a clinical phrase."""
    return bool(_PHRASE.search(kind.strip()))


def _ordinal(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def _form_value(form: Form, dates: Iterable[str], rng: random.Random) -> tuple[int, str]:
    """The clone's value for a dated name, and whether it came from its dates or was drawn."""
    years = [int(m[1]) for value in dates if (m := _DATE.fullmatch(str(value).strip()))]
    if form.form != "ordinal" and years:
        return min(years), "date"
    return rng.randint(form.values[0], form.values[-1]), "drawn"


def _in_form(label: str | None, form: Form) -> bool:
    """The label carries the form's value where the form puts it."""
    if not label:
        return False
    if form.form == "year_last":
        return bool(_YEAR_LAST.search(label.strip()))
    return bool((_ORDINAL_FIRST if form.form == "ordinal" else _YEAR_FIRST).match(label.strip()))


def _stamp(form: Form, value: int, series: str) -> str:
    """The name in the kind's form: "2009 Veltran Series", "63rd Halden Awards", "Orsk general election, 2008"."""
    if form.form == "year_last":
        return f"{series}{form.sep}{value}"
    return f"{_ordinal(value) if form.form == 'ordinal' else value} {series}"


def _restamp(label: str, form: Form, value: int) -> str:
    """A derived label with the source's value replaced by the clone's."""
    if form.form == "year_last":
        return _YEAR_LAST.sub(lambda m: f"{m[1]}{value}", label, count=1)
    pattern, new = (_ORDINAL_FIRST, _ordinal(value)) if form.form == "ordinal" else (_YEAR_FIRST, str(value))
    return pattern.sub(new, label, count=1) if pattern.match(label) else label


def _series(name: str) -> str:
    """A generated series name with any year or ordinal the model added anyway taken off."""
    return _VALUE.sub("", _VALUE.sub("", name, count=1), count=1).strip(" ,:–-")


def kind_of(type: str) -> str:
    """The naming kind of a Freebase type: ``KIND_OF_TYPE``, else the type's last segment in words."""
    return KIND_OF_TYPE.get(type) or type.rsplit(".", 1)[-1].replace("_", " ")


def attrs_for(clone_id: str, kept_edges: list[Triple], labels: dict[str, str]) -> dict[str, str]:
    """``{relation: target label}`` over the clone's out-edges to labelled kept nodes."""
    out: dict[str, list[str]] = {}
    for s, r, o in kept_edges:
        key = r.rsplit(".", 1)[-1].replace("_", " ")
        if s == clone_id and r not in CLASS_RELATIONS and o in labels and (key in out or len(out) < MAX_ATTRIBUTES):
            if labels[o] not in out.setdefault(key, []):
                out[key].append(labels[o])
    return {key: ", ".join(values) for key, values in out.items()}


def llm_name(kind: str, attrs: dict, k: int, call: Callable[[str], str], avoid: Iterable[str] = (),
             template: str = PROMPT, **fields: str) -> list[str]:
    """Up to ``k`` names the model proposes for a ``kind`` with these attributes, cleaned."""
    lines = [f"{a}: {v}" for a, v in list(attrs.items())[:MAX_ATTRIBUTES]] or ["(none)"]
    prompt = template.format(
        kind=kind, attributes="\n".join(lines), k=k, avoid=", ".join(avoid) or "(none)", **fields)
    names = (_BULLET.sub("", line).strip().strip(_QUOTES).strip() for line in (call(prompt) or "").splitlines())
    return [name for name in names if _LETTER.search(name) and not name.endswith(":")][:k]


def fake_name(n: int) -> str:
    """A stand-in for a model's name in a dry run or a test, its stem unique."""
    letters, rest = [], n
    for i in range(6):                               # n in the radices 14, 5, 14, ...: its first five letters fix it
        rest, digit = divmod(rest, 5 if i % 2 else 14)
        letters.append(("aeiou" if i % 2 else "bdfgklmnprstvz")[digit])
    return "".join(letters).capitalize() + f" {n}"


def openai_call(model: str = "gpt-4.1-mini", temperature: float = 0.8,
                sleep: Callable[[float], None] = time.sleep, url: str = OPENAI_URL,
                extra: dict | None = None) -> Callable[[str], str]:
    """The naming model: a chat completion per prompt at an OpenAI-compatible ``url``, retried on 429 / 5xx."""
    key = os.environ.get("OPENAI_API_KEY")
    headers = {"Authorization": f"Bearer {key}"} if key else {}

    def call(prompt: str) -> str:
        body = {"model": model, "temperature": temperature, "max_tokens": 120,
                "messages": [{"role": "user", "content": prompt}], **(extra or {})}
        for wait in (*BACKOFF, None):
            try:
                reply = requests.post(url, headers=headers, json=body, timeout=60)
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
    return call


def _words(label: str) -> re.Pattern:
    return re.compile(rf"(?<!\w){re.escape(label)}(?!\w)", re.I)


def _derived(cid: str, names: dict[str, str], labels: dict[str, str], linked: set[frozenset]) -> list[str]:
    """The clone's source label with each named parent's old label replaced by its new name."""
    label = labels.get(cid)
    parents = [p for p in names if label and p in labels and labels[p].casefold() != label.casefold()
               and frozenset((cid, p)) in linked and _words(labels[p]).search(label)]
    for p in sorted(parents, key=lambda p: -len(labels[p])):
        label = _words(labels[p]).sub(lambda _, new=names[p]: new, label, count=1)
    return [label] if parents else []


def _generated(kind: str, attrs: dict, rng: random.Random, pools: Pools, call: Callable[[str], str],
               avoid: list[str] = (), form: Form | None = None,
               value: int | None = None) -> Iterator[tuple[str, str | None, bool]]:
    """``ATTEMPTS`` batches of ``K`` candidates, drawn as needed: recombined, or the model's."""
    for _ in range(ATTEMPTS):
        if kind in RECOMBINED:
            yield from ((recombine(kind, attrs, rng, pools), None, False) for _ in range(K))
        elif form is not None:
            for name in llm_name(kind, attrs, K, call, avoid, SERIES_PROMPT, placement=PLACEMENT[form.form],
                                 example=_stamp(form, value, "<name>")):
                series = _series(name)
                yield (_stamp(form, value, series) if series else name), series, True
        else:
            yield from ((name, None, True) for name in llm_name(kind, attrs, K, call, avoid,
                                                                PHRASE_PROMPT if phrase_kind(kind) else PROMPT))


def _rejection(index: LabelIndex, name: str, series: str | None, kind: str | None) -> str | None:
    """Why the candidate is refused, or None when it is free."""
    if series == "":
        return "no series"
    if normalise(name) in index.taken:
        return "duplicate"
    return index.reason(name, kind) or (index.reason(series) if series else None)


def assign(clone: Clone, kinds: dict[str, str], attrs: dict[str, dict], index: LabelIndex,
           call: Callable[[str], str], rng: random.Random, log: Path, pools: Pools,
           source_labels: dict[str, str], dates: dict[str, list[str]] | None = None,
           samples: dict[str, list[str]] | None = None) -> dict[str, str]:
    """A new name for every clone in ``kinds``; rejections counted in ``assign.stats``."""
    linked = {frozenset((s, o)) for s, _, o in clone.edges}
    sources = [_words(label) for label in source_labels.values() if label]
    names: dict[str, str] = {}
    shared: dict[str, str] = {}                       # a dated source label without its value -> its series' new name
    with Path(log).open("a", encoding="utf-8") as f:
        for cid in sorted(kinds, key=lambda c: len(source_labels.get(c) or "")):
            kind = kinds[cid]
            safe = {a: v for a, v in attrs.get(cid, {}).items() if not any(s.search(f"{a}\n{v}") for s in sources)}
            entry = {"clone_id": cid, "kind": kind, "attrs": safe, "candidates": [], "rejected": [],
                     "chosen": None, "calls": []}

            def logged(prompt: str, entry: dict = entry) -> str:
                reply = call(prompt)
                entry["calls"].append({"prompt": prompt, "reply": reply})
                return reply
            form = label_form((samples or {}).get(cid) or ()) if kind not in RECOMBINED else None
            form = form if form is not None and _in_form(source_labels.get(cid), form) else None
            value = key = None
            first = [(name, None, False) for name in _derived(cid, names, source_labels, linked)]
            if form is not None:
                value, origin = _form_value(form, (dates or {}).get(cid) or (), rng)
                entry["form"] = {"form": form.form, "value": value, "from": origin}
                key = normalise(_VALUE.sub("", source_labels.get(cid) or "", count=1)) or None
                first = ([(_stamp(form, value, shared[key]), None, False)] if key in shared else []) + [
                    (_restamp(name, form, value), None, False) for name, _, _ in first]
            elif phrase_kind(kind):
                entry["style"] = "phrase"
            index.refresh()
            candidates = _generated(kind, safe, rng, pools, logged, index.recent[kind][-RECENT:], form, value)
            try:
                for name, series, checked in itertools.chain(first, candidates):
                    entry["candidates"].append(name)
                    reason = _rejection(index, name, series, kind if checked else None)
                    if reason is None:
                        entry["chosen"] = names[cid] = name
                        index.take(name, kind)
                        if key and series:
                            shared.setdefault(key, series)
                        break
                    entry["rejected"].append([name, reason])
                    assign.stats[reason] += 1
            finally:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            if entry["chosen"] is None:
                raise NamingFailed(cid)
    return names


assign.stats = collections.Counter()
