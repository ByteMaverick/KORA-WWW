"""The question sets, their questions as `QuestionSample`s, and the gold-locked sample a system is handed (README)."""
from __future__ import annotations

import gzip
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

STANDARD = ("webqsp", "cwq", "grailqa", "metaqa")
UNSEEN_SETS = ("webqsp", "cwq")
UNSEEN = tuple(f"{ds}-unseen{half}" for ds in UNSEEN_SETS for half in ("", "-real"))
DATASETS = STANDARD + UNSEEN
SPLITS = {**{d: ("test",) for d in STANDARD}, **{u: ("test", "validation") for u in UNSEEN}}
#: The suffix a set's benchmark name carries for each split, as the baselines read it (`cwq-test`).
SPLIT_SUFFIX = {"test": "test", "validation": "val"}


def graph_of(dataset: str) -> str:
    """The graph a set is asked on: MetaQA's own, or Freebase (the Unseen sets: Freebase with the added entities)."""
    return "metaqa" if dataset == "metaqa" else "freebase"


def check(dataset: str, split: str) -> None:
    if dataset not in SPLITS:
        raise ValueError(f"unknown dataset {dataset!r}: one of {', '.join(DATASETS)}")
    if split not in SPLITS[dataset]:
        raise ValueError(f"{dataset} has no {split!r} split: {', '.join(SPLITS[dataset])}")


def path_of(dataset: str, split: str, root: str | None = None) -> str:
    check(dataset, split)
    root = root or DATA
    if dataset in STANDARD:
        return os.path.join(root, "questions", dataset, f"{split}.json")
    return os.path.join(root, "unseen", dataset.split("-")[0], f"{split}.json")


def _read(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"no question file at {path}")
    with (gzip.open if path.endswith(".gz") else open)(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def pairs(dataset: str, split: str = "test", root: str | None = None) -> list:
    """[(twin record, source record)] of an Unseen set, in file order."""
    base = dataset.split("-")[0]
    data = _read(path_of(f"{base}-unseen", split, root))
    return [(p["twin"], p["source"]) for p in data["pairs"]]


def records(dataset: str, split: str = "test", root: str | None = None) -> list:
    """The question records of a set, in file order."""
    if dataset in STANDARD:
        data = _read(path_of(dataset, split, root))
        return list(data["questions"] if isinstance(data, dict) else data)
    half = 1 if dataset.endswith("-real") else 0
    return [p[half] for p in pairs(dataset, split, root)]


def questions(dataset: str, split: str = "test", root: str | None = None) -> dict:
    """{id: record} of a set."""
    return {str(q["id"]): q for q in records(dataset, split, root)}


@dataclass
class QuestionSample:
    """Single normalised KGQA question, independent of source benchmark."""

    qid:        str                         # unique identifier
    question:   str                         # natural language question
    answers:    List[str]                   # list of canonical answer strings
    answer_ids: List[str] = field(default_factory=list)  # entity IDs (when available)

    benchmark:  str = ""                    # the set, e.g. "cwq", "webqsp-unseen"
    kg:         str = ""                    # "freebase" or "metaqa"
    source_file: str = ""                   # path the sample came from

    # Optional extras (None: not available)
    sparql:       Optional[str] = None      # ground-truth SPARQL (if present)
    topic_entity: Optional[Dict] = None    # {mid: label} or {uri: label}
    question_type: Optional[str] = None    # a question category, when the set gives one
    template_id:  Optional[str] = None     # a question template, when the set gives one
    kg_source:    Optional[str] = None     # "metaqa" for MetaQA
    raw:          Optional[Dict] = None    # original record (for debugging)

    def __repr__(self) -> str:
        short_q = self.question[:60] + "…" if len(self.question) > 60 else self.question
        return (
            f"QuestionSample(qid={self.qid!r}, benchmark={self.benchmark!r}, "
            f"n_answers={len(self.answers)}, q={short_q!r})"
        )


def sample_of(rec: dict, dataset: str, source_file: str = "") -> QuestionSample:
    """One record as a sample; as the source's loader did, the gold ids are not carried."""
    answers = rec.get("answers", [])
    if isinstance(answers, str):
        answers = [answers]
    kg = graph_of(dataset)
    return QuestionSample(
        qid          = str(rec.get("id", "")),
        question     = rec.get("question", ""),
        answers      = [str(a) for a in answers],
        answer_ids   = [],
        benchmark    = dataset,
        kg           = kg,
        kg_source    = "metaqa" if kg == "metaqa" else None,
        topic_entity = rec.get("topic_entity") or None,
        source_file  = source_file,
        raw          = rec,
    )


def load(dataset: str, split: str = "test", max_samples: int | None = None,
         root: str | None = None) -> List[QuestionSample]:
    path = path_of(dataset, split, root)
    samples = [sample_of(r, dataset, path) for r in records(dataset, split, root)]
    return samples if max_samples is None else samples[:max_samples]


#: Every QuestionSample field that holds or derives from the gold (the raw record carries the gold query).
GOLD_FIELDS = frozenset({"answers", "answer_ids", "sparql", "raw"})


class GoldAccessError(RuntimeError):
    """A framework read the gold."""


class BlindSample:
    """A QuestionSample whose gold fields cannot be read or written; everything else passes through."""

    __slots__ = ("_sample",)

    def __init__(self, sample):
        object.__setattr__(self, "_sample", sample)

    def __getattr__(self, name):
        if name == "_sample":   # not set yet (a copy being built): never forward
            raise AttributeError(name)
        if name == "__dict__":   # vars(s) would hand over the real sample's fields, gold included
            raise GoldAccessError("a framework read the sample's __dict__, which holds the gold (harness/dataset.py)")
        if name in GOLD_FIELDS:
            raise GoldAccessError(f"a framework read the gold field {name!r} of question "
                                  f"{getattr(self._sample, 'qid', '?')} (harness/dataset.py)")
        return getattr(self._sample, name)

    def __setattr__(self, name, value):
        if name in GOLD_FIELDS:
            raise GoldAccessError(f"a framework wrote the gold field {name!r} (harness/dataset.py)")
        setattr(self._sample, name, value)

    def __copy__(self):
        import copy
        return BlindSample(copy.copy(self._sample))

    def __deepcopy__(self, memo):
        import copy
        return BlindSample(copy.deepcopy(self._sample, memo))

    def __repr__(self) -> str:
        return f"BlindSample({self._sample!r})"


def blind(sample):
    return None if sample is None else sample if isinstance(sample, BlindSample) else BlindSample(sample)


def is_gold_arm(framework: str, stop: str = "") -> bool:
    return framework.endswith("_gold") or stop == "gold"
