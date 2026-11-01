"""A run's cell: one row slot per question of its set, read only from the run's live shard files."""
from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass, field

from harness.metrics.reasoning import DuplicateRowError, live_files

@dataclass
class Cell:
    """One cell: its questions, each one's row (None: missing) with its source, shard and folder; counts."""
    questions: dict
    rows: dict = field(default_factory=dict)          # qid -> (row | None, source, shard | None, cell dir | None)
    prov: Counter = field(default_factory=Counter)
    found: bool = False                               # some designated source cell exists

    def take(self, qid: str, row, source: str, shard, cell_dir) -> None:
        if qid in self.rows:
            raise DuplicateRowError(f"a question has two rows after composition (sources {self.rows[qid][1]}, {source})")
        self.rows[qid] = (row, source, shard, cell_dir)


def read_cell(path: str) -> tuple:
    """({qid: (shard, row)} of a cell's live shards or None, the unparsed lines); a duplicate raises."""
    if not os.path.isdir(path):
        return None, 0
    out, unparsed = {}, 0
    for f in live_files(path):
        shard = int(os.path.basename(f)[1:-len(".jsonl")])
        for line in open(f, encoding="utf-8", errors="replace"):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except ValueError:
                unparsed += 1
                continue
            if not isinstance(r, dict):
                unparsed += 1
                continue
            qid = str(r.get("qid"))
            if qid in out:
                raise DuplicateRowError(f"a question has two live rows in {path}")
            out[qid] = (shard, r)
    return out, unparsed


def load_ids(path: str | None) -> set | None:
    """Ids from a JSON list, {"ids": [...]} or {"questions": [{"id"}]} file; None without a path."""
    if not path:
        return None
    d = json.load(open(path, encoding="utf-8"))
    if isinstance(d, dict):
        d = d.get("ids") or d.get("questions") or []
    return {str(x["id"]) if isinstance(x, dict) else str(x) for x in d}


def _count_unused(cell: Cell, source: str, rows: dict | None, used: set) -> None:
    for qid in rows or {}:
        if qid in used:
            continue
        cell.prov[f"not taken:{source}" if qid in cell.questions else f"outside the list:{source}"] += 1


def one_source(questions: dict, path: str, source: str) -> Cell:
    """A cell read whole from one source cell (Tables 2 and 3)."""
    cell = Cell(questions)
    rows, cell.prov["unparsed row lines"] = read_cell(path)
    cell.found = rows is not None
    for qid in questions:
        shard, row = (rows or {}).get(qid, (None, None))
        cell.take(qid, row, source, shard, path if row is not None else None)
        cell.prov[f"from:{source}" if row is not None else f"missing:{source}"] += 1
    _count_unused(cell, source, rows, set(questions))
    return cell
