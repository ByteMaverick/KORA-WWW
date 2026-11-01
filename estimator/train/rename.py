"""Renamed copies of training rows, so entity names carry no signal the estimator can lean on.

Renaming works on the rendered pair (question, evidence) the model reads: the entity names in the path lines' subject
and object fields and in the candidate list are replaced, consistently within a row, and their mentions in the
question (case-insensitive, longest first, on word boundaries). Relation names, ?x nodes, raw ids and literal values
(numbers, dates, comparisons, (largest)/(smallest)) are never touched. The label is unchanged: the structure is the
same. Three name sources: invented name-like names (used for training), random letter strings, and other real names
from the rows.

The released estimator's training rows, from the notebook's Data A and Data B (`estimator/data/rendered/{A,B}.parquet`):

    python -m estimator.train.rename --rows A.parquet B.parquet --out train.parquet [--fraction 1/3] [--seed-tag T]

then fine-tune on them with the notebook's training cell (`train_and_save`).
"""
from __future__ import annotations

import argparse
import json
import random
import re
import string
from fractions import Fraction

HEAD = "EVIDENCE (facts retrieved so far, as subject | relation | object)"
#: The per-row seed prefix of the training copies; this one reproduces the paper's renamed copies.
SEED_TAG = "j1h"

LITERAL = re.compile(
    r"^\s*(\(.*\)|(>=|<=|!=|>|<|=|in|not in)\s.*|[-+]?[\d][\d.,:/\-]*(\s+\S+)?|true|false|yes|no)\s*$", re.I)
RAW_ID = re.compile(r"^([mg]\.[0-9a-z_]+|[A-Za-z]+KG:\S+|KQAPro:\S+)$")


def is_name(s: str) -> bool:
    s = s.strip()
    return bool(s) and not s.startswith("?") and not LITERAL.match(s) and not RAW_ID.match(s) and len(s) >= 2


def parts(evidence: str):
    """(path lines, candidate count text, candidate names) of a rendered evidence block."""
    body, _, tail = evidence.partition("\n\nCANDIDATE ANSWER (")
    lines = body.split(HEAD + "\n", 1)[1].splitlines() if HEAD + "\n" in body else []
    count, _, cands = tail.partition(" items)\n")
    names = [] if cands.strip() in ("", "(empty)") else cands.split("; ")
    return lines, count, names


def names_in(evidence: str) -> list[str]:
    """The entity names a row shows: candidates, and every subject or object field that is a single name."""
    lines, _, cands = parts(evidence)
    out = [c for c in cands if is_name(c)]
    for line in lines:
        f = line.strip().split(" | ")
        if len(f) < 3:
            continue
        for field in (f[0], " | ".join(f[2:])):
            field = field.strip()
            if not is_name(field):
                continue
            if any(field.startswith(c) for c in cands) and ("(+" in field or ", " in field):
                continue                                          # the candidate list itself: renamed name by name
            out.append(field)
    return list(dict.fromkeys(out))


SYL = ["b", "d", "f", "g", "k", "l", "m", "n", "p", "r", "s", "t", "v", "z", "br", "dr", "kr", "st", "th", "tr", "gr",
       "fl", "sh", "ch", "vr", "sk", "pl", "j", "h", "w"]
VOW = ["a", "e", "i", "o", "u", "ae", "ei", "ou", "ia", "y"]
CODA = ["", "", "n", "r", "l", "s", "th", "k", "m", "x", "nd", "rk", "sh"]


def _word(rng):
    k = rng.choice([2, 2, 3])
    return "".join(rng.choice(SYL) + rng.choice(VOW) + (rng.choice(CODA) if j == k - 1 else "") for j in range(k)).capitalize()


def invented(orig, rng, pool=None):
    """A name-like invented name with the original's word count (1-3)."""
    return " ".join(_word(rng) for _ in range(max(1, min(3, len(orig.split())))))


def random_string(orig, rng, pool=None):
    """Random letters, word by word, the original's word lengths."""
    words = orig.split()[:3] or ["x"]
    return " ".join("".join(rng.choice(string.ascii_lowercase) for _ in range(max(3, len(w)))).capitalize() for w in words)


def other_real(orig, rng, pool):
    """A real name of another entity from the rows, the same word count when one exists."""
    k = min(3, len(orig.split()))
    cands = pool.get(k) or pool.get(1)
    for _ in range(20):
        x = rng.choice(cands)
        if x.lower() != orig.lower():
            return x
    return invented(orig, rng)


GENERATORS = {"invented": invented, "random": random_string, "real": other_real}


def _alternation(keys):
    keys = sorted(keys, key=len, reverse=True)
    return re.compile(r"(?<![0-9A-Za-z])(" + "|".join(re.escape(k) for k in keys) + r")(?![0-9A-Za-z])", re.I)


def rename_row(question: str, evidence: str, how: str, seed: str, pool=None):
    """(question, evidence, n names renamed, question mentions replaced) with every entity name replaced."""
    found = names_in(evidence)
    if not found:
        return question, evidence, 0, 0
    rng = random.Random(seed)
    gen = GENERATORS[how]
    new = {}
    for n in found:
        x = gen(n, rng, pool)
        while x in new.values():                                   # two entities never share a new name
            x = gen(n, rng, pool)
        new[n] = x
    lower = {k.lower(): v for k, v in new.items()}
    pat = _alternation(new)
    sub = lambda s: pat.sub(lambda m: lower[m.group(0).lower()], s)
    lines, count, cands = parts(evidence)
    out_lines = []
    for line in lines:
        f = line.strip().split(" | ")
        if len(f) < 3:
            out_lines.append(line)
            continue
        s, r, o = f[0], f[1], " | ".join(f[2:])
        out_lines.append(f"  {new.get(s, sub(s))} | {r} | {new.get(o, sub(o))}")
    body = HEAD + "\n" + "\n".join(out_lines)
    cand_text = "; ".join(new.get(c, c) for c in cands) if cands else "(empty)"
    ev = f"{body}\n\nCANDIDATE ANSWER ({count} items)\n{cand_text}"
    q_new, hits = pat.subn(lambda m: lower[m.group(0).lower()], question)
    return q_new, ev, len(new), hits


def name_pool(frames):
    """Real names by word count, from the given datasets' rows (for the `real` source)."""
    pool = {}
    for df in frames:
        for ev in df.evidence:
            for n in names_in(ev):
                pool.setdefault(min(3, len(n.split())), set()).add(n)
    return {k: sorted(v) for k, v in pool.items()}


def renamed(df, how: str, tag: str, pool=None):
    """The rows of df, every one renamed; labels and meta kept; counts of what was renamed."""
    import numpy as np
    rows = [rename_row(q, e, how, f"{tag}-{k}", pool) for k, (q, e) in enumerate(zip(df.question, df.evidence))]
    out = df.copy()
    out["question"] = [r[0] for r in rows]
    out["evidence"] = [r[1] for r in rows]
    stats = {"rows": len(df), "rows with a name renamed": sum(r[2] > 0 for r in rows),
             "names renamed per row (mean)": float(np.mean([r[2] for r in rows])) if rows else 0.0,
             "rows whose question had a mention replaced": sum(r[3] > 0 for r in rows)}
    return out, stats


def with_renamed_copies(rows, fraction: float, seed: int = 0, tag: str = SEED_TAG):
    """The rows plus invented-name copies of a random `fraction` of them; a text with both labels dropped, duplicates
    removed (the original kept). Returns (rows, stats)."""
    import pandas as pd
    picked = rows.sample(frac=fraction, random_state=seed)
    copies, stats = renamed(picked, "invented", tag)
    mix = pd.concat([rows, copies], ignore_index=True)
    both = mix.groupby(["question", "evidence"]).label.transform("nunique") > 1
    out = mix[~both].drop_duplicates(["question", "evidence"], keep="first").reset_index(drop=True)
    return out, {"copies": stats, "rows": len(out), "both labels dropped": int(both.sum())}


def read_rows(path: str):
    """A rows file (.parquet or .jsonl) as a DataFrame."""
    import pandas as pd
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_json(path, lines=True)


def ab_rows(a, b):
    """A and B together as the notebook mixes them: B first, a text in both kept once, a text with both labels dropped."""
    import pandas as pd
    cols = ["question", "evidence", "label", "meta"]
    mix = pd.concat([b[cols], a[cols]], ignore_index=True)
    both = mix.groupby(["question", "evidence"]).label.transform("nunique") > 1
    return mix[~both].drop_duplicates(["question", "evidence"], keep="first").reset_index(drop=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m estimator.train.rename")
    ap.add_argument("--rows", nargs=2, required=True, metavar=("A", "B"), help="Data A and Data B rows (.parquet/.jsonl)")
    ap.add_argument("--out", required=True, help="the training rows (.parquet or .jsonl)")
    ap.add_argument("--fraction", type=lambda s: float(Fraction(s)), default=1 / 3,
                    help="share of rows copied with invented names (default 1/3)")
    ap.add_argument("--seed-tag", default=SEED_TAG, help="per-row seed prefix of the copies")
    a = ap.parse_args(argv)
    ab = ab_rows(read_rows(a.rows[0]), read_rows(a.rows[1]))
    rows, stats = with_renamed_copies(ab, a.fraction, seed=0, tag=a.seed_tag)
    if a.out.endswith(".parquet"):
        rows.to_parquet(a.out, index=False)
    else:
        rows.to_json(a.out, orient="records", lines=True, force_ascii=False)
    print(json.dumps({"A+B rows": len(ab), **stats}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
