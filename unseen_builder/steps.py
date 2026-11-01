"""The steps after build: merge, heldout, pollution, trim, pairs and load (python -m unseen_builder.main STEP)."""
from __future__ import annotations

import argparse
import collections
import json
import os
import shutil
import sys
from pathlib import Path

from .checks import (CLONE_IRI, CLONE_KEYS, META, POLLUTED, SIZES, SPLITS, Unresolved, _json, _jsonl, _write_json,
                     excluded, failing, rule_out, run, source_labels, trim, write_meta, write_split)
from .graph import NEXT_ID, Freebase, bulk_load
from .names import normalise, openai_call
from .twin import _template


# step merge: the shards of one build joined
DUPLICATE = "duplicate name across shards"


def _files(shard: Path, split: str) -> Path:
    """The folder of a shard holding its split (the shard or its ``preview/``), with its build.json."""
    found = [d for d in (shard, shard / "preview") if (d / f"{split}.json").is_file()]
    if len(found) != 1:
        raise ValueError(f"{shard}: " + (f"a {split}.json in it and in its preview/: remove the one that is not this "
                                         "build's" if found else f"no {split}.json in it or in its preview/"))
    if not (found[0] / "build.json").is_file():
        raise ValueError(f"{found[0]}: no build.json: the shard's build did not finish")
    return found[0]


def _one_build(dirs: list[Path], builds: list[dict], set: str, split: str) -> None:
    """Refuse unless the shards are one build's: one set, split, offset, limit and N, every k once."""
    for d, b in zip(dirs, builds):
        if not b["args"].get("shard"):
            raise ValueError(f"{d}: not a shard (built without --shard)")
    runs = [(b["args"]["set"], b["args"]["split"], b["args"].get("offset", 0), b["args"]["limit"],
             b["args"]["shard"][1]) for b in builds]
    if any(run[:2] != (set, split) or run[2:] != runs[0][2:] for run in runs):
        raise ValueError(f"shards disagree on set, split, offset, limit or N (merging {set} {split}): "
                         + ", ".join(f"{d} {run}" for d, run in zip(dirs, runs)))
    n, ks = runs[0][4], sorted(b["args"]["shard"][0] for b in builds)
    if ks != list(range(n)):
        raise ValueError(f"shards k = {ks} do not cover 0..{n - 1} once each")


def _shared(groups: list[list[str]]) -> str | None:
    """The first id that two groups share, or None."""
    seen: set[str] = set()
    for group in groups:
        if hit := seen & set(group):
            return min(hit)
        seen |= set(group)
    return None


def _clashing(parts: list[dict], mappings: list[dict], ruled: dict[str, str]) -> list[dict]:
    """The twins of a later shard with a clone named as an earlier shard's kept twin's."""
    taken: set[str] = set()
    dropped: list[dict] = []
    for part, mapping in zip(parts, mappings):
        names = {c: normalise(name) for c, name in mapping["names"].items()}
        clash = {c for c, name in names.items() if name in taken}
        for t in part["questions"]:
            if t["id"] in ruled:
                continue
            if clash & set(t["clone_ids"]):
                dropped.append(t)
            else:
                taken.update(names[c] for c in t["clone_ids"] if c in names)
    return dropped


def _summed(counts: list[dict]) -> dict:
    out: dict = {}
    for c in counts:
        for key, value in c.items():
            if isinstance(value, dict):
                total = out.setdefault(key, {})
                for k, n in value.items():
                    total[k] = total.get(k, 0) + n
            elif key != "ids":
                out[key] = out.get(key, 0) + value
    out["ids"] = [min(c["ids"][0] for c in counts), max(c["ids"][1] for c in counts)]
    return out


def merge_main(argv=None, splits: Path = SPLITS) -> int:
    p = argparse.ArgumentParser(prog="python -m unseen_builder.main merge", description=(
        "Join the shards of one build into its split, as if one run had made them."))
    p.add_argument("--set", choices=["webqsp", "cwq"], required=True)
    p.add_argument("--split", choices=["test", "val", "train"], required=True)
    p.add_argument("--shards", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--replica-root", type=Path, help="write <root>/load/next_id.txt after the shards' clone numbers")
    args = p.parse_args(argv)
    # everything is read and computed first; nothing is written until every check has passed and every file is read
    dirs = [_files(shard, args.split) for shard in args.shards]
    builds = [_json(d / "build.json") for d in dirs]
    _one_build(dirs, builds, args.set, args.split)
    parts = [_json(d / f"{args.split}.json") for d in dirs]
    mappings = [_json(d / "mapping.json") for d in dirs]
    if twin := _shared([[t["id"] for t in part["questions"]] for part in parts]):
        raise ValueError(f"twin ids: {twin} is in two shards")
    if clone := _shared([[c for clones in m["sources"].values() for c in clones] for m in mappings]):
        raise ValueError(f"clone ids: {clone} is in two shards")

    ruled: dict[str, str] = {}               # a shard's own suite rules a twin out: build.excluded on its report
    reports = [_json(d / "report.json") if (d / "report.json").is_file() else None for d in dirs]
    for report, part in zip(reports, parts):
        if report:
            reasons = excluded(report)
            ruled.update({t["id"]: reasons[t["id"]] for t in part["questions"] if t["id"] in reasons})
    owner = {c: t["id"] for part in parts for t in part["questions"] for c in t["clone_ids"]}
    for report in filter(None, reports):     # another shard's clone changed this shard's source: its twin goes
        for clones in report["real_unchanged"]["summary"].get("foreign", {}).values():
            ruled.update({owner[c]: POLLUTED for c in clones if c in owner and owner[c] not in ruled})
    clashing = _clashing(parts, mappings, ruled)
    reasons = {**ruled, **{t["id"]: DUPLICATE for t in clashing}}
    dropped = [t for part in parts for t in part["questions"] if t["id"] in reasons]
    gone = {t["twin_id"]: reasons[t["id"]] for t in dropped}
    removed = {c for t in dropped for c in t["clone_ids"]}
    order = {q["id"]: i for i, q in enumerate(_json(splits / args.set / f"{args.split}.json")["questions"])}
    twins = sorted((t for part in parts for t in part["questions"] if t["twin_id"] not in gone),
                   key=lambda t: order[t["twin_id"]])
    meta = sorted(({**m, **dict.fromkeys(META), "dropped_reason": gone[m["twin_id"]]} if m["twin_id"] in gone else m
                   for d in dirs for m in _jsonl(d / f"{args.split}_meta.jsonl")), key=lambda m: order[m["twin_id"]])
    source = parts[0]["source"].split(", shard ")[0] + f", {len(dirs)} shards, seeds {[b['seed'] for b in builds]})"

    mapping: dict = {"sources": collections.defaultdict(list), **{key: {} for key in CLONE_KEYS}, "literals": []}
    drops: dict = {}
    for m, d in zip(mappings, dirs):
        for s, clones in m["sources"].items():
            if kept := [c for c in clones if c not in removed]:
                mapping["sources"][s] += kept
        for key in CLONE_KEYS:                           # an older shard has no "unpadded"
            mapping[key].update({c: v for c, v in m.get(key, {}).items() if c not in removed})
        mapping["literals"] += [literal for literal in m["literals"] if literal[0] not in removed]
        for reason, shard_drops in _json(d / "drops.json").items():
            entry = drops.setdefault(reason, {"count": 0, "ids": []})
            entry["count"] += shard_drops["count"]
            entry["ids"] += shard_drops["ids"]
    for source, reason in gone.items():
        entry = drops.setdefault(reason, {"count": 0, "ids": []})
        entry["count"] += 1
        entry["ids"].append(source)
    for entry in drops.values():
        entry["ids"].sort(key=order.get)                                     # the source's order, as the split
    lines = [line for d in dirs for line in (d / "clones.nt").read_text(encoding="utf-8").splitlines(keepends=True)
             if not removed & set(CLONE_IRI.findall(line))]
    counts = _summed([b["counts"] for b in builds])
    counts.update(built=len(twins), raw=len(twins), mod_entities=len({_template(t["gold"]["query"]) for t in twins}),
                  clones=counts["clones"] - len(removed), triples=len(lines),
                  dropped_duplicate_names={"count": len(clashing), "ids": [t["id"] for t in clashing]})
    for reason in gone.values():
        counts["dropped"][reason] = counts["dropped"].get(reason, 0) + 1
    if args.replica_root:                     # past every clone number a shard used, loaded or not, ruled out or not
        next_id = args.replica_root / "load" / NEXT_ID
        old = int(next_id.read_text(encoding="utf-8")) if next_id.is_file() else 1
        number = max([old, *(b["counts"]["ids"][1] + 1 for b in builds)])                   # never lowered

    args.out.mkdir(parents=True, exist_ok=True)
    shared = [] if builds[0]["args"].get("offset") else [splits / f"{args.set}-unseen"]   # a later wave: --out only
    for where in (args.out, *shared):
        write_split(where / f"{args.split}.json", args.set, args.split, source, twins)
        write_meta(where / f"{args.split}_meta.jsonl", meta)
    _write_json(args.out / "mapping.json", mapping)
    _write_json(args.out / "drops.json", drops)
    (args.out / "clones.nt").write_text("".join(lines), encoding="utf-8")
    _write_json(args.out / "build.json", {"shards": [str(s) for s in args.shards], "seeds": [b["seed"] for b in builds],
                                          "counts": counts, "seconds": round(sum(b["seconds"] for b in builds), 1)})
    if args.replica_root:
        next_id.parent.mkdir(parents=True, exist_ok=True)
        next_id.write_text(f"{number}\n", encoding="utf-8")
    print(f"[merge] {args.set} {args.split}: {len(dirs)} shards, {len(twins)} twins ({len(ruled)} ruled out by a "
          f"shard's suite or another shard's, {len(clashing)} dropped for a duplicate name), read {counts['read']}, "
          f"clone numbers {counts['ids'][0]}..{counts['ids'][1]}", flush=True)
    return 0


# step heldout (step 8) and step pollution: the release's second round
def heldout_main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m unseen_builder.main heldout", description=(
        "The held-out filter: a model outside the evaluated set gets every surviving twin twice, and a twin fails "
        "when the model answers the twin question closed-book or names a topic clone's real source from the clone's "
        "facts."))
    p.add_argument("--raw", type=Path, required=True, help="raw.jsonl: every validated twin")
    p.add_argument("--mapping", type=Path, nargs="+", required=True, help="the merged mapping.json files")
    p.add_argument("--ext", required=True, help="the extended graph with every twin's clones loaded")
    p.add_argument("--model", required=True)
    p.add_argument("--url", required=True)
    p.add_argument("--extra", type=json.loads, default=None)
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    twins = [json.loads(line) for line in args.raw.read_text(encoding="utf-8").splitlines() if line.strip()]
    labels = source_labels([json.loads(m.read_text(encoding="utf-8")) for m in args.mapping])
    call = openai_call(args.model, temperature=0.0, url=args.url, extra=args.extra)
    summary = run(twins, labels, Freebase(args.ext), call, args.out, args.threads)
    print("[heldout] " + ", ".join(f"{k} {v}" for k, v in summary.items()), flush=True)
    return 0


def pollution_main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m unseen_builder.main pollution", description=(
        "Rule out the twins whose clones changed another split's or another set's real questions, after the first "
        "round of checks."))
    p.add_argument("--checked", type=Path, help="holds <set>/<split>/ from round 2's suites")
    p.add_argument("--out", type=Path, help="never --checked or inside it")
    p.add_argument("--gate", type=Path, help="round 3: exit 1 unless every report under it passes checks 1 and 2")
    args = p.parse_args(argv)
    if args.gate:
        reports, failed = sorted(args.gate.glob("*/*/report.json")), failing(args.gate)
        for where, checks in failed.items():
            print(f"[gate] {where}: fails {', '.join(checks)}", flush=True)
        print(f"[gate] {len(reports)} reports, {len(failed)} failing checks 1 or 2", flush=True)
        return 1 if failed or not reports else 0
    if not (args.checked and args.out):
        p.error("--checked and --out, or --gate")
    if args.out.resolve().is_relative_to(args.checked.resolve()):
        p.error("--out is inside --checked: the step never writes its inputs")
    found = rule_out(args.checked)
    for set_ in found["sets"]:
        ids = sorted({t for where, twins in found["ruled_out"].items() if where.split("/")[0] == set_ for t in twins})
        (args.out / set_).mkdir(parents=True, exist_ok=True)
        (args.out / set_ / "polluted.txt").write_text("".join(i + "\n" for i in ids), encoding="utf-8")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "pollution.json").write_text(json.dumps(found, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    for where, twins in sorted(found["ruled_out"].items()):
        print(f"[pollution] {where}: {len(twins)} twins ruled out (cross-split)", flush=True)
    unowned = sum(len(c) for sources in found["unowned"].values() for c in sources.values())
    unattributed = sum(map(len, found["unattributed"].values()))
    print(f"[pollution] ruled out {sum(map(len, found['ruled_out'].values()))} twins in all; {unowned} foreign "
          f"clones owned by no checked twin; {unattributed} unattributed sources (trim refuses them)", flush=True)
    return 0


# step trim
def trim_main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m unseen_builder.main trim", description=(
        "Cut a set's merged, validated twins to the release; every validated twin and every exclusion stays in "
        "raw.jsonl."))
    p.add_argument("--set", choices=["webqsp", "cwq"], required=True)
    p.add_argument("--merged", type=Path, required=True,
                   help="holds <split>/ as the merge step wrote it, per merged split")
    p.add_argument("--out", type=Path, required=True, help="the release; never one of the inputs")
    p.add_argument("--splits", type=Path, default=SPLITS,
                   help="the split files' root: <set>/test.json (the source) and <set>-unseen/ (the merged twins)")
    p.add_argument("--exclude", type=Path, default=None,
                   help="twin ids to leave out (one a line), e.g. heldout_excluded.txt: recorded, never released")
    p.add_argument("--polluted", type=Path, default=None,
                   help="the pollution step's twin ids (one a line): ruled out as cross-split polluters, recorded")
    p.add_argument("--val-from", type=Path, default=None,
                   help="the val wave's directory (a build of the train split with --offset, merged and checked): "
                        "val is its first --val twins; without it val is the train twins after the released train")
    for split, n in SIZES.items():
        p.add_argument(f"--{split}", type=int, default=n)
    args = p.parse_args(argv)
    exclude = set(args.exclude.read_text(encoding="utf-8").split()) if args.exclude else set()
    polluted = set(args.polluted.read_text(encoding="utf-8").split()) if args.polluted else None
    sizes = {s: getattr(args, s) for s in SIZES}
    try:
        counts = trim(args.set, args.merged, args.out, sizes, splits=args.splits, exclude=exclude,
                      val_from=args.val_from, polluted=polluted)
    except Unresolved as e:
        print(f"[trim] {args.set}: refused: {e}", file=sys.stderr, flush=True)
        return 2
    summary = _json(args.out / "trim.json")
    short = summary["test_short"]
    if summary["val_short"] > 0:
        print(f"[trim] {args.set}: warning: val has {counts['val']} twins of the {sizes['val']} asked "
              f"({summary['val_source']})", file=sys.stderr, flush=True)
    print(f"[trim] {args.set}: " + ", ".join(f"{k} {v}" for k, v in counts.items())
          + f"; test strata short {short['strata']} ({short['twins'] - short['unplaced']} twins moved to other "
          f"strata, {short['unplaced']} unplaced)", flush=True)
    return 0


# step pairs: the pair files the harness reads
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "unseen"
SETS = ("webqsp", "cwq")
#: The sources a twin of each split may have (``provenance.source_split``): a test twin's comes from the source test
#: split, a val twin's from train or validation, never from a test question.
SOURCES = {"test": {"test"}, "val": {"train", "val"}}
FILE = {"test": "test.json", "val": "validation.json"}


class Refused(Exception):
    """The pair cannot be written; the message names the ids."""


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_ids(path: Path) -> list[str]:
    """A JSON list of ids, or one id per line."""
    text = Path(path).read_text(encoding="utf-8")
    try:
        ids = json.loads(text)
    except json.JSONDecodeError:
        return [line.strip() for line in text.splitlines() if line.strip()]
    if not isinstance(ids, list):
        raise Refused(f"{path}: not a JSON list of ids")
    return [str(i) for i in ids]


def _named(ids) -> str:
    ids = list(ids)
    return f"{len(ids)}: {', '.join(ids)}"


def kept(twins: list[dict], ids: list[str] | None) -> list[dict]:
    """The twins ``ids`` names, in that order (a twin's id or its source's); all of them when there is no list."""
    seen: dict[str, int] = {}
    for t in twins:
        seen[t["twin_id"]] = seen.get(t["twin_id"], 0) + 1
    if twice := [i for i, n in seen.items() if n > 1]:
        raise Refused(f"twins of one source more than once, {_named(twice)}")
    if ids is None:
        return twins
    by = {key: t for t in twins for key in (t["id"], t["twin_id"])}
    if missing := [i for i in ids if i not in by]:
        raise Refused(f"listed ids that are no twin's, {_named(missing)}")
    out = [by[i] for i in ids]
    if len({t["twin_id"] for t in out}) < len(out):
        raise Refused("a twin is listed twice (by its id and its source's, or twice)")
    return out


class Sources:
    """A set's source records: the split files, read once."""

    def __init__(self, splits: Path, name: str):
        self.splits, self.name = splits, name
        self.files: dict[str, dict[str, dict]] = {}

    def split(self, split: str) -> dict[str, dict]:
        if split not in self.files:
            path = self.splits / self.name / f"{split}.json"
            self.files[split] = {str(q["id"]): q for q in read_json(path)["questions"]} if path.is_file() else {}
        return self.files[split]

    def find(self, twin: dict) -> dict | None:
        source = (twin.get("provenance") or {}).get("source_split") or ""
        return self.split(source).get(twin["twin_id"]) if source else None


def mismatch(twin: dict, real: dict) -> str | None:
    """How a twin differs from its source where the generator never changes it, or None."""
    tg, rg = twin.get("gold") or {}, real.get("gold") or {}     # the clones' ids and names only
    if tg.get("shape") != rg.get("shape"):
        return "gold shape"
    hops = lambda g: [(h.get("relation"), h.get("direction")) for h in g.get("hops") or []]
    return "gold hops" if hops(tg) != hops(rg) else None


def pair(name: str, split: str, twins_path: Path, ids: list[str] | None, splits: Path) -> dict:
    """The pair file's contents, every check passed."""
    path = twins_path / f"{split}.json" if twins_path.is_dir() else twins_path
    doc = read_json(path)
    if doc.get("dataset") != f"{name}-unseen":
        raise Refused(f"{path} holds {doc.get('dataset')!r} twins, not {name}-unseen")
    twins = kept(doc["questions"], ids)
    sources = Sources(splits, name)
    wrong_split = [t["twin_id"] for t in twins
                   if (t.get("provenance") or {}).get("source_split") not in SOURCES[split]]
    if wrong_split:
        raise Refused(f"{split} twins whose source split (provenance.source_split) is not one of "
                      f"{sorted(SOURCES[split])}, {_named(wrong_split)}")
    reals, missing, unlike = [], [], []
    for t in twins:
        real = sources.find(t)
        if real is None:
            missing.append(t["twin_id"])
        elif why := mismatch(t, real):
            unlike.append(f"{t['twin_id']} ({why})")
        reals.append(real)
    if missing:
        raise Refused(f"twins whose source is not found, {_named(missing)}")
    if unlike:
        raise Refused(f"twins that do not match their source, {_named(unlike)}")
    return {"pairs": [{"source": real, "twin": t} for real, t in zip(reals, twins)]}


def _dump(value) -> str:
    return json.dumps(value, indent=1, ensure_ascii=False) + "\n"


def write_pair(out: Path, name: str, split: str, doc: dict) -> Path:
    """The pair file, written beside itself first and then renamed."""
    path = out / name / FILE[split]
    temp = path.with_name(path.name + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        temp.write_text(_dump(doc), encoding="utf-8")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)
    return path


def _pairs_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m unseen_builder.main pairs", description=(
        "The Unseen benchmark as the harness's pair file: a set's twins beside their real sources."))
    p.add_argument("--set", required=True, choices=SETS)
    p.add_argument("--split", required=True, choices=sorted(SOURCES))
    p.add_argument("--twins", type=Path, required=True, help="a folder holding <split>.json, or the twin file")
    p.add_argument("--ids", type=Path, help="keep only these twins, in this order (a JSON list or one id per line)")
    p.add_argument("--splits", type=Path, default=SPLITS, help="the split files (the sources)")
    p.add_argument("--out", type=Path, default=OUT, help="where the pair file goes (default: data/unseen)")
    return p


def pairs_main(argv=None) -> int:
    a = _pairs_parser().parse_args(argv)
    try:
        doc = pair(a.set, a.split, a.twins, read_ids(a.ids) if a.ids else None, a.splits)
    except Refused as e:
        print(f"[pairs] refused, nothing written: {e}", file=sys.stderr)
        return 1
    path = write_pair(a.out, a.set, a.split, doc)
    print(f"[pairs] {a.set} {a.split}: {len(doc['pairs'])} pairs -> {path}")
    return 0


# step load: the released clones.nt files loaded into a fresh copy
def release(root: Path, host: str, port: int, files: list[Path], loader=bulk_load) -> list[Path]:
    """Copy each file into ``<root>/load/`` and bulk-load it; the loaded paths."""
    folder = Path(root) / "load"
    folder.mkdir(parents=True, exist_ok=True)
    loaded = []
    for nt in files:
        nt = Path(nt)
        target = folder / f"release-{nt.parent.parent.name}-{nt.parent.name}-{nt.name}"   # <set>/<split>/clones.nt
        shutil.copyfile(nt, target)
        loader(Path(root), host, port, target)
        loaded.append(target)
    return loaded


def load_main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m unseen_builder.main load", description=(
        "Rebuild the extended graph from its release files: load every merged ``clones.nt`` into a fresh Freebase "
        "replica."))
    p.add_argument("--replica-root", type=Path, required=True)
    p.add_argument("--isql-host", required=True)
    p.add_argument("--isql-port", type=int, required=True)
    p.add_argument("files", nargs="+", type=Path)
    args = p.parse_args(argv)
    for path in release(args.replica_root, args.isql_host, args.isql_port, args.files):
        print(f"[load] loaded {path}", flush=True)
    return 0
