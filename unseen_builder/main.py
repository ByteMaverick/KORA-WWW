"""The Unseen builder, one step at a time: python -m unseen_builder.main STEP [flags] (see README.md)."""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
import time
import traceback
import zlib
from pathlib import Path

from .checks import (META, SPLITS, _check, _counts, _exclude, _execute, _held, _load, _load_and_check, _print_checks,
                     _saved, _write_json, _written, chat_judge, excluded, write_meta, write_split)
from .graph import (MAX_PATH, NEXT_ID, PATH_TOO_LARGE, Freebase, LiteralRange, RowCap, _cached, _capped_degree,
                    _datatype, benchmark_nodes, collides_remote, ntriples)
from .names import (OPENAI_URL, POOL_FILES, LabelIndex, NamingFailed, assign, attrs_for, build_pools, kind_of,
                    load_pools, openai_call, type_labels)
from .steps import heldout_main, load_main, merge_main, pairs_main, pollution_main, trim_main
from .twin import (_FULL_DATE, CLOSED_CLASS_FREEBASE, LITERAL_UNMAPPED, Clone, Gold, Rejected, _path_size, _topics,
                   clone_path, expose, fresh_ids, is_id, locate_mentions, pad_source, parse, path_from, rejection,
                   resample, twin, unmapped_literals)


# steps 1-6: one question into one twin (Build.question), in file order
POOLS = Path(__file__).resolve().parent / "pools"
TARGETS = 500                  # pad_source's fresh targets of a type: drawn from this many, an even sample of it
PROGRESS = 25                  # questions per progress line
HUB = 2000                     # non-boilerplate edges (in + out) from which a non-topic path node is a hub (kept
                               # real when the path only points at it, else the twin drops); a topic is cloned, its
                               # padding capped at HUB out-edges (``pad_source``)
REAL_TO_CLONE = "real constant would point at a clone"   # an unnamed constant the direction rule had to clone


class Dropped(Exception):
    """The question fails a step; its message is the drop reason."""


class Build:
    """One run: the real graph, the caches, the names index, and what the twins built so far contributed."""

    def __init__(self, real: Freebase, rng: random.Random, index: LabelIndex, call, pools, log: Path, split: str,
                 start: int = 1, avoid: set[str] = frozenset(), errors: Path | None = None):
        self.real, self.rng, self.index, self.call, self.pools, self.log, self.split, self.errors = (
            real, rng, index, call, pools, log, split, errors)
        ids, self.start, self.next_id, self.avoid = fresh_ids(start), start, start, set(avoid)
        self.salt = f"{rng.getrandbits(32):08x}"                     # the run's hash bucket of each type's members

        def fresh() -> str:                                           # one counter: clone ids unique on the replica
            self.next_id += 1
            return ids()
        self.fresh = fresh
        self.ranges: dict[str, LiteralRange | None] = {}              # None: not dates or numbers, never moves
        self.targets: dict[str, list[str] | None] = {}               # None: the type's scan failed, for the run
        self.hubs: dict[str, bool] = {}
        self.samples: dict[str, list[str] | None] = {}               # type -> its real labels, for dated kinds
        self.clones: list[Clone] = []
        self.twins: list[dict] = []
        self.padded: dict[str, set[str]] = {}                        # twin id -> the nodes and literals padding
                                                                      # added (targets, CVT clones, values)
        self.names, self.literals, self.notable, self.attrs, self.labels, self.types = {}, {}, {}, {}, {}, {}

    def pick_target(self, type: str) -> str | None:
        """A real node of ``type`` for padding, drawn from an even sample of the type less the avoid set."""
        def members(type: str) -> list[str]:
            return [s for s in self.real.members(type, TARGETS, self.salt) if is_id(s) and s not in self.avoid]
        pool = _cached(self.targets, type, members, self.errors)
        return self.rng.choice(pool) if pool else None

    def hub(self, id: str) -> bool:
        """Whether a path node has ``HUB`` edges or more (``clone_path``'s hub rule); asked once a run."""
        if id not in self.hubs:
            self.hubs[id] = _capped_degree(self.real, id, HUB) >= HUB
        return self.hubs[id]

    def question(self, q: dict, gold: Gold) -> dict:
        """Build the twin of ``q``, keep what it contributes, return it; ``Dropped`` when a step fails."""
        real = self.real
        expected, got = _execute(real, q)
        if expected != got:
            raise Dropped("gold differs on the real graph")
        q = {**q, "topic_entity_ids": _topics(q)}
        try:
            rows = real.select(expose(q["gold"]["query"]))
        except RowCap:                                                # the path is past the server's row cap
            raise Dropped(PATH_TOO_LARGE) from None
        if _path_size(gold, rows) > MAX_PATH:                         # before a query per node is spent on it
            raise Dropped(PATH_TOO_LARGE)
        path = path_from(gold, rows, set(q["topic_entity_ids"]), real.is_cvt, real.types)
        # a non-topic constant the text names is an identity, cloned and swapped; one it does not name ("Marriage"
        # behind "married") is a schema constraint of the question, kept real and never swapped
        anchors = sorted(n for n, node in path.nodes.items() if node.role in ("topic", "constant"))
        labels = real.labels(anchors)
        aliases = real.aliases(sorted(labels))
        shown = locate_mentions(q["question"], {s: [labels[s], *sorted(aliases.get(s, ()))] for s in labels})
        unnamed = {n for n in anchors if path.nodes[n].role == "constant" and n not in shown}
        try:
            clone = clone_path(path, CLOSED_CLASS_FREEBASE, self.fresh, real.target_type, hub=self.hub,
                               stay_real=lambda id: id in unnamed)
        except Rejected as e:                                         # a hub on the path, a class into a clone
            raise Dropped(str(e)) from None
        if any(n in clone.mapping and not path.nodes[n].is_cvt for n in unnamed):
            raise Dropped(REAL_TO_CLONE)            # the direction rule cloned it, but the text has no mention to swap
        # every clone but a mediator is named; an answer always is, even a nameless node (its twin answers by name)
        named = [s for s in clone.mapping if not path.nodes[s].is_cvt or path.nodes[s].role == "answer"]
        kept = [n for n in path.nodes if n not in clone.mapping]
        kept += [a for a in q["answer_ids"] if is_id(a) and a not in clone.mapping and a not in kept]
        labels.update(real.labels(sorted({*named, *kept} - set(anchors))))
        aliases.update(real.aliases(sorted(s for s in named if s in labels and s not in anchors)))
        source_labels = {s: [labels[s], *sorted(aliases.get(s, ()))] for s in named if s in labels}
        swap = {s: found for s, found in source_labels.items() if path.nodes[s].role in ("topic", "constant")}
        spans = locate_mentions(q["question"], swap)                  # twin's own conditions, before any spending
        if reason := rejection(q, clone, spans, swap):
            raise Dropped(reason)
        ranges = {p: r for p in {p for _, p, _ in path.literals}
                  if (r := _cached(self.ranges, p, real.literal_range, self.errors))}
        res = resample(gold, path, ranges, self.rng, q["question"], cloned=set(clone.mapping))   # before padding
        if not res.ok:
            raise Dropped(res.reason)
        if unmapped_literals(q["question"], q["gold"]["query"], res)[0]:
            raise Dropped(LITERAL_UNMAPPED)
        before, path_clones = len(clone.literals), set(clone.mapping.values())
        added = pad_source(clone, real.out_edges, self.pick_target, self.rng, fresh=self.fresh, is_cvt=real.is_cvt,
                           types_of=real.types, target_type=real.target_type,
                           literal_range=lambda p: _cached(self.ranges, p, real.literal_range, self.errors),
                           cap=HUB, objects=real.objects, shift=res.shift, scale=res.scale)
        padded = ({n for s, _, o in added for n in (s, o) if n not in path_clones}      # targets, CVT clones
                  | {v for _, _, v in clone.literals[before:]})                         # padded literal values
        kinds = {clone.mapping[s]: kind_of(clone.types[clone.mapping[s]]) if clone.mapping[s] in clone.types
                 else "entity" for s in named}
        attrs = {c: attrs_for(c, clone.kept_edges, labels) for c in kinds}
        written = {(clone.mapping[s], p, old): new for (s, p, old), new in res.values.items()}
        dates = {c: [written.get((c, p, v), v) for owner, p, v in clone.literals         # every date it will carry,
                     if owner == c and ((bounds.kind == "date") if (bounds := self.ranges.get(p))  # padded ones too
                                        else bool(_FULL_DATE.match(v)))]
                 for c in kinds}
        samples = {c: _cached(self.samples, t, lambda t: type_labels(real, t), self.errors)
                   for c, t in clone.types.items() if c in kinds}
        try:
            names = assign(clone, kinds, attrs, self.index, self.call, self.rng, self.log, self.pools,
                           {clone.mapping[s]: found[0] for s, found in source_labels.items()}, dates=dates,
                           samples=samples)
        except NamingFailed:
            raise Dropped("naming failed") from None
        record = twin(q, clone, names, res, swap, {k: labels[k] for k in kept if k in labels}, self.split,
                      unnamed=sorted(unnamed))
        if record is None:                                            # twin's conditions are the checks above
            raise RuntimeError("twin refused a question its checks passed")
        record["flags"] += [flag for flag in clone.flags if flag not in record["flags"]]   # hub_kept_real, topic_hub
        notable = {c: real.notable(s) for s, c in clone.mapping.items() if s in path.nodes}   # a padded CVT has none
        if padded:
            self.padded[record["id"]] = padded
        self.clones.append(clone)
        self.twins.append(record)
        self.names.update(names)
        self.literals.update(written)
        self.notable.update(notable)
        self.attrs.update(attrs)
        self.labels.update({clone.mapping[s]: found for s, found in source_labels.items()})
        self.types.update(clone.types)
        return record


def _run(b: Build, questions: list[dict], limit: int | None, out: Path, t0: float,
         shard: tuple[int, int] = (0, 1), offset: int = 0) -> tuple[int, dict, list]:
    """Build the shard's questions in file order: the questions read, the drops by reason, the meta lines."""
    drops: dict[str, list[str]] = collections.defaultdict(list)
    meta: list[dict] = []

    def drop(id: str, reason: str) -> None:
        drops[reason].append(id)
        meta.append({"twin_id": id, **dict.fromkeys(META), "dropped_reason": reason})
    read = parsed = 0
    for q in questions:
        if limit is not None and parsed - offset >= limit:
            break
        try:
            gold = parse(q["gold"]["query"])
        except Exception:
            if shard[0] == 0 and parsed >= offset:
                read += 1
                drop(q["id"], "gold query does not parse")
            continue
        parsed += 1
        if parsed <= offset or (parsed - offset - 1) % shard[1] != shard[0]:
            continue
        read += 1
        try:
            t = b.question(q, gold)
            meta.append({"twin_id": t["twin_id"], "shape": t["gold"]["shape"], "clone_ids": t["clone_ids"],
                         "answer_class": t["answer_class"], "flags": t["flags"], "dropped_reason": None})
        except Dropped as reason:
            drop(q["id"], str(reason))
        except Exception as e:
            drop(q["id"], f"error: {type(e).__name__}")
            with (out / "errors.log").open("a", encoding="utf-8") as f:
                f.write(f"== {q['id']}\n{traceback.format_exc()}\n")
        if parsed % PROGRESS == 0:
            print(f"[build] {parsed} questions: {len(b.twins)} twins, {sum(map(len, drops.values()))} dropped, "
                  f"{time.time() - t0:.0f} s", flush=True)
    return read, drops, meta


# the twins, their drops, the mapping and clones.nt written
def _write(b: Build, args, drops: dict, meta: list[dict], out: Path, seed: int) -> None:
    """Write the split, its meta lines, drops.json, clones.nt and mapping.json (with the build record) to ``out``."""
    source = (f"{args.set} {args.split}: every gold-path entity cloned into an extended Freebase replica and renamed, "
              f"its literals re-sampled (data/extension, seed {seed}"
              + (f", from parsed question {args.offset + 1}" if args.offset else "")
              + (f", the first {args.limit} parsed questions" if args.limit is not None else "")
              + (f", shard {args.shard[0]}/{args.shard[1]}" if args.shard else "") + ")")
    write_split(out / f"{args.split}.json", args.set, args.split, source, b.twins)
    write_meta(out / f"{args.split}_meta.jsonl", meta)
    _write_json(out / "drops.json", {reason: {"count": len(ids), "ids": ids} for reason, ids in drops.items()})
    datatypes = {p: _datatype(b.ranges.get(p)) for clone in b.clones for _, p, _ in clone.literals}
    ntriples(b.clones, b.names, b.literals, b.notable, datatypes, out / "clones.nt")
    _write_json(out / "mapping.json", {**_saved(b), "build": {
        "args": vars(args), "seed": seed, "ids": [b.start, b.next_id - 1], "name_literals_skipped": ntriples.skipped,
        "name_rejections": dict(assign.stats)}})


# step build
def _shard(text: str) -> tuple[int, int]:
    k, n = map(int, text.split("/"))
    if not 0 <= k < n:
        raise argparse.ArgumentTypeError(f"{text}: need 0 <= k < N")
    return k, n


def _have_pools(dir: Path) -> bool:
    return all((dir / file).is_file() for file in POOL_FILES.values())


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m unseen_builder.main build", description=(
        "Build the unseen twins of a WebQSP or CWQ split: clone each gold path into the extended replica, rename, "
        "validate."))
    p.add_argument("--set", choices=["webqsp", "cwq"])
    p.add_argument("--split", choices=["test", "val", "train"])
    p.add_argument("--offset", type=int, default=0,
                   help="skip the first N questions whose gold parses: a later wave of the split (trim's val wave: "
                        "--split train --offset 3000 --limit 600 reads parsed questions 3,001-3,600)")
    p.add_argument("--limit", type=int, help="the first N questions whose gold query parses (after --offset), in "
                                             "file order")
    p.add_argument("--real", required=True, help="the real Freebase SPARQL endpoint (read only)")
    p.add_argument("--ext", help="the extended replica's SPARQL endpoint")
    p.add_argument("--isql-host", help="the extended replica's isql host")
    p.add_argument("--isql-port", type=int, help="the extended replica's isql port")
    p.add_argument("--replica-root", type=Path, help="the extended replica's database directory")
    p.add_argument("--out", type=Path)
    p.add_argument("--splits", type=Path, default=None, help="the split files' root (default: data/questions)")
    p.add_argument("--names-model", default="gpt-4.1-mini")
    p.add_argument("--names-url", default=OPENAI_URL, help="an OpenAI-compatible chat completions URL")
    p.add_argument("--names-extra", type=json.loads, default=None,
                   help="JSON merged into every naming request, e.g. a local server's thinking switch")
    p.add_argument("--names-shared", type=Path, default=None,
                   help="a file of names handed out, shared by concurrent shards so they never reuse one")
    p.add_argument("--seed", type=int, default=20261017)
    p.add_argument("--judge-url", help="an OpenAI-compatible server for the model checks (its /v1)")
    p.add_argument("--judge-model", default="qwen3-30b-a3b")
    p.add_argument("--no-model-checks", action="store_true", help="skip the attribute and guess-the-source checks")
    p.add_argument("--pools", type=Path, default=POOLS,
                   help="the name pools (built from --real when missing, except by a shard)")
    p.add_argument("--pools-only", action="store_true", help="build the name pools when missing, then exit")
    p.add_argument("--id-start", type=int,
                   help="the first clone number (default: the number in <replica-root>/load/next_id.txt, else 1)")
    p.add_argument("--no-next-id", action="store_true", help="neither read nor write <replica-root>/load/next_id.txt")
    p.add_argument("--shard", type=_shard, metavar="k/N",
                   help="every N-th parsed question from the k-th (needs --id-start, --no-next-id and the pools)")
    p.add_argument("--dry-run", action="store_true", help="stop before the load and the suite; write to <out>/preview")
    p.add_argument("--suite-only", action="store_true",
                   help="re-run the suite on an earlier stage's files in --splits-in, writing to --out")
    p.add_argument("--splits-in", type=Path,
                   help="with --suite-only: the directory holding <split>.json and <split>_meta.jsonl; never written")
    p.add_argument("--build-in", type=Path,
                   help="with --suite-only: the directory holding mapping.json, clones.nt, drops.json and build.json "
                        "(default: --splits-in); never written")
    return p


def build_main(argv=None, make_graphs=lambda args: (Freebase(args.real, cache_dir=args.replica_root / "load" / "members"
                                                      if args.replica_root else None), Freebase(args.ext)),
         make_call=lambda args: openai_call(args.names_model, url=args.names_url, extra=args.names_extra),
         make_judge=lambda args: None if args.no_model_checks else chat_judge(args.judge_url, args.judge_model),
         loader=_load, splits: Path = SPLITS, avoid: set[str] | None = None) -> int:
    """The ``build`` step; the graphs, models, loader, splits folder and avoid set are injectable."""
    parser = _parser()
    args = parser.parse_args(argv)
    splits = args.splits or splits
    if args.pools_only:
        if not _have_pools(args.pools):
            build_pools(make_graphs(args)[0], args.pools, random.Random(args.seed))   # the plain seed: one set
        print(f"[build] name pools in {args.pools}", flush=True)
        return 0
    if missing := [f"--{name}" for name in ("set", "split", "ext", "out") if getattr(args, name) is None]:
        parser.error(f"the following arguments are required: {', '.join(missing)}")
    if args.dry_run and args.suite_only:
        parser.error("--dry-run or --suite-only, not both")
    if args.suite_only != bool(args.splits_in) or (args.build_in and not args.suite_only):
        parser.error("--suite-only reads --splits-in (and --build-in), and they need --suite-only")
    if args.suite_only and args.out.resolve() in {d.resolve() for d in (args.splits_in, args.build_in) if d}:
        parser.error("--out is an input directory: --suite-only never rewrites its inputs")
    if not (args.dry_run or args.suite_only) and None in (args.isql_host, args.isql_port, args.replica_root):
        parser.error("--isql-host, --isql-port and --replica-root are needed unless --dry-run or --suite-only")
    if args.no_model_checks == bool(args.judge_url) and not args.dry_run:
        parser.error("give --judge-url or --no-model-checks, not both or neither")
    if args.shard and (args.id_start is None or not args.no_next_id):
        parser.error("--shard needs --id-start and --no-next-id: shards take disjoint clone numbers")
    if args.shard and not args.suite_only and not _have_pools(args.pools):
        parser.error(f"--shard needs the name pools in {args.pools}: build them first with --pools-only")
    t0, out = time.time(), args.out / "preview" if args.dry_run else args.out
    shard = f"#{args.shard[0]}/{args.shard[1]}" if args.shard else ""
    wave = f"@{args.offset}" if args.offset else ""                               # a later wave of the split
    seed = args.seed + zlib.crc32(f"{args.set}/{args.split}{wave}{shard}".encode())   # no two builds draw alike
    out.mkdir(parents=True, exist_ok=True)
    real, ext = make_graphs(args)
    questions = json.loads((splits / args.set / f"{args.split}.json").read_text(encoding="utf-8"))["questions"]
    if args.suite_only:                        # a released split may hold twins topped up from another source split
        questions = [q for s in ("test", "val", "train") if (splits / args.set / f"{s}.json").is_file()
                     for q in json.loads((splits / args.set / f"{s}.json").read_text(encoding="utf-8"))["questions"]]
    if args.suite_only:
        files = args.build_in or args.splits_in
        saved = json.loads((files / "mapping.json").read_text(encoding="utf-8"))
        record = saved.get("build")                   # the build's own seed (a merge's or trim's mapping: none)
        seed = record["seed"] if record else seed
        twins = json.loads((args.splits_in / f"{args.split}.json").read_text(encoding="utf-8"))["questions"]
        report = _check(real, ext, twins, questions, saved, make_judge(args), out, seed,
                        inserts=(files / "clones.nt").read_text(encoding="utf-8").splitlines())
        _print_checks(report)
        reasons = excluded(report)
        written = _exclude(reasons, *_written(args.splits_in, files, args.split), args.splits_in, files, out,
                           args.split, always=True)
        path = files / "build.json"
        if path.is_file():                            # a finished build's, a merge's or trim's: counts follow files
            built = json.loads(path.read_text(encoding="utf-8"))
            if any(t["id"] in reasons for t in twins):
                built["counts"].update(_counts(*written))
            _write_json(out / "build.json", built)
        elif not record:
            raise RuntimeError(f"{files}/mapping.json has no build record: build.json cannot be recovered")
        else:                                         # its load or suite failed: recovered, so merge can take it
            counts = {**_counts(*written), "ids": record["ids"],
                      **{key: record[key] for key in ("name_literals_skipped", "name_rejections")}}
            _write_json(out / "build.json", {"args": record["args"], "seed": record["seed"], "counts": counts,
                                             "seconds": round(time.time() - t0, 1), "suite_only": True})
        return 0

    for log in ("names.log.jsonl", "errors.log"):
        (out / log).write_text("", encoding="utf-8")
    if not _have_pools(args.pools):
        build_pools(real, args.pools, random.Random(args.seed))       # shared by every build: the plain seed
    next_id = args.replica_root / "load" / NEXT_ID if args.replica_root and not args.no_next_id else None
    start = args.id_start if args.id_start is not None else (
        int(next_id.read_text(encoding="utf-8")) if next_id and next_id.is_file() else 1)
    if not args.dry_run and (held := _held(ext, [fresh_ids(start)()])):       # refused before anything is spent
        raise RuntimeError(f"{args.ext} already holds {held}, a clone of an earlier build: check {next_id} or "
                           "give --id-start")
    index = LabelIndex(collides_remote(real, ext), shared=args.names_shared)
    assign.stats.clear()
    b = Build(real, random.Random(seed), index, make_call(args), load_pools(args.pools), out / "names.log.jsonl",
              args.split, start, benchmark_nodes(splits) if avoid is None else avoid, out / "errors.log")
    _, drops, meta = _run(b, questions, args.limit, out, t0, args.shard or (0, 1), args.offset)
    _write(b, args, drops, meta, out, seed)
    report = None if args.dry_run else _load_and_check(b, args, ext, make_judge(args), loader, questions, next_id,
                                                       seed)
    written = _written(out, out, args.split)
    if report:                                        # the twins the suite rules out leave every file
        written = _exclude(excluded(report, b.padded), *written, out, out, out, args.split)

    counts = {**_counts(*written), "ids": [start, b.next_id - 1], "name_literals_skipped": ntriples.skipped,
              "name_rejections": dict(assign.stats)}
    seconds = round(time.time() - t0, 1)
    _write_json(out / "build.json", {"args": vars(args), "seed": seed, "counts": counts, "seconds": seconds})
    print(f"[build] {args.set} {args.split}: read {counts['read']}, built {counts['built']} (mod entities "
          f"{counts['mod_entities']}), {counts['triples']} triples, clone numbers {start}..{b.next_id - 1}, "
          f"{seconds} s", flush=True)
    for reason, n in sorted(counts["dropped"].items(), key=lambda item: -item[1]):
        print(f"[build]   dropped {n}: {reason}", flush=True)
    if report:
        _print_checks(report)
    return 0


# the steps
STEPS = {"build": build_main, "merge": merge_main, "trim": trim_main, "pairs": pairs_main, "load": load_main,
         "heldout": heldout_main, "pollution": pollution_main}
HELP = {"build": "build the twins of a split: expose, clone, pad, name, rewrite, load, validate",
        "merge": "join the shards of one build", "trim": "cut the validated twins to the release",
        "pairs": "write a released split as the harness's pair file",
        "load": "load released clones.nt files into a fresh copy of the database",
        "heldout": "the held-out filter: ask a model outside the evaluated set about every twin",
        "pollution": "rule out twins whose clones changed another split's real questions"}


def main(argv=None) -> int:
    """Run one step with its own flags: ``python -m unseen_builder.main STEP --help`` lists them."""
    p = argparse.ArgumentParser(prog="python -m unseen_builder.main", description=__doc__)
    steps = p.add_subparsers(dest="step", required=True, metavar="STEP")
    for name, text in HELP.items():
        steps.add_parser(name, help=text, add_help=False)
    args, rest = p.parse_known_args(argv)
    return STEPS[args.step](rest)


if __name__ == "__main__":
    sys.exit(main())
