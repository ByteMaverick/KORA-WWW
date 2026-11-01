"""The command line: `python -m harness run | score | compare`; settings from the shell, then `.env`."""
from __future__ import annotations

import argparse
import csv
import os
import sys

from harness.dataset import DATASETS, check
from harness.settings import TIMEOUT_SECONDS
from harness.systems import STOPS, SYSTEMS, UnsupportedStop, default_stop, framework_of

ALL_STOPS = ("native", "estimator", "llm", "gold")


def _load_dotenv() -> None:
    from kora.app import load_dotenv
    load_dotenv()


def _timeout_default() -> int:
    value = os.environ.get("HARNESS_TIMEOUT", "").strip()
    try:
        return int(value) if value else TIMEOUT_SECONDS
    except ValueError:
        raise SystemExit(f"HARNESS_TIMEOUT is {value!r}: a whole number of seconds")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m harness", description="Run KGQA systems on the paper's question sets "
                                "and score the runs.")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="run one system on one question set; one row per question")
    r.add_argument("--system", required=True, choices=SYSTEMS)
    r.add_argument("--stop", choices=ALL_STOPS, default=None,
                   help="who stops the run: " + "; ".join(f"{s}: {', '.join(v)}" for s, v in STOPS.items())
                        + " (the first is the default)")
    r.add_argument("--dataset", required=True, choices=DATASETS)
    r.add_argument("--split", default="test", choices=("test", "validation"))
    r.add_argument("--out", required=True, help="the run's folder")
    r.add_argument("--model", dest="engine", default=os.environ.get("KORA_LLM_MODEL", ""),
                   help="the served model id (default: KORA_LLM_MODEL)")
    r.add_argument("--api-base", default=None,
                   help="the model server's OpenAI-compatible base URL (default: OPENAI_BASE_URL, else "
                        "KORA_LLM_BASE_URL)")
    r.add_argument("--sparql-endpoint", default=None, help="Freebase's SPARQL endpoint (default: FREEBASE_SPARQL_URL)")
    r.add_argument("--timeout", type=int, default=None,
                   help=f"seconds per question (default: HARNESS_TIMEOUT, else {TIMEOUT_SECONDS}); `score` always "
                        f"counts a row over {TIMEOUT_SECONDS} s as a miss, the paper's limit")
    r.add_argument("--max-samples", type=int, default=None, help="the first N questions only")
    r.add_argument("--num-shards", type=int, default=1, help="split the questions into N shards by index modulo N")
    r.add_argument("--shard-index", type=int, default=0, help="the shard this process runs, in [0, N-1]")
    r.add_argument("--ids", default=None, help="a JSON file of question ids: run only those")
    r.add_argument("--resume", action="store_true", help="skip the questions the shard file already holds")

    s = sub.add_parser("score", help="score runs: one CSV row per run")
    s.add_argument("runs", nargs="+", help="run folders")
    s.add_argument("--csv", default=None, help="write the CSV here (default: stdout)")
    s.add_argument("--no-extract", action="store_true",
                   help="do not call the extractor: free-text rows it has not read stay unscored")

    c = sub.add_parser("compare", help="the paired difference of two runs (A minus B), with its bootstrap interval")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--no-extract", action="store_true")
    return p


def _run(args: argparse.Namespace) -> int:
    try:
        args.framework = framework_of(args.system, args.stop)
    except UnsupportedStop as error:
        print(f"[harness] {error}", file=sys.stderr)
        return 2
    args.stop = args.stop or default_stop(args.system)
    try:
        check(args.dataset, args.split)
    except ValueError as error:
        print(f"[harness] {error}", file=sys.stderr)
        return 2
    if not args.engine:
        print("[harness] no model: pass --model or set KORA_LLM_MODEL", file=sys.stderr)
        return 2
    args.api_base = args.api_base or os.environ.get("OPENAI_BASE_URL", "") or os.environ.get("KORA_LLM_BASE_URL", "")
    args.timeout = args.timeout if args.timeout is not None else _timeout_default()
    args.kg_name = "freebase"
    endpoint = args.sparql_endpoint or os.environ.get("FREEBASE_SPARQL_URL", "")
    if endpoint:
        os.environ["SPARQL_ENDPOINT"] = endpoint
    elif args.dataset != "metaqa" and args.system != "direct_llm":
        print("[harness] no Freebase endpoint: pass --sparql-endpoint or set FREEBASE_SPARQL_URL", file=sys.stderr)
        return 2
    if args.dataset == "metaqa" and args.system != "direct_llm" and not os.environ.get("NEO4J_METAQA_URI"):
        print("[harness] no MetaQA graph: set NEO4J_METAQA_URI", file=sys.stderr)
        return 2
    if args.stop == "estimator" and not os.environ.get("ESTIMATOR_URL"):
        print("[harness] --stop estimator needs the Stopping Estimator's server: set ESTIMATOR_URL", file=sys.stderr)
        return 2
    if args.system == "kora":             # a KORA setting that disagrees with the run stops it here, not per question
        from harness.systems import kora
        try:
            kora.check_settings(kora.environment(args.stop, args.dataset))
        except RuntimeError as error:
            print(str(error), file=sys.stderr)
            return 2
    from harness.runner import run_dataset
    try:
        return run_dataset(args)
    except (FileNotFoundError, ValueError) as error:
        print(f"[harness] {error}", file=sys.stderr)
        return 2


def _score(args: argparse.Namespace) -> int:
    from harness.metrics.common.report import COLUMNS, RunError, score_run
    try:
        rows = [score_run(d, run_extractor=not args.no_extract) for d in args.runs]
    except (RunError, FileNotFoundError) as error:
        print(f"[harness] {error}", file=sys.stderr)
        return 2
    out = open(args.csv, "w", newline="", encoding="utf-8") if args.csv else sys.stdout
    try:
        w = csv.DictWriter(out, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    finally:
        if args.csv:
            out.close()
    return 0


def _compare(args: argparse.Namespace) -> int:
    from harness.metrics.common.report import RunError, compare
    try:
        rows = compare(args.a, args.b, run_extractor=not args.no_extract)
    except (RunError, FileNotFoundError) as error:
        print(f"[harness] {error}", file=sys.stderr)
        return 2
    pct = lambda x: "" if x is None else f"{100 * x:.1f}"
    diff = lambda x: "" if x is None else f"{100 * x:+.1f}"
    for r in rows:
        ci = r["ci95"]
        extra = f"  McNemar p={r['mcnemar_p']:.4f}" if "mcnemar_p" in r else ""
        print(f"{r['metric']:>4}: A {pct(r['a'])}  B {pct(r['b'])}  A-B {diff(r['difference'])}  "
              f"95% CI [{diff(ci[0]) if ci else ''}, {diff(ci[1]) if ci else ''}]  n={r['n']}{extra}")
    return 0


def main(argv: list[str] | None = None) -> int:
    _load_dotenv()
    args = _parser().parse_args(argv)
    if args.command == "run":
        return _run(args)
    if args.command == "score":
        return _score(args)
    return _compare(args)
