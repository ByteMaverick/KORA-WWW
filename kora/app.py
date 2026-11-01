"""KORA's settings, from the environment, and its command line, `python -m kora answer`."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from pydantic import BaseModel

DEFAULT_SECONDS = 140.0
DEFAULT_TOKENS = 400_000

STOP_MODES: dict[str, dict[str, str]] = {
    "estimator": {"KORA_ROOT_JUDGE": "judge", "KORA_JUDGE_IMPL": "estimator", "KORA_CLOCK_FALLBACK": "best"},
    "llm": {"KORA_ROOT_JUDGE": "knowledge", "KORA_JUDGE_IMPL": "llm", "KORA_CLOCK_FALLBACK": "held"},
    "gold": {"KORA_ROOT_JUDGE": "oracle", "KORA_ORACLE": "strict", "KORA_JUDGE_IMPL": "llm",
             "KORA_CLOCK_FALLBACK": "held"},
}


def load_dotenv(path: Path | None = None) -> None:
    """Read `KEY=value` lines from `.env` into the environment, keeping what the shell already set."""
    env = path or Path.cwd() / ".env"
    if not env.is_file():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key and value and key not in os.environ:
            os.environ[key] = value


def stop_mode(value: str | None = None) -> str:
    """`KORA_STOP=estimator|llm|gold`, estimator when unset: who stops the run (see `STOP_MODES`)."""
    if value is None:
        value = os.environ.get("KORA_STOP", "")
    value = value.strip().lower() or "estimator"
    if value not in STOP_MODES:
        raise ValueError(f"KORA_STOP is {value!r}: one of {', '.join(STOP_MODES)}")
    return value


def apply_stop(mode: str | None = None) -> str:
    """Write the chosen mode's settings and the arm defaults into the environment."""
    mode = stop_mode(mode)
    os.environ.update(STOP_MODES[mode])
    os.environ.setdefault("KORA_DECOMPOSER", "structural")
    return mode


GRAPH_SETTINGS = {"freebase-sparql": "FREEBASE_SPARQL_URL", "metaqa": "NEO4J_METAQA_URI"}


def missing(mode: str, graph: str) -> str | None:
    """The first setting a run in this mode on this graph needs and does not have, or None."""
    needed = [GRAPH_SETTINGS[graph]]
    if mode == "estimator":
        needed.append("ESTIMATOR_URL")
    needed.append("OPENAI_API_KEY")
    return next((name for name in needed if not os.environ.get(name, "").strip()), None)


def clock_fallback(value: str | None = None) -> str:
    """`KORA_CLOCK_FALLBACK=held|best`, best when unset: what a run that never stopped answers with."""
    if value is None:
        value = os.environ.get("KORA_CLOCK_FALLBACK", "best")
    value = value.strip().lower()
    if value not in ("held", "best"):
        raise ValueError(f"KORA_CLOCK_FALLBACK is {value!r}: held or best")
    return value


def judge_impl(value: str | None = None) -> str:
    """`KORA_JUDGE_IMPL=estimator|llm` (estimator when unset): what scores the leaf and root judges."""
    if value is None:
        value = os.environ.get("KORA_JUDGE_IMPL", "")
    value = value.strip().lower() or "estimator"
    if value not in ("estimator", "llm"):
        raise ValueError(f"KORA_JUDGE_IMPL is {value!r}: estimator or llm")
    return value


def estimator_url(value: str | None = None) -> str | None:
    """The estimator server's URL, `ESTIMATOR_URL`; None when unset."""
    if value is None:
        value = os.environ.get("ESTIMATOR_URL", "")
    return value.strip() or None


def estimator_tau(value: str | None = None) -> float | None:
    """`ESTIMATOR_TAU` when set; None means the threshold the judge server reports."""
    if value is None:
        value = os.environ.get("ESTIMATOR_TAU", "")
    if not value.strip():
        return None
    try:
        return float(value)
    except ValueError:
        raise ValueError(f"ESTIMATOR_TAU is {value!r}: a number") from None


def root_judge(value: str | None = None) -> str:
    """`KORA_ROOT_JUDGE=judge|knowledge|oracle`, judge when unset: who decides the root."""
    if value is None:
        value = os.environ.get("KORA_ROOT_JUDGE", "judge")
    value = value.strip().lower()
    if value not in ("judge", "knowledge", "oracle"):
        raise ValueError(f"KORA_ROOT_JUDGE is {value!r}: judge, knowledge or oracle")
    return value


def oracle_rule(value: str | None = None) -> str:
    """`KORA_ORACLE=generous|strict`, generous when unset: what the oracle accepts at the root."""
    if value is None:
        value = os.environ.get("KORA_ORACLE", "generous")
    value = value.strip().lower()
    if value not in ("generous", "strict"):
        raise ValueError(f"KORA_ORACLE is {value!r}: generous or strict")
    return value


def property_choice(value: str | None = None) -> str:
    """`KORA_PROPERTY=model|match` (model when unset): how an operate's property is chosen."""
    if value is None:
        value = os.environ.get("KORA_PROPERTY", "model")
    value = value.strip().lower()
    if value not in ("model", "match"):
        raise ValueError(f"KORA_PROPERTY is {value!r}: model or match")
    return value


class Settings(BaseModel):
    """One object, built from the environment."""

    model_config = {"extra": "forbid"}

    stop: str = "estimator"
    tokens: int = DEFAULT_TOKENS
    seconds: float = DEFAULT_SECONDS
    property_choice: str = "model"
    fallback: str = "best"

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            stop=stop_mode(),
            tokens=int(os.environ.get("KORA_TOKENS", DEFAULT_TOKENS)),
            seconds=float(os.environ.get("KORA_SECONDS", DEFAULT_SECONDS)),
            property_choice=property_choice(),
            fallback=clock_fallback(),
        )


# The command line below reads the settings as `settings.<name>`: this module is that namespace.
settings = sys.modules[__name__]


def parse_topics(raw: list[str]) -> list[tuple[str, str]]:
    """`ID` or `ID=Name`, as (id, name); the name is empty when not given."""
    topics = []
    for item in raw:
        ident, _, name = item.partition("=")
        topics.append((ident.strip(), name.strip().replace("_", " ")))
    return topics


def answer(question: str, graph: str, topics: list[tuple[str, str]], gold: list[str]) -> dict:
    """One question through the harness's `Harness.run`, the one place the three root judges are wired."""
    from kora.run import Harness
    from kora.modules.controller.models import Budget

    config = settings.Settings.from_env()
    harness = Harness(graph, judge=config.stop == "gold",
                      budget=Budget(tokens=config.tokens, seconds=config.seconds))
    try:
        anchors = [(ident, name or harness.kg.label(ident) or ident) for ident, name in topics]
        row = harness.run({"q": question, "gold": gold, "id": None, "hops": None, "anchor": anchors[0],
                           "anchors": anchors})
    finally:
        harness.close()
    answers = row.get("projection") or []
    held_back = ("q", "gold", "i", "id")
    return {"question": question, "stop": config.stop, "answers": answers,
            "provenance": {k: v for k, v in row.items() if k not in held_back}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m kora", description="Ask a knowledge graph a question.")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("answer", help="one question, end to end; prints the answers and the provenance as JSON")
    p.add_argument("question")
    p.add_argument("--graph", default="freebase-sparql", choices=sorted(settings.GRAPH_SETTINGS))
    p.add_argument("--topic", action="append", default=[], metavar="ID",
                   help="a topic entity's id, or ID=Name; repeatable")
    p.add_argument("--gold", action="append", default=[], metavar="ANSWER",
                   help="a true answer; repeatable; required by, and only read under, KORA_STOP=gold")
    args = parser.parse_args(argv)

    try:
        settings.load_dotenv()
        mode = settings.apply_stop()
    except ValueError as error:
        print(f"kora: {error}", file=sys.stderr)
        return 2
    if not args.topic:
        print("kora: give the question's topic entity with --topic ID", file=sys.stderr)
        return 2
    if mode == "gold" and not args.gold:
        print("kora: KORA_STOP=gold needs the true answer: pass --gold ANSWER", file=sys.stderr)
        return 2
    if mode != "gold" and args.gold:
        print(f"kora: --gold is read only under KORA_STOP=gold, not {mode}", file=sys.stderr)
        return 2
    absent = settings.missing(mode, args.graph)
    if absent:
        print(f"kora: {absent} is not set (KORA_STOP={mode}, graph {args.graph}); put it in .env or the shell",
              file=sys.stderr)
        return 2

    try:
        out = answer(args.question, args.graph, parse_topics(args.topic), args.gold)
    except RuntimeError as error:   # the graph or the estimator server did not answer
        print(f"kora: {error}", file=sys.stderr)
        return 1
    print(json.dumps(out, indent=1, default=str))
    return 0 if out["provenance"].get("error") is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
