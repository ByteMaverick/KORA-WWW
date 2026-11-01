"""Run one of GCR's workflow scripts with each question timed and held to the harness's per-question limit."""
from __future__ import annotations

import argparse
import ast
import json
import multiprocessing
import os
import signal
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from harness import settings  # noqa: E402
from baselines.gcr.patch_upstream import DROPPED  # noqa: E402

FUNCTION = {1: "prediction", 2: "make_prediction"}
#: Their own scoring, called once after the loop (src/utils/qa_utils.py). Not run: the export leaves the gold out
#: (a_entity and answer empty, export.py), on which it divides by zero (eval_acc: / len(answer)), and the harness
#: scores convert.py's rows instead.
EVALUATION = {1: "eval_path_result_w_ans", 2: "eval_result"}

# Set in run(), inherited by forked workers.
_INNER = None
_STEP = 0
_TIMES = ""
_LIMIT = float(settings.TIMEOUT_SECONDS)
_EARLIER: dict = {}
_WRITE = threading.Lock()


class QuestionTimeout(BaseException):
    """Raised by the alarm; a BaseException so the authors' `except Exception` retry loops let it through."""


def load_times(*paths: str | Path) -> dict:
    """qid -> {1: seconds, 2: seconds, "timeout": bool}, from times files (the last line per step wins)."""
    out: dict = {}
    for path in paths:
        p = Path(path)
        if not p.exists():
            continue
        for line in p.open(encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                q = out.setdefault(str(r["id"]), {"timeout": False})
                q[int(r["step"])] = float(r["seconds"])
                q["timeout"] = q["timeout"] or bool(r.get("timeout"))
    return out


def _record(qid: str, seconds: float, timeout: bool) -> None:
    line = json.dumps({"id": qid, "step": _STEP, "seconds": round(seconds, 3), "timeout": timeout}) + "\n"
    with _WRITE, open(_TIMES, "a", encoding="utf-8") as f:   # one short line per write: whole from forked workers;
        f.write(line)                                         # the lock orders a thread pool's workers


def _alarm(signum, frame):
    raise QuestionTimeout()


def _bounded(call, budget: float):
    """(call's result, False), or (None, True) past budget seconds (SIGALRM, or a waited-for thread)."""
    if threading.current_thread() is not threading.main_thread():
        box: dict = {}

        def target():
            try:
                box["result"] = call()
            except BaseException as e:   # handed back to the caller, as the call itself would raise it
                box["error"] = e
        worker = threading.Thread(target=target, daemon=True)
        worker.start()
        worker.join(budget)
        if worker.is_alive():
            setattr(worker, DROPPED, True)
            return None, True
        if "error" in box:
            raise box["error"]
        return box["result"], False
    previous = signal.signal(signal.SIGALRM, _alarm)
    signal.setitimer(signal.ITIMER_REAL, budget)
    try:
        return call(), False
    except QuestionTimeout:
        return None, True
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def timed_call(data, *args, **kwargs):
    """The authors' per-question function, timed and bounded (bound as FUNCTION[step] in their namespace)."""
    qid = str(data["id"])
    earlier = _EARLIER.get(qid, {})
    if earlier.get("timeout"):                 # it ran out before (step 1: no step 2; a rerun: not again)
        if _STEP not in earlier:
            _record(qid, 0.0, True)
        return None
    if _STEP in earlier:                       # their resume: done in an earlier run, timed then
        return _INNER(data, *args, **kwargs)
    budget = _LIMIT - sum(v for k, v in earlier.items() if k != "timeout")
    if budget <= 0:
        _record(qid, 0.0, True)
        return None
    start = time.perf_counter()
    result, timeout = _bounded(lambda: _INNER(data, *args, **kwargs), budget)
    seconds = time.perf_counter() - start
    timeout = timeout or seconds > budget      # a bare `except:` in their code swallowed the alarm
    _record(qid, seconds, timeout)
    return None if timeout else result


def _not_evaluated(predict_file, *args, **kwargs) -> None:
    print(f"[gcr timed] their evaluation of {predict_file} not run: the gold is not in GCR's input; "
          "the harness scores convert.py's rows")


def _is_main_guard(node) -> bool:
    return (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name) and node.test.left.id == "__name__")


def run(script: str, step: int, times: str, argv: list[str], limit: float | None = None, after: tuple = ()) -> None:
    """Run the authors' script as `python <script> <argv>` would, with FUNCTION[step] timed."""
    global _INNER, _STEP, _TIMES, _LIMIT, _EARLIER
    source = Path(script).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=script)
    guards = [n for n in tree.body if _is_main_guard(n)]
    if len(guards) != 1:
        raise SystemExit(f"{script}: expected one `if __name__ == '__main__'` block, found {len(guards)}")
    defs = ast.Module(body=[n for n in tree.body if n is not guards[0]], type_ignores=[])
    namespace = {"__name__": "gcr_workflow", "__file__": str(script)}
    sys.path.insert(0, str(Path(script).resolve().parent))
    exec(compile(defs, script, "exec"), namespace)
    name, evaluation = FUNCTION[step], EVALUATION[step]
    for needed in (name, evaluation):
        if needed not in namespace:
            raise SystemExit(f"{script}: no function {needed}; upstream changed")
    namespace[evaluation] = _not_evaluated
    _INNER, _STEP, _TIMES = namespace[name], step, str(times)
    _LIMIT = float(settings.TIMEOUT_SECONDS if limit is None else limit)
    _EARLIER = load_times(*after, times)
    namespace[name] = timed_call
    multiprocessing.set_start_method("fork", force=True)
    saved = sys.argv
    sys.argv = [str(script), *argv]
    try:
        exec(compile(ast.Module(body=guards[0].body, type_ignores=[]), script, "exec"), namespace)
    finally:
        sys.argv = saved
        _INNER, _STEP, _TIMES, _LIMIT, _EARLIER = None, 0, "", float(settings.TIMEOUT_SECONDS), {}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], allow_abbrev=False)
    p.add_argument("--step", type=int, choices=(1, 2), required=True)
    p.add_argument("--times", required=True, help="the times file this step writes (and resumes from)")
    p.add_argument("--after", action="append", default=[], help="step 2: step 1's times file")
    p.add_argument("script")
    p.add_argument("their_args", nargs=argparse.REMAINDER, help="the script's own arguments, unchanged")
    a = p.parse_args(argv)
    os.makedirs(os.path.dirname(os.path.abspath(a.times)), exist_ok=True)
    run(a.script, a.step, a.times, a.their_args, after=tuple(a.after))
    return 0


if __name__ == "__main__":
    from baselines.gcr import timed as _self   # the functions pickled into their Pool are this module's
    raise SystemExit(_self.main())
