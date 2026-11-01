"""What produced a result row (code, authors' code, settings, split): row["manifest"] in GCR's and KG-R1's rows."""
from __future__ import annotations

import functools
import hashlib
import os
import re
import subprocess

from harness import settings
from harness.metrics.reasoning import SCORER_VERSION

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#: Where each system's authors' code is vendored (a system missing here runs only harness code).
VENDORED = {"kbbinder_official": "baselines/kbbinder"}


def _git(path: str, *cmd: str) -> str | None:
    try:
        out = subprocess.run(["git", "-C", path, *cmd], capture_output=True, text=True, timeout=20)
    except Exception:
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def _checkout(path: str) -> str | None:
    """HEAD of the git checkout rooted at `path`; None when `path` is not the top of one."""
    top = _git(path, "rev-parse", "--show-toplevel")
    if top is None or os.path.realpath(top) != os.path.realpath(path):
        return None
    return _git(path, "rev-parse", "HEAD")


def code_record(home: str, stamp: str) -> dict:
    """The code at `home`: {commit, dirty, source}, from git, else from the commit in a `stamp` file."""
    head = _checkout(home)
    if head is not None:
        return {"commit": head, "dirty": bool(_git(home, "status", "--porcelain", "--untracked-files=no")),
                "source": "git"}
    commit = open(stamp).read().strip() if stamp and os.path.exists(stamp) else None
    return {"commit": commit or None, "dirty": None, "source": "stamp" if commit else None}


@functools.lru_cache(maxsize=None)
def repository_commit() -> dict:
    """This repository's code: git on a checkout, a COMMIT file at its root on a copy without .git."""
    return code_record(ROOT, os.path.join(ROOT, "COMMIT"))


@functools.lru_cache(maxsize=None)
def tree_sha256(folder: str) -> str | None:
    """sha256 over the vendored files of a folder (paths and contents, sorted; caches and data excluded)."""
    base = os.path.join(ROOT, folder)
    if not os.path.isdir(base):
        return None
    h = hashlib.sha256()
    for dp, dn, fn in os.walk(base):
        dn[:] = sorted(d for d in dn if d not in ("__pycache__", "tests", "data"))
        for f in sorted(fn):
            if f.endswith((".pyc", ".log")):
                continue
            p = os.path.join(dp, f)
            if os.path.islink(p):
                continue
            h.update(os.path.relpath(p, base).encode() + b"\0" + open(p, "rb").read() + b"\0")
    return h.hexdigest()


def upstream(framework: str) -> dict | None:
    folder = VENDORED.get(framework)
    if folder is None:
        return None
    doc = os.path.join(ROOT, folder, "UPSTREAM.md")
    commit = None
    if os.path.exists(doc):
        m = re.search(r"\bat `([0-9a-f]{7,40})`|^`([0-9a-f]{40})`", open(doc, encoding="utf-8").read(), re.M)
        commit = m and (m.group(1) or m.group(2))
    return {"folder": folder, "commit": commit, "sha256": tree_sha256(folder)}


def _decoding(framework: str, engine: str) -> dict | None:
    try:
        return settings.decoding(framework, engine)
    except KeyError:
        return None


def file_sha256(path: str) -> str | None:
    if not path or not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def run_manifest(args) -> dict:
    framework, engine = getattr(args, "framework", ""), getattr(args, "engine", "")
    path = getattr(args, "dataset_path", "") or ""
    model = settings.model_of(engine)
    return {
        "code": repository_commit(),
        "kora": None,
        "upstream": upstream(framework),
        "system": settings.system(framework),
        "decoding": _decoding(framework, engine),
        "model": {"engine": engine, "row": model, "settings": settings.MODELS.get(model),
                  "reasoning_effort": settings.reasoning_effort(engine),
                  "cli_temperature": getattr(args, "temperature", None),
                  "cli_max_tokens": getattr(args, "max_tokens", None),
                  "endpoints": getattr(args, "llm_endpoints", None) or getattr(args, "api_base", None)},
        "timeout_seconds": int(getattr(args, "question_timeout_seconds", 0) or 0),
        "split": {"file": os.path.relpath(os.path.abspath(path), ROOT) if path else None, "sha256": file_sha256(path)},
        "scorer_version": SCORER_VERSION,
    }
