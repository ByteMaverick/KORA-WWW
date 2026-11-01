"""GCR's step-2 client as the run script runs it, for a preflight: one request through the patched authors' class."""
from __future__ import annotations

import importlib.util
import os
import shutil
import sys
import tempfile
import types
from contextlib import contextmanager
from pathlib import Path

from baselines.gcr import patch_upstream, step2_env

PINNED = Path(__file__).resolve().parent / "upstream"


@contextmanager
def _clone():
    """The patched clone's folder: GCR_CLONE's, or the authors' files in upstream/ patched in a temporary folder."""
    given = os.environ.get("GCR_CLONE")
    if given:
        chat = Path(given) / "src" / "llms" / "chatgpt.py"
        if not chat.exists() or patch_upstream.MARK not in chat.read_text():
            raise RuntimeError(f"{given}: not patched with {patch_upstream.MARK.split('/')[-1]} "
                               f"(git -C {given} checkout -- src/llms; python -m baselines.gcr.patch_upstream {given})")
        yield Path(given)
        return
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp) / "graph-constrained-reasoning"
        shutil.copytree(PINNED, repo)
        patch_upstream.patch(repo)
        yield repo


def _module(repo: Path):
    """The patched chatgpt.py as its package would import it, under a name of its own."""
    name = "gcr_step2_llms"
    pkg = types.ModuleType(name)
    pkg.__path__ = [str(repo / "src" / "llms")]
    saved = {k: sys.modules.get(k) for k in (name, f"{name}.chatgpt", f"{name}.base_language_model")}
    sys.modules[name] = pkg
    try:
        spec = importlib.util.spec_from_file_location(f"{name}.chatgpt", repo / "src" / "llms" / "chatgpt.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"{name}.chatgpt"] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        for k, v in saved.items():
            sys.modules.pop(k, None) if v is None else sys.modules.__setitem__(k, v)


def send(engine: str, url: str):
    """One step-2 request for `engine` to `url` (.../v1) through the patched authors' client; their reply."""
    env = {**step2_env.values(engine), "OPENAI_BASE_URL": url,
           "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY") or "EMPTY"}
    saved = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        with _clone() as repo:
            mod = _module(repo)
            mod.time = types.SimpleNamespace(sleep=lambda s: None)      # their 30 s pause between attempts
            llm = mod.ChatGPT(types.SimpleNamespace(retry=0, model_name=engine))
            llm.prepare_for_inference()
            return llm.generate_sentence("Question: ping?\nReasoning Paths:\n")
    finally:
        for k, v in saved.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
