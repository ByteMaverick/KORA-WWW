"""Let GCR's answer step call our model servers (edits in UPSTREAM.md); run once on a clone, safe again."""
from __future__ import annotations

import sys
from pathlib import Path

MARK = "# patched by baselines/gcr/patch_upstream.py v3"
OLD_MARK = "# patched by baselines/gcr/patch_upstream.py"
EDITS = 9   # MARK lines in chatgpt.py once patched (edits 2, 3, 6 one each; 4, 5 and 7 two each)
#: The attribute timed.py sets on a thread it has dropped (edit 7 reads it).
DROPPED = "gcr_dropped"
#: The authors' files at 9518e8e that this patch is written against.
UPSTREAM_SHA256 = {
    "__init__.py": "2256d23744b81b1f29994dc35516f6b2c692c7aea5c05c437469e6fba270a613",
    "chatgpt.py": "d16f56b2738eb92241c1de3172a05d2b514c44d487211874dfcad578bb63bfff",
    "base_language_model.py": "fdcbdc8e074507a6c48bca7516dc92b13b34d0748d15b39fb08cc8e922870491",
}

_INIT = [("    'gpt': ChatGPT,",
          f"    'gpt': ChatGPT,\n    'qwen': ChatGPT,  {MARK}\n    'gpt-oss': ChatGPT,  {MARK}")]

_CHAT = [
    ("import time\n", f"import time\nimport threading  {MARK}: edit 7\n"),
    ("os.environ['TIKTOKEN_CACHE_DIR'] = './tmp'\n",
     f"os.environ.setdefault('TIKTOKEN_CACHE_DIR', './tmp')  {MARK}: an environment's cache wins\n"),
    ("OPENAI_MODEL = ['gpt-4', 'gpt-3.5-turbo']\n",
     "OPENAI_MODEL = ['gpt-4', 'gpt-3.5-turbo']\n"
     f"SERVED = ('qwen', 'gpt-oss')  {MARK}: our served models, by name\n"),
    ('''    else:
        raise NotImplementedError(f"""get_token_limit() is not implemented for model {model}.""")''',
     f'''    elif model.lower().startswith(SERVED):
        num_tokens_limit = 32768  {MARK}: the servers' context
    else:
        raise NotImplementedError(f"""get_token_limit() is not implemented for model {{model}}.""")'''),
    ('''        try:
            encoding = tiktoken.encoding_for_model(self.model_name)
            num_tokens = len(encoding.encode(text))
        except KeyError:
            raise KeyError(f"Warning: model {self.model_name} not found.")''',
     f'''        if self.model_name.lower().startswith(SERVED):
            return len(text) // 3  {MARK}: never tiktoken for a served model (no download offline)
        try:
            encoding = tiktoken.encoding_for_model(self.model_name)
            num_tokens = len(encoding.encode(text))
        except Exception:  {MARK}
            num_tokens = len(text) // 3  # a count that cannot be made is estimated; errs long'''),
    ('''        self.client = client
''',
     f'''        self.client = client
        self.decoding = {{"max_tokens": int(os.environ["GCR_MAX_TOKENS"]), "top_p": float(os.environ["GCR_TOP_P"]),
                         "extra_body": {{"top_k": int(os.environ["GCR_TOP_K"])}}}}  {MARK}: the settings table
        if os.environ.get("GCR_REASONING_EFFORT"):
            self.decoding["extra_body"]["reasoning_effort"] = os.environ["GCR_REASONING_EFFORT"]
'''),
    ('''        while cur_retry <= num_retry:
            try:''',
     f'''        while cur_retry <= num_retry:
            if getattr(threading.current_thread(), "{DROPPED}", False):  {MARK}: timed.py dropped it
                return None
            try:'''),
    ('''                    timeout=60,
                    temperature=0.0
                    )''',
     f'''                    timeout=60,
                    temperature=0.0,
                    **self.decoding,  {MARK}
                    )'''),
]


def _apply(path: Path, edits: list) -> None:
    s = path.read_text()
    if MARK in s:
        return
    if OLD_MARK in s:
        repo = path.parents[2]
        raise SystemExit(f"{path} carries an earlier version of this patch; restore it first: "
                         f"git -C {repo} checkout -- {path.relative_to(repo)}")
    for old, new in edits:
        if old not in s:
            raise SystemExit(f"{path}: upstream changed ({old.splitlines()[0]!r} not found); patch by hand")
        s = s.replace(old, new, 1)
    path.write_text(s)


def patch(repo: Path) -> None:
    repo = Path(repo)
    _apply(repo / "src" / "llms" / "__init__.py", _INIT)
    _apply(repo / "src" / "llms" / "chatgpt.py", _CHAT)
    (repo / "tmp").mkdir(exist_ok=True)   # their TIKTOKEN_CACHE_DIR, relative to the clone step 2 runs from
    assert (repo / "src" / "llms" / "chatgpt.py").read_text().count(MARK) == EDITS, "patch incomplete"


if __name__ == "__main__":
    patch(Path(sys.argv[1]))
    print("patched", sys.argv[1])
