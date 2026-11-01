"""GCR step 2's decoding from the settings table, as shell exports for the patched client."""
from __future__ import annotations

import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from harness import settings  # noqa: E402


def values(model: str) -> dict[str, str]:
    d = settings.decoding("gcr", model)
    return {"GCR_MAX_TOKENS": str(d["max_tokens"]), "GCR_TOP_P": str(d["top_p"]), "GCR_TOP_K": str(d["top_k"]),
            "GCR_REASONING_EFFORT": settings.reasoning_effort(model) or ""}


if __name__ == "__main__":
    print("export " + " ".join(f"{k}={shlex.quote(v)}" for k, v in values(sys.argv[1]).items()))
