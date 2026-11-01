"""The Stopping Estimator's server for the baselines' estimator arms: its URL (`ESTIMATOR_URL`) and the cut a
candidate's p must reach (`ESTIMATOR_TAU` when set, else the threshold the server reports on GET /health, from its
checkpoint's threshold.json). Each adapter sends its own POST /score requests."""
from __future__ import annotations

import json
import os
import urllib.request

_SERVED: dict[str, float] = {}


def url() -> str:
    found = os.environ.get("ESTIMATOR_URL", "").strip().rstrip("/")
    if not found:
        raise RuntimeError("the estimator arm needs the Stopping Estimator's server: set ESTIMATOR_URL")
    return found


def threshold() -> float:
    """ESTIMATOR_TAU when set, else the server's threshold (asked once per URL); never a default."""
    explicit = os.environ.get("ESTIMATOR_TAU", "").strip()
    if explicit:
        try:
            return float(explicit)
        except ValueError:
            raise ValueError(f"ESTIMATOR_TAU is {explicit!r}: a number") from None
    base = url()
    if base not in _SERVED:
        try:
            with urllib.request.urlopen(base + "/health", timeout=30) as response:
                info = json.load(response)
        except Exception as error:  # noqa: BLE001 - reported, never defaulted
            raise RuntimeError(f"the estimator server at {base} did not answer /health ({error!r}): start it, or "
                               "set ESTIMATOR_TAU") from error
        value = info.get("threshold") if isinstance(info, dict) else None
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise RuntimeError(f"the estimator server at {base} reports no threshold on /health (got {value!r}): "
                               "set ESTIMATOR_TAU")
        _SERVED[base] = float(value)
    return _SERVED[base]
