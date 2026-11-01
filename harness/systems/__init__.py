"""The systems the harness runs, the stop rules each takes, and each one's handler (README: systems)."""
from __future__ import annotations

import importlib

STOPS: dict[str, tuple] = {
    "kora": ("estimator", "llm", "gold"),
    "tog": ("native", "estimator", "gold"),
    "pog": ("native", "estimator", "gold"),
    "cor": ("native", "estimator", "gold"),
    "sog": ("native",),
    "kbbinder": ("native",),
    "direct_llm": ("native",),
}
SYSTEMS = tuple(STOPS)
BASELINES = ("tog", "pog", "cor", "sog", "kbbinder")
#: The framework id and entry point of a baseline the paper's harness named otherwise (its settings-table rows).
OFFICIAL = {"kbbinder": "kbbinder_official"}


class UnsupportedStop(ValueError):
    """A system asked to stop by a rule it does not take."""


def default_stop(system: str) -> str:
    return STOPS[system][0]


def framework_of(system: str, stop: str | None = None) -> str:
    """The framework id of a system under a stop rule; refuses a combination the system does not take."""
    if system not in STOPS:
        raise ValueError(f"unknown system {system!r}: one of {', '.join(SYSTEMS)}")
    stop = stop or default_stop(system)
    if stop not in STOPS[system]:
        raise UnsupportedStop(f"{system} does not take --stop {stop}: it takes {', '.join(STOPS[system])}")
    if system in ("kora", "direct_llm"):
        return system
    return OFFICIAL.get(system) or f"{system}_{stop}"


def handler_of(framework: str):
    """The handler that runs one question of this framework."""
    if framework == "direct_llm":
        from harness.systems.direct_llm import run_direct_llm
        return run_direct_llm
    if framework == "kora":
        from harness.systems.kora import run
        return run
    family = framework.split("_", 1)[0]
    if family not in BASELINES:
        raise ValueError(f"unknown framework {framework!r}")
    return _baseline(family)


def _baseline(family: str):
    def run(question, args, sample=None):
        if sample is None:
            raise NotImplementedError(f"[{family}] runs in dataset mode only")
        try:
            module = importlib.import_module(f"baselines.{family}.run")
        except ModuleNotFoundError as e:
            if e.name and (e.name.startswith(f"baselines.{family}") or e.name == "baselines"):
                raise NotImplementedError(f"[{family}] the baseline package baselines/{family} is not installed") from e
            raise
        return getattr(module, f"run_{OFFICIAL.get(family, family)}")(sample=sample, args=args)
    return run
