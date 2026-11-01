"""Graph backends. The original exports a Freebase and a Wikidata backend and builds both when imported; the harness
always passes its Freebase backend in (baselines/cor/run.py), so there is no default."""
from __future__ import annotations

from baselines.cor.chain_of_relations.kg_backend.base import KGBackend


def get_default_backend() -> KGBackend:
	raise RuntimeError("CoR in this harness is always given its backend (baselines/cor/run.py)")


__all__ = ["KGBackend", "get_default_backend"]
