"""Kora: requirement-aware, budget-aware knowledge graph question answering."""

from __future__ import annotations

from pathlib import Path

__version__ = "0.1.0"

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT: Path = PROJECT_ROOT / "artifacts"

__all__ = [
    "ARTIFACT_ROOT",
    "PROJECT_ROOT",
    "__version__",
]
