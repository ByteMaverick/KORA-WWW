"""Meaning-based similarity between a question and a relation name."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path

import numpy as np

from kora import ARTIFACT_ROOT

MODEL = "text-embedding-3-small"
BATCH = 500


def readable(name: str) -> str:
    """`FINANCE_CURRENCY` -> `finance currency`; `location/country` -> `location country`."""
    return name.replace("_", " ").replace("/", " ").replace(".", " ").lower()


class Embedder:
    """Vectors for strings, cached in memory and on disk by model."""

    def __init__(self, model: str | None = None, cache_dir: Path | None = None,
                 base_url: str | None = None, query_prefix: str | None = None):
        named = model or os.environ.get("KORA_EMBED_MODEL") or None
        self.base_url = base_url or os.environ.get("KORA_EMBED_BASE_URL") or None
        if self.base_url and not named:
            raise ValueError("KORA_EMBED_BASE_URL is set without KORA_EMBED_MODEL: the server's vectors "
                             f"would be cached as {MODEL}'s")
        self.model = named or MODEL
        self.query_prefix = (os.environ.get("KORA_EMBED_QUERY_PREFIX", "")
                             if query_prefix is None else query_prefix)
        self.cache_dir = cache_dir or Path(os.environ.get("KORA_EMBED_CACHE_DIR") or (ARTIFACT_ROOT / "embeddings"))
        self._vectors: dict[str, np.ndarray] = {}
        self._pending: list[str] = []
        self._client = None
        self.calls = 0
        self.tokens = 0
        self._lock = threading.Lock()
        self._load()

    def _stem(self) -> str:
        return self.model.replace("/", "--")

    def _path(self) -> Path:
        return self.cache_dir / f"{self._stem()}.json"

    def _load(self) -> None:
        path = self._path()
        if not path.exists():
            return
        try:
            raw = json.loads(path.read_text() or "{}")
        except (json.JSONDecodeError, OSError):
            raw = {}
        if not isinstance(raw, dict):
            raw = {}
        self._vectors = {k: np.asarray(v, dtype=np.float32) for k, v in raw.items()}
        added = self._log_path()
        if added.exists():
            try:
                for line in added.read_text().splitlines():
                    try:
                        entry = json.loads(line)
                        self._vectors[entry["t"]] = np.asarray(entry["v"], dtype=np.float32)
                    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                        continue
            except OSError:
                pass

    def _log_path(self) -> Path:
        return self.cache_dir / f"{self._stem()}.jsonl"

    def flush(self) -> None:
        with self._lock:
            pending, self._pending = self._pending, []
            if not pending:
                return
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            with open(self._log_path(), "a", encoding="utf-8") as handle:
                for text in pending:
                    vector = self._vectors.get(text)
                    if vector is not None:
                        handle.write(json.dumps({"t": text, "v": vector.tolist()}) + "\n")

    def save(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        target = self._path()
        temporary = target.with_suffix(f".{os.getpid()}.tmp")
        temporary.write_text(json.dumps({k: v.tolist() for k, v in self._vectors.items()}))
        self._pending = []
        # Everything is in the compacted file now; the log would double-count.
        try:
            self._log_path().unlink(missing_ok=True)
        except OSError:
            pass
        os.replace(temporary, target)

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI

            key = "local" if self.base_url else os.environ.get("OPENAI_API_KEY")
            self._client = OpenAI(api_key=key, base_url=self.base_url, timeout=30.0, max_retries=2)
        return self._client

    def vectors(self, texts: list[str], persist: bool = True) -> np.ndarray:
        with self._lock:
            missing = [t for t in dict.fromkeys(texts) if t not in self._vectors]
            for start in range(0, len(missing), BATCH):
                chunk = missing[start : start + BATCH]
                response = self.client.embeddings.create(model=self.model, input=chunk)
                self.calls += 1
                self.tokens += int(getattr(getattr(response, "usage", None), "total_tokens", 0) or 0)
                for text, item in zip(chunk, response.data, strict=True):
                    vector = np.asarray(item.embedding, dtype=np.float32)
                    self._vectors[text] = vector / (np.linalg.norm(vector) or 1.0)
                    if persist:
                        self._pending.append(text)
            result = np.stack([self._vectors[t] for t in texts])
        # After the lock: flush takes it itself, and a Lock is not reentrant.
        if missing and persist:
            self.flush()
        return result

    def similarity(self, question: str, names: list[str]) -> list[float]:
        if not names:
            return []
        # The question in memory only; the relation names are worth keeping.
        wanted = self.vectors([self.query_prefix + question], persist=False)[0]
        matrix = self.vectors([readable(n) for n in names])
        return (matrix @ wanted).tolist()


def fingerprint(names: list[str]) -> str:
    """A short id for a set of names, so a cache can say which graph it is for."""
    return hashlib.sha256("\n".join(sorted(names)).encode()).hexdigest()[:12]
