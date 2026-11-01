"""The prompt registry."""

from __future__ import annotations

import hashlib
from contextvars import ContextVar
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from kora import ARTIFACT_ROOT
from kora.tools.llm import LLMError

BUILTIN_ROOT = Path(__file__).parent
SHIPPED = "prompts.yaml"

PROMPT_ROOTS = [BUILTIN_ROOT, ARTIFACT_ROOT / "prompts"]

RENDERED: ContextVar[str] = ContextVar("rendered", default="")


class Exemplar(BaseModel):
    model_config = ConfigDict(extra="forbid")

    shape: str
    question: str
    block: str


class Prompt(BaseModel):
    """One prompt: a named, versioned, hashable parameter set."""

    model_config = ConfigDict(extra="forbid")

    name: str
    version: str = "baseline"
    system: str = ""
    template: str = ""
    examples: list[Exemplar] = Field(default_factory=list)

    @property
    def id(self) -> str:
        return f"{self.name}.{self.version}"

    @property
    def hash(self) -> str:
        payload = f"{self.name}\x00{self.version}\x00{self.system}\x00{self.template}"
        if self.examples:
            payload += "\x00" + "\x00".join(f"{e.shape}\x00{e.question}\x00{e.block}" for e in self.examples)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def rendered(self) -> str:
        return "\n".join(f"QUESTION {e.question}\n{e.block.rstrip()}\n" for e in self.examples)

    def render(self, **values: object) -> str:
        try:
            text = self.template.format(**values)
        except KeyError as exc:
            raise LLMError(f"prompt {self.id} is missing variable {exc}") from None
        RENDERED.set(self.id)
        return text

    @classmethod
    def load(cls, path: str | Path, name: str | None = None) -> Prompt:
        path = Path(path)
        if not path.is_file():
            raise LLMError(f"prompt artifact not found: {path}")
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        data["name"] = name or path.parent.name
        data["version"] = path.stem
        data.pop("__unused__", None)
        return cls.model_validate(
            {k: v for k, v in data.items() if k in cls.model_fields}
        )

    @classmethod
    def section(cls, prompt_id: str, data: dict) -> Prompt:
        name, version = prompt_id.split(".", 1)
        data = {**(data or {}), "name": name, "version": version}
        return cls.model_validate({k: v for k, v in data.items() if k in cls.model_fields})


class PromptRegistry:
    """Every prompt Kora can use, addressed by `<module>.<version>`."""

    def __init__(self, roots: list[Path] | None = None):
        self._prompts: dict[str, Prompt] = {}
        for root in PROMPT_ROOTS if roots is None else roots:
            self.load_tree(root)

    def load_tree(self, root: str | Path) -> int:
        root = Path(root)
        if not root.is_dir():
            return 0
        found = 0
        shipped = root / SHIPPED
        if shipped.is_file():
            for prompt_id, data in (yaml.safe_load(shipped.read_text(encoding="utf-8")) or {}).items():
                self.register(Prompt.section(prompt_id, data))
                found += 1
        for path in sorted(root.glob("*/*.yaml")):
            self.register(Prompt.load(path, name=path.parent.name))
            found += 1
        return found

    def register(self, prompt: Prompt) -> Prompt:
        self._prompts[prompt.id] = prompt
        return prompt

    def get(self, prompt_id: str) -> Prompt:
        if prompt_id in self._prompts:
            return self._prompts[prompt_id]
        # A bare module name resolves to its baseline.
        baseline = f"{prompt_id}.baseline"
        if baseline in self._prompts:
            return self._prompts[baseline]
        raise LLMError(
            f"unknown prompt {prompt_id!r}; registered: {sorted(self._prompts)}"
        )

    def hashes(self) -> dict[str, str]:
        return {pid: p.hash for pid, p in sorted(self._prompts.items())}

    def __contains__(self, prompt_id: str) -> bool:
        return prompt_id in self._prompts

    def __len__(self) -> int:
        return len(self._prompts)


PROMPTS = PromptRegistry()
