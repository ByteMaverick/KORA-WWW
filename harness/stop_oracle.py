"""The gold-aware stop of the gold-stop arms (`--stop gold`), one rule for every system (README: the stop switch)."""
from __future__ import annotations

import json
import re

_WORDS = re.compile(r"[a-z0-9]+")
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
COUNT_WORDS = re.compile(r"\b(how many|number of|count of)\b", re.IGNORECASE)
PRECISION, RECALL, TOP = 0.5, 0.5, 50


def matches(gold: str, candidate: str) -> bool:
    """KORA's matcher (kora/modules/sufficiency/sufficiency.py, ``matches``), copied unchanged."""
    first, second = _WORDS.findall(str(gold).lower()), _WORDS.findall(str(candidate).lower())
    if not first or not second:
        return False
    if all(word.isdigit() for word in first):
        short, long = (first, second) if len(first) <= len(second) else (second, first)
        return long[:len(short)] == short
    short, long = (first, second) if len(first) <= len(second) else (second, first)
    return any(long[i:i + len(short)] == short for i in range(len(long) - len(short) + 1))


def _path_names(path) -> list[str]:
    """Entity names in an agent's path: triples, 'a --rel--> b' chains, SoG table rows, CoR's TargetEntity."""
    names: list[str] = []
    for p in path or []:
        if isinstance(p, (list, tuple)) and len(p) >= 3 and not isinstance(p[0], (list, tuple)):
            names += [str(p[0]), str(p[-1])]
        elif isinstance(p, (list, tuple)):
            names += _path_names(p)
        elif isinstance(p, str):
            for chain in re.findall(r"[^\n]*-->[^\n]*", p):
                names += [x.strip() for x in re.split(r"--[^>]*-->", chain) if x.strip()]
            names += [x.strip() for x in re.findall(r"^\|\s*[^|\n]+\|\s*([^|\n]+?)\s*\|", p, re.M)]
            for inner in re.findall(r"TargetEntity\(([^)]*)\)", p):
                names += [x.strip() for x in inner.split(",") if x.strip()]
    seen, out = set(), []
    for n in names:
        key = n.lower()
        if n and key not in seen and not set(n) <= set("|-: ") and key not in ("neighbor name", "neighbor id"):
            seen.add(key)
            out.append(n)
    return out


def _mentioned(name: str, text: str) -> int | None:
    """Where the name's words first appear, in order and adjacent, in the text (word index), or None."""
    a, b = _WORDS.findall(name.lower()), _WORDS.findall(text.lower())
    if not a:
        return None
    for i in range(len(b) - len(a) + 1):
        if b[i:i + len(a)] == a:
            return i
    return None


def answer_items(answer: str, path=None) -> tuple[list[str], str]:
    """The answer's items and how they were found."""
    text = str(answer or "").strip()
    if not text:
        return [], "empty"
    m = re.search(r"\{.*\}", text, re.S)   # a JSON object with an "answer" list (CoR) or PoG's {"A": {...}}
    if m:
        try:
            v = json.loads(m.group(0))
            if isinstance(v, dict):
                for key in ("answer", "Answer"):
                    if isinstance(v.get(key), list):
                        return [str(x).strip() for x in v[key] if str(x).strip()], "json"
                if isinstance(v.get("A"), dict) and v["A"].get("Answer"):
                    text = str(v["A"]["Answer"])
        except Exception:
            pass
    braces = [b for b in re.findall(r"\{([^{}:]{1,2000})\}", text) if b.strip()]
    if braces:   # ToG writes {answer}
        return [x.strip() for b in braces for x in re.split(r"[,;\n]", b) if x.strip()], "braces"
    if "Final answer:" in text:
        text = text.split("Final answer:", 1)[1].strip()
    pieces = [x.strip(" .*-") for x in re.split(r"[,;\n]", text) if x.strip(" .*-")]
    if pieces and all(len(x.split()) <= 8 for x in pieces):
        return pieces, "list"
    names = _path_names(path)   # a sentence: the graph names it mentions, in order of mention
    found = sorted((pos, n) for n in names if (pos := _mentioned(n, text)) is not None)
    if found:
        return [n for _, n in found], "names_in_text"
    return [text], "text"


def score(answer: str, golds: list[str], question: str = "", question_type: str | None = None,
          path=None) -> dict:
    golds = [str(g) for g in golds if str(g).strip()]
    items, how = answer_items(answer, path)
    out = {"items": items, "items_from": how}
    if not golds or not items:
        return {**out, "precision": 0.0, "recall": 0.0, "f1": 0.0, "accept": False}
    if len(golds) == 1 and re.fullmatch(r"\d+", golds[0].strip()) and COUNT_WORDS.search(question or ""):
        nums = _NUMBER.findall(re.sub(r"(?<=\d),(?=\d{3})", "", str(answer)))
        ok = bool(nums) and float(nums[0]) == float(golds[0])
        return {**out, "precision": float(ok), "recall": float(ok), "f1": float(ok), "accept": ok, "rule": "count"}
    found = sum(1 for g in golds if any(matches(g, x) for x in items))
    top = items[:TOP]
    right = sum(1 for x in top if any(matches(g, x) for g in golds))
    r, p = found / len(golds), right / len(top)
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {**out, "precision": round(p, 3), "recall": round(r, 3), "f1": round(f1, 3),
            "accept": p >= PRECISION and r >= RECALL}


def _plain(path):
    """The path as JSON-safe strings."""
    if path is None:
        return None
    if isinstance(path, str):
        return [path]
    out = []
    for p in path:
        if isinstance(p, (list, tuple)) and p and all(isinstance(x, (list, tuple)) for x in p):
            out += [" | ".join(map(str, x)) for x in p]
        elif isinstance(p, (list, tuple)):
            out.append(" | ".join(map(str, p)))
        else:
            out.append(str(p))
    return out


class StopOracle:
    def __init__(self, golds: list[str], question: str = "", question_type: str | None = None):
        self.golds = [str(g) for g in golds if str(g).strip()]
        self.question, self.question_type = question, question_type
        self.log: list[dict] = []

    def right(self, step, judge_stop: bool, answer: str, path=None) -> bool:
        """Log one decision point and say whether the answer written there is good enough to stop on."""
        from harness.metrics.reasoning import score_text
        s = score(answer, self.golds, self.question, self.question_type, path)
        hit = score_text(answer or "", self.golds)[0]
        self.log.append({"step": step, "judge_stop": bool(judge_stop), "right": bool(s["accept"]),
                         "accept": bool(s["accept"]), "hit": bool(hit), "precision": s["precision"],
                         "recall": s["recall"], "f1": s["f1"], "rule": s.get("rule", "precision/recall"),
                         "items_from": s["items_from"], "items": s["items"][:200], "answer": answer or "",
                         "path": _plain(path)})
        return bool(s["accept"])
