"""The free-text answer extractor: a fixed model lists a prose answer's items (README: the extractor)."""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPT_FILE = os.path.join(HERE, "extract_prompt.md")
CACHE_NAME = "extract.jsonl"

MAX_TOKENS = 1024
PER_SERVER = 16
TRIES, FIRST_WAIT, TIMEOUT_S = 3, 2.0, 120

_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")


def load_prompt(path: str = PROMPT_FILE) -> tuple:
    """(system body, user body) from the two markers of the prompt artifact."""
    text = open(path, encoding="utf-8").read()
    head, rest = text.split("<!-- SYSTEM -->", 1)
    system, user = rest.split("<!-- USER -->", 1)
    return system.strip(), user.strip()


def prompt_hash(path: str = PROMPT_FILE) -> str:
    system, user = load_prompt(path)
    return hashlib.sha256((system + "\n<!-- USER -->\n" + user).encode("utf-8")).hexdigest()


def build_messages(question: str, response: str, prompt: tuple | None = None) -> list:
    """The chat messages. Only the question and the response text enter; there is no other argument to put a gold in."""
    system, user = prompt or load_prompt()
    fields = {"question": question, "response": response}
    filled = re.sub(r"\{(question|response)\}", lambda m: fields[m.group(1)], user)   # one pass: no re-substitution
    return [{"role": "system", "content": system}, {"role": "user", "content": filled}]


def messages_for_row(row: dict, prompt: tuple | None = None) -> list:
    """The messages for a stored result row: its `question` and its `answer` text, no other field is read."""
    return build_messages(str(row.get("question") or ""), str(row.get("answer") or ""), prompt)


def verbatim_split(reply: str, response: str) -> tuple:
    """(kept, rejected) from the reply: one item per line, kept only if a substring of the response."""
    kept, rejected = [], []
    for line in (reply or "").splitlines():
        item = line.strip()
        if not item:
            continue
        if item not in response:
            stripped = _BULLET.sub("", line).strip()
            if stripped and stripped in response:
                item = stripped
        target = kept if item in response else rejected
        if item not in target:
            target.append(item)
    return kept, rejected


def entry(qid: str, phash: str, model: str, response: str, reply: str | None) -> dict:
    """One cache object. `reply` None = the call failed (status "error")."""
    if reply is None:
        return {"qid": qid, "prompt_hash": phash, "extractor_model": model, "items": [], "rejected": [], "status": "error"}
    kept, rejected = verbatim_split(reply, response)
    return {"qid": qid, "prompt_hash": phash, "extractor_model": model, "items": kept, "rejected": rejected,
            "status": "ok" if kept else "empty"}


class Transient(Exception):
    pass


def call_model(base_url: str, model: str, messages: list) -> str:
    """One chat completion at temperature 0, thinking off; Transient on a retryable failure, else ValueError."""
    body = json.dumps({"model": model, "messages": messages, "temperature": 0, "max_tokens": MAX_TOKENS,
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
            d = json.load(r)
    except urllib.error.HTTPError as e:
        if e.code == 429 or e.code >= 500:
            raise Transient(f"http {e.code}")
        raise ValueError(f"http {e.code}")
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
        raise Transient(type(e).__name__)
    except ValueError:                            # a body that is not JSON
        raise Transient("bad body")
    try:
        text = d["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise ValueError("no content")
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL)


class Servers:
    """Bounded in-flight requests per server; each call goes to the least loaded server below its cap."""

    def __init__(self, urls, cap: int = PER_SERVER):
        self.urls, self.max_cap = list(urls), cap
        self.cap = {u: cap for u in self.urls}
        self.busy = {u: 0 for u in self.urls}
        self.cv = threading.Condition()

    def acquire(self) -> str:
        with self.cv:
            while True:
                free = [u for u in self.urls if self.busy[u] < self.cap[u]]
                if free:
                    u = min(free, key=lambda x: self.busy[x])
                    self.busy[u] += 1
                    return u
                self.cv.wait()

    def release(self, u: str) -> None:
        with self.cv:
            self.busy[u] -= 1
            self.cv.notify_all()

    def set_cap(self, u: str, cap: int) -> None:
        with self.cv:
            self.cap[u] = max(2, min(self.max_cap, cap))
            self.cv.notify_all()


def extract_one(servers: Servers, model: str, messages: list, call: Callable = call_model, sleep=time.sleep) -> str | None:
    """The reply, or None after TRIES transient failures or one non-transient failure (never raises)."""
    wait = FIRST_WAIT
    for attempt in range(TRIES):
        url = servers.acquire()
        try:
            return call(url, model, messages)
        except Transient:
            pass
        except ValueError:
            return None
        finally:
            servers.release(url)
        if attempt < TRIES - 1:
            sleep(wait)
            wait *= 2
    return None


def load_cache(path: str, phash: str, model: str) -> dict:
    """{qid: entry} of the standing lines; the file is rewritten without the others."""
    if not os.path.exists(path):
        return {}
    keep, total = {}, 0
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        total += 1
        try:
            e = json.loads(line)
        except ValueError:
            continue                                           # a torn last line: the row is extracted again
        if e.get("prompt_hash") == phash and e.get("extractor_model") == model and e.get("status") in ("ok", "empty"):
            keep[str(e["qid"])] = e
    if total != len(keep):
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for e in keep.values():
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        os.replace(tmp, path)
    return keep


class Writer:
    def __init__(self):
        self.lock = threading.Lock()

    def append(self, path: str, obj: dict) -> None:
        with self.lock:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def run_pass(rows: list, path: str, servers: Servers, model: str, *, workers: int | None = None,
             call: Callable = call_model, sleep=time.sleep) -> Counter:
    """Extract every row of one run that has no standing cache line in `path`. Returns the counts."""
    prompt = load_prompt()
    phash = prompt_hash()
    counts, lock, writer = Counter(), threading.Lock(), Writer()
    have = load_cache(path, phash, model)

    def task(row):
        qid = str(row.get("qid"))
        response = str(row.get("answer") or "")
        if not response.strip():
            e = entry(qid, phash, model, response, "")          # no response, no call
        else:
            e = entry(qid, phash, model, response, extract_one(servers, model, messages_for_row(row, prompt), call, sleep))
        writer.append(path, e)
        with lock:
            counts[{"ok": "new_ok", "empty": "new_empty", "error": "new_error"}[e["status"]]] += 1
            counts["rejected_items"] += len(e["rejected"])

    todo = []
    for row in rows:
        counts["rows_seen"] += 1
        if str(row.get("qid")) in have:
            counts["cached"] += 1
        elif isinstance((row.get("fw_metrics") or {}).get("answer_items"), list):
            counts["list_rows_skipped"] += 1      # the system returned a list: not free text, nothing to extract
        else:
            todo.append(row)
    with ThreadPoolExecutor(max_workers=workers or PER_SERVER * len(servers.urls)) as pool:
        list(pool.map(task, todo))
    return counts
