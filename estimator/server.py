"""The Stopping Estimator's server. A client posts states with `extras` ({ids, n, query, chain}); each becomes the
(question, evidence) pair with `estimator.render`, the renderer the training rows were made with, with names looked up
from the graph. A state the renderer refuses (or one without extras) scores 0.0: keep searching. Two-class models give
P(yes) = softmax[1]; one-logit models sigmoid.

    python -m estimator.server --checkpoint DIR [--port 8100] [--log decisions.jsonl] [--device cuda] [--batch 32]

DIR holds `model/`, `tokenizer/` and `threshold.json` ({"tau": ...}); `ESTIMATOR_CHECKPOINT` stands in for
--checkpoint. POST /score {"states", "questions", "extras"} -> {"p": [...], "ms"}; GET /health -> {"ok", "model",
"threshold", counters}. Freebase ids are named from `FREEBASE_SPARQL_URL` (for the Unseen sets, the extended graph,
so a cloned entity is named from the graph that holds it); MetaQA ids are their own names.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from estimator.render import from_kora, ids_in, ids_to_name, joined, validate

MAX_LEN = 7999
FREEBASE_ID = re.compile(r"[mg]\.[0-9a-z_]+")
WARM_UP = ("warm up", "EVIDENCE (facts retrieved so far, as subject | relation | object)\n  A | r | B\n\n"
                      "CANDIDATE ANSWER (1 items)\nB")


def load(checkpoint: str, device: str = "cuda", batch: int = 32):
    """(score, tau) for a checkpoint folder: score(pairs) -> P(yes) per (question, evidence) pair."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    info = json.load(open(f"{checkpoint}/threshold.json"))
    max_len = int(info.get("max_len", MAX_LEN))
    tok = AutoTokenizer.from_pretrained(f"{checkpoint}/tokenizer")
    model = AutoModelForSequenceClassification.from_pretrained(f"{checkpoint}/model").to(device).eval()
    gpu = threading.Lock()

    @torch.no_grad()
    def score(pairs):
        out = []
        with gpu:
            for i in range(0, len(pairs), batch):
                qs = [q for q, _ in pairs[i:i + batch]]; evs = [e for _, e in pairs[i:i + batch]]
                enc = tok(qs, evs, truncation=True, max_length=max_len, padding=True, return_tensors="pt").to(device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(**enc).logits.float()
                p = torch.softmax(logits, -1)[:, 1] if logits.shape[-1] == 2 else torch.sigmoid(logits.squeeze(-1))
                out += p.cpu().tolist()
        return [float(x) for x in out]

    return score, info.get("tau")


def namer(freebase_url: str | None):
    """names_for(ids) -> {id: name} for the ids the graph can name, cached."""
    fb = None
    if freebase_url:
        from kora.tools.sparql import FreebaseSPARQL
        fb = FreebaseSPARQL(freebase_url, timeout=30)
    cache, cache_lock = {}, threading.Lock()

    def names_for(ids):
        with cache_lock:
            want = [i for i in ids if i not in cache]
        fbids = [i for i in want if FREEBASE_ID.fullmatch(i)]
        found = {}
        if fbids and fb is not None:
            try:
                found.update({k: v for k, v in fb.labels(fbids).items() if v})
            except Exception as e:  # noqa: BLE001 - an unnamed id stays an id
                print(f"[names] {e!r}", flush=True)
        with cache_lock:
            cache.update(found)
            return {i: cache[i] for i in ids if i in cache}

    return names_for


def serve(score, names_for, tau, port: int = 8100, host: str = "0.0.0.0", log: str | None = None,
          model: str = "") -> ThreadingHTTPServer:
    """The HTTP server, bound and ready; the caller runs `serve_forever`."""
    logfh = open(log, "a", encoding="utf-8") if log else None
    log_lock = threading.Lock()
    stats = {"requests": 0, "states": 0, "rewritten": 0, "refused": 0, "started": time.time(), "threshold": tau}

    def pair_of(state, question, extra):
        """A client's state -> (question, evidence) with the training renderer; None if refused."""
        if not extra:
            return None
        ids = [str(i) for i in extra.get("ids") or []]; query = extra.get("query"); chain = extra.get("chain") or []
        need = ids_to_name(state, ids) + ids_in(query)
        for link in chain:
            need += list(link.get("ids") or [])[:30] + ids_in(link.get("query"))
        names = names_for(list(dict.fromkeys(need)))
        q, ev = from_kora(question, state, ids, extra.get("n") or len(ids), query, names, chain)
        if validate(q, ev):
            return None
        stats["rewritten"] += 1; stats["chained"] = stats.get("chained", 0) + bool(chain)
        return q, ev

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass

        def _send(self, code, obj):
            body = json.dumps(obj).encode(); self.send_response(code); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

        def do_GET(self): self._send(200, {"ok": True, "model": model, **stats})

        def do_POST(self):
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                states = req["states"]; questions = req.get("questions") or [None] * len(states)
                extras = req.get("extras") or [None] * len(states)
                t = time.time(); pairs = [pair_of(s, q, x) for s, q, x in zip(states, questions, extras)]
                ok = [k for k, pr in enumerate(pairs) if pr is not None]
                scored = score([pairs[k] for k in ok]) if ok else []; p = [0.0] * len(pairs)
                for k, v in zip(ok, scored):
                    p[k] = v
                stats["refused"] += len(pairs) - len(ok)
                texts = [joined(*pr) if pr else None for pr in pairs]
                stats["requests"] += 1; stats["states"] += len(states)
                if logfh is not None:
                    with log_lock:
                        for s, q, x, tx, pr in zip(states, questions, extras, texts, p):
                            logfh.write(json.dumps({"t": round(time.time(), 2), "p": round(pr, 4), "refused": tx is None,
                                                    "question": q, "state": s, "extra": x, "text": tx},
                                                   ensure_ascii=False) + "\n")
                        logfh.flush()
                self._send(200, {"p": p, "ms": round((time.time() - t) * 1000, 1)})
            except Exception as e:  # noqa: BLE001
                self._send(500, {"error": repr(e)})

    return ThreadingHTTPServer((host, port), Handler)


def main(argv=None) -> int:
    from kora.app import load_dotenv
    load_dotenv()
    ap = argparse.ArgumentParser(prog="python -m estimator.server")
    ap.add_argument("--checkpoint", default=os.environ.get("ESTIMATOR_CHECKPOINT"),
                    help="folder with model/, tokenizer/ and threshold.json (default: ESTIMATOR_CHECKPOINT)")
    ap.add_argument("--port", type=int, default=8100); ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--log", help="append every scored state to this JSONL file")
    ap.add_argument("--device", default="cuda"); ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--freebase-url", default=os.environ.get("FREEBASE_SPARQL_URL"),
                    help="SPARQL endpoint that names Freebase ids (default: FREEBASE_SPARQL_URL)")
    a = ap.parse_args(argv)
    if not a.checkpoint:
        ap.error("no checkpoint: pass --checkpoint DIR or set ESTIMATOR_CHECKPOINT")
    if not os.path.isfile(os.path.join(a.checkpoint, "threshold.json")):
        ap.error(f"{a.checkpoint} has no threshold.json: the folder needs model/, tokenizer/ and threshold.json")
    if not a.freebase_url:
        print("[estimator] FREEBASE_SPARQL_URL is unset: Freebase ids are shown unnamed", flush=True)
    score, tau = load(a.checkpoint, a.device, a.batch)
    print(score([WARM_UP]), flush=True)
    srv = serve(score, namer(a.freebase_url), tau, a.port, a.host, a.log, a.checkpoint)
    print(f"[estimator] serving {a.checkpoint} (tau {tau}) on port {srv.server_port}", flush=True)
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
