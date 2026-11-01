"""The relation-ranking embedder: an OpenAI-compatible /v1/embeddings server."""
from __future__ import annotations

import argparse
import base64
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np


def serve(model_path: str, name: str, port: int, batch: int) -> None:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_path, device="cpu")
    served = {name, model_path, os.path.basename(model_path.rstrip("/"))}
    lock = threading.Lock()   # one encode at a time: the model is not shared safely across threads

    class Handler(BaseHTTPRequestHandler):
        def _reply(self, code: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):  # noqa: N802
            if self.path.rstrip("/") == "/v1/models":
                return self._reply(200, {"object": "list", "data": [{"id": name, "object": "model"}]})
            return self._reply(404, {"error": "not found"})

        def do_POST(self):  # noqa: N802
            if self.path.rstrip("/") != "/v1/embeddings":
                return self._reply(404, {"error": "not found"})
            request = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            if request.get("model") not in served:
                return self._reply(404, {"error": f"this server serves {name}, not {request.get('model')}"})
            texts = request.get("input") or []
            texts = [texts] if isinstance(texts, str) else texts
            with lock:
                vectors = np.asarray(model.encode(texts, batch_size=batch, normalize_embeddings=True,
                                                  convert_to_numpy=True), dtype=np.float32)
                count = sum(len(model.tokenizer.encode(t)) for t in texts)
            base64_wanted = request.get("encoding_format") == "base64"
            data = [{"object": "embedding", "index": i,
                     "embedding": base64.b64encode(v.tobytes()).decode() if base64_wanted else v.tolist()}
                    for i, v in enumerate(vectors)]
            return self._reply(200, {"object": "list", "model": name, "data": data,
                                     "usage": {"prompt_tokens": count, "total_tokens": count}})

        def log_message(self, *args):  # quiet: one line per request would bury the worker log
            return

    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m kora.embed_server")
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5", help="a model id or a local folder")
    parser.add_argument("--name", default="BAAI/bge-small-en-v1.5", help="the id clients ask for (KORA_EMBED_MODEL)")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()
    serve(args.model, args.name, args.port, args.batch)


if __name__ == "__main__":
    main()
