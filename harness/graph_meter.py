"""Counts and logs every graph query a framework makes, whichever client it uses (README: the graph-call log)."""
from __future__ import annotations

import io
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.parse

_lock = threading.Lock()
_counts = {"queries": 0, "seconds": 0.0, "errors": 0}
_installed = False

_log_lock = threading.Lock()
_last_seq: dict[str, int] = {}          # log file -> the last seq written to it
_written = {"bytes": 0, "alerts": 0}    # bytes this process logged; multiples of the alert size warned of
_local = threading.local()              # .depth > 0 while a wrapped call runs on this thread


def reset() -> None:
    with _lock:
        _counts.update(queries=0, seconds=0.0, errors=0)


def snapshot() -> dict:
    with _lock:
        return {"graph_queries": _counts["queries"],
                "graph_seconds": round(_counts["seconds"], 3),
                "graph_errors": _counts["errors"]}


def _record(started: float, ok: bool) -> None:
    with _lock:
        _counts["queries"] += 1
        _counts["seconds"] += time.perf_counter() - started
        _counts["errors"] += 0 if ok else 1


def _log_dir() -> str:
    return "" if os.getenv("HARNESS_GRAPH_LOG", "") == "0" else os.getenv("HARNESS_TRACE_DIR", "")


def _qid() -> str:
    llm = sys.modules.get("harness.llm")     # harness.runner sets its TRACE_QID before each question
    return str(getattr(llm, "TRACE_QID", "") or "")


def _first_seq(path: str) -> int:
    """The lines a file already holds (0 if new); a last line cut short is ended first."""
    lines, last = 0, b"\n"
    try:
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                lines, last = lines + block.count(b"\n"), block[-1:]
    except FileNotFoundError:
        return 0
    if last != b"\n":
        with open(path, "ab") as f:
            f.write(b"\n")
        lines += 1
    return lines


def _alert(written: int) -> None:
    try:
        size = float(os.getenv("HARNESS_GRAPH_LOG_ALERT_GB", "") or 5) * 1e9
    except ValueError:
        size = 5e9
    if size > 0 and written // size > _written["alerts"]:
        _written["alerts"] = int(written // size)
        print(f"[graph_meter] warning: this process has logged {written / 1e9:.1f} GB of graph calls "
              f"(HARNESS_GRAPH_LOG_ALERT_GB={size / 1e9:g}; HARNESS_GRAPH_LOG=0 turns the log off)",
              file=sys.stderr, flush=True)


def _write(trace_dir: str, qid: str, fields: dict, body, charset) -> None:
    """Appends one call's record to its question's log. Never raises an Exception."""
    try:
        body = bytes(body) if isinstance(body, (bytes, bytearray)) else None
        text = None
        if body is not None:
            try:
                text = body.decode(charset or "utf-8", errors="replace")
            except LookupError:                 # a charset Python does not know
                text = body.decode("utf-8", errors="replace")
        fields.update(bytes=None if body is None else len(body), body=text)
        rest = json.dumps(fields, ensure_ascii=False, default=str)     # outside the lock: a body can be large
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in (qid or "no-qid"))   # as _trace_call
        path = os.path.join(trace_dir, f"{safe}.graph.jsonl")
        with _log_lock:
            os.makedirs(trace_dir, exist_ok=True)
            if path not in _last_seq:
                _last_seq[path] = _first_seq(path)
            _last_seq[path] += 1
            line = f'{{"qid": {json.dumps(qid, ensure_ascii=False)}, "seq": {_last_seq[path]}, {rest[1:]}\n'
            data = line.encode("utf-8", errors="replace")
            try:
                with open(path, "ab") as f:
                    f.write(data)
            except BaseException:       # the line may be cut short (the question's timeout): end it on the next write
                _last_seq.pop(path, None)
                raise
            _written["bytes"] += len(data)
            _alert(_written["bytes"])
    except Exception:
        pass


def _no_body(result):
    return result, None, None, None


def _metered(client: str, describe, call, ok_of=lambda result: True, capture=_no_body):
    """Runs call() once, counted, and with the log on, logged; a failed body read is the call's error."""
    if getattr(_local, "depth", 0):
        return call()                       # inside another wrapped call (SPARQLWrapper's urlopen): it records
    trace_dir, qid, ts = _log_dir(), _qid(), time.time()
    about = (None, None, None)
    if trace_dir:
        try:
            about = describe()
        except Exception:
            pass
    started = time.perf_counter()
    _local.depth = 1
    try:
        result = call()
        status, body, charset = None, None, None
        if trace_dir:
            result, status, body, charset = capture(result)
    except Exception as exc:
        _record(started, False)
        if trace_dir:
            _write(trace_dir, qid, _fields(ts, client, about, _error_status(exc), started, repr(exc)), None, None)
        raise
    finally:
        _local.depth = 0
    _record(started, ok_of(result))
    if trace_dir:
        _write(trace_dir, qid, _fields(ts, client, about, status, started, None), body, charset)
    return result


def _fields(ts, client, about, status, started, error) -> dict:
    endpoint, method, query = about
    return {"ts": ts, "client": client, "endpoint": endpoint, "method": method, "query": query,
            "status": status, "seconds": round(time.perf_counter() - started, 6), "error": error}


def _error_status(exc: BaseException):
    for e in (exc, exc.__cause__, exc.__context__):    # SPARQLWrapper raises its own error from the HTTPError
        if isinstance(e, urllib.error.HTTPError):
            return e.code
    return None


def _endpoint(url) -> str:
    parts = urllib.parse.urlsplit(str(url))
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def _query_in(value, raw: bool = False):
    """The SPARQL in one part of a request (its "query", else "update"; raw: the text); never reads a file."""
    if isinstance(value, dict):
        found = value.get("query", value.get("update"))
    elif isinstance(value, (list, tuple)):
        pairs = dict(p for p in value if isinstance(p, (list, tuple)) and len(p) == 2)
        found = pairs.get("query", pairs.get("update"))
    elif isinstance(value, (bytes, bytearray, str)):
        text = value.decode("utf-8", errors="replace") if isinstance(value, (bytes, bytearray)) else value
        form = urllib.parse.parse_qs(text, keep_blank_values=True)
        found = form.get("query", form.get("update"))
        if found is None and raw:
            return text or None
    else:
        return None
    if isinstance(found, (list, tuple)):
        found = found[0] if found else None
    if isinstance(found, (bytes, bytearray)):
        found = found.decode("utf-8", errors="replace")
    return found


def _first_query(*parts):
    for value, raw in parts:
        found = _query_in(value, raw)
        if found:
            return found
    return None


def _status(response):
    try:
        status = getattr(response, "status", None)
        return response.getcode() if status is None and hasattr(response, "getcode") else status
    except Exception:
        return None


def _charset(response):
    try:
        return response.headers.get_content_charset()
    except Exception:
        return None


class _Buffered(io.BytesIO):
    """A read urllib response: reads, iteration and `with` from the buffer, all else from the original."""

    def __init__(self, original, body: bytes):
        self._original = original
        super().__init__(body)

    def close(self) -> None:
        super().close()
        self._original.close()

    def __getattr__(self, name):
        if name == "_original":
            raise AttributeError(name)
        return getattr(self._original, name)


def _read_whole(response) -> bytes:
    try:
        return response.read()
    except BaseException:
        try:
            response.close()
        except Exception:
            pass
        raise


def _buffer_urllib(response):
    body = _read_whole(response)
    return _Buffered(response, body), _status(response), body, _charset(response)


def _buffer_sparqlwrapper(result):
    response = getattr(result, "response", None)
    if response is None or not hasattr(response, "read"):
        return result, None, None, None
    body = _read_whole(response)
    result.response = _Buffered(response, body)
    return result, _status(response), body, _charset(response)


_REQUESTS_ARGS = ("params", "data", "headers", "cookies", "files", "auth", "timeout", "allow_redirects",
                  "proxies", "hooks", "stream", "verify", "cert", "json")    # Session.request's, after the url


def install() -> None:
    global _installed
    if _installed:
        return
    _installed = True
    try:
        import requests
        original = requests.sessions.Session.request

        def request(self, method, url, *args, **kwargs):
            if "sparql" not in str(url).lower():
                return original(self, method, url, *args, **kwargs)
            sent = {**dict(zip(_REQUESTS_ARGS, args)), **kwargs}

            def describe():
                return (_endpoint(url), str(method).upper(),
                        _first_query((sent.get("params"), False), (sent.get("data"), True),
                                     (sent.get("json"), False), (urllib.parse.urlsplit(str(url)).query, False)))

            def capture(response):
                status = getattr(response, "status_code", None)
                if sent.get("stream"):
                    return response, status, None, None
                return response, status, getattr(response, "content", None), getattr(response, "encoding", None)

            return _metered("requests", describe, lambda: original(self, method, url, *args, **kwargs),
                            lambda r: getattr(r, "status_code", 200) < 400, capture)

        requests.sessions.Session.request = request
    except ImportError:
        pass

    import urllib.request
    original_urlopen = urllib.request.urlopen

    def urlopen(url, *args, **kwargs):
        full_url = url.full_url if isinstance(url, urllib.request.Request) else str(url)
        if "sparql" not in full_url.lower():
            return original_urlopen(url, *args, **kwargs)
        data = kwargs.get("data", args[0] if args else None)

        def describe():
            body = data if data is not None else getattr(url, "data", None)
            method = getattr(url, "method", None) or ("POST" if body is not None else "GET")
            return (_endpoint(full_url), method.upper(),
                    _first_query((body, True), (urllib.parse.urlsplit(full_url).query, False)))

        return _metered("urllib", describe, lambda: original_urlopen(url, *args, **kwargs), capture=_buffer_urllib)

    urllib.request.urlopen = urlopen

    try:
        from SPARQLWrapper import SPARQLWrapper
        original_query = SPARQLWrapper.query

        def query(self, *args, **kwargs):
            def describe():
                return _endpoint(self.endpoint), str(self.method).upper(), self.queryString

            return _metered("sparqlwrapper", describe, lambda: original_query(self, *args, **kwargs),
                            capture=_buffer_sparqlwrapper)

        SPARQLWrapper.query = query
    except ImportError:
        pass
    try:
        import neo4j
        original_run = neo4j.Session.run

        def run(self, query, parameters=None, **kwargs):
            def describe():
                address = getattr(getattr(self, "_pool", None), "address", None)
                return (None if address is None else str(address), "RUN",
                        {"text": getattr(query, "text", query), "parameters": {**(parameters or {}), **kwargs}})

            return _metered("neo4j", describe, lambda: original_run(self, query, parameters, **kwargs))

        neo4j.Session.run = run
    except (ImportError, AttributeError):
        pass
