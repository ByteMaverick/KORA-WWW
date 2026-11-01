"""Our question sets in GCR's input format (the rmanluo/RoG-* schema), each with its subgraph, no gold."""
from __future__ import annotations

import argparse
import collections
import json
import os
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from harness.dataset import UNSEEN, records as question_records
from harness.subgraph_topics import topic_nodes

MAX_TRIPLES = 2000
LABEL = "http://www.w3.org/2000/01/rdf-schema#label"
GRAPH = os.environ.get("CF_BIOKG_GRAPH", "http://biokg.org/")
BASE = "http://skill-lab-research.org/"
ROG_SETS = ("cwq", "webqsp")   # Freebase sets with RoG's released subgraphs
#: The Unseen halves on Freebase (the twins and their real sources), both exported live from the extended graph.
UNSEEN_FREEBASE = tuple(u for u in UNSEEN if u.startswith(("webqsp", "cwq")))
LIVE_FREEBASE = ("grailqa", *UNSEEN_FREEBASE)   # Freebase sets exported live from the endpoint
#: The question file a set's split is read from, where it is not <split>.json (none here: GrailQA's test.json
#: holds the 1,000 every system runs)
QUESTION_FILE: dict = {}
FB = "http://rdf.freebase.com/ns/"
KG_OF_PREFIX = {"PrimeKG": "primekg", "OptimusKG": "optimuskg", "AfriMedKG": "afrimedkg", "KQAPro": "kqapro"}
#: From the paper harness's FiDeLiS subgraph builder (FiDeLiS is not in this release): the Freebase predicates left
#: out, and how a slow question is tried again.
SKIP = ("type.object.name", "type.object.key", "common.topic.description", "common.topic.alias",
        "common.topic.topic_equivalent_webpage", "/key/", "wikipedia")
TRIES = int(os.getenv("SUBGRAPH_TRIES", "2"))                 # a question's queries, tried this many times
PAUSE = float(os.getenv("SUBGRAPH_RETRY_PAUSE", "30"))       # seconds between two tries
SLOW_IN_A_ROW = 5     # questions in a row that time out: the graph is overloaded, not the questions; the build stops


def timed_out(error: BaseException) -> bool:
    """The graph took too long on a query (a hub): the question gets no subgraph. Not an outage, which fails."""
    text = f"{type(error).__name__}: {error}".lower()
    if "connect" in text:                                         # [Errno 110] Connection timed out, ConnectTimeout
        return False
    try:
        import requests
        if isinstance(error, requests.ReadTimeout):
            return True
        if isinstance(error, requests.RequestException):
            return any(m in text for m in ("sr171", "s1t00", "estimated execution time", "transaction timed out"))
    except ImportError:
        pass
    if isinstance(error, (TimeoutError, socket.timeout)):        # waiting for the answer (urllib wraps a connect one)
        return True
    return any(m in text for m in ("sr171", "s1t00", "estimated execution time", "transaction timed out", "s1tat"))


def with_retries(fn, *args, tries: int | None = None, pause: float | None = None):
    """fn(*args), tried again after a pause when it fails; the last try's error is raised."""
    tries, pause = tries or TRIES, PAUSE if pause is None else pause
    for k in range(tries):
        try:
            return fn(*args)
        except Exception as e:
            if k == tries - 1:
                raise
            print(f"  [subgraph] try {k + 1} of {tries} failed ({type(e).__name__}: {str(e)[:120]}); again in {pause:.0f} s")
            time.sleep(pause)


def _complete_last_line(path: str) -> None:
    """Cut a half-written last line, so a resumed export starts on a line of its own."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return
    with open(path, "rb+") as f:
        end = f.seek(0, os.SEEK_END)
        f.seek(end - 1)
        if f.read(1) == b"\n":
            return
        at = end
        while at > 0:                  # back to the last whole line, a block at a time (the file runs to gigabytes)
            start = max(0, at - (1 << 20))
            f.seek(start)
            cut = f.read(at - start).rfind(b"\n")
            if cut >= 0:
                f.truncate(start + cut + 1)
                return
            at = start
        f.truncate(0)


def _cap(one_hop: list, two_hop: list) -> tuple[list, bool]:
    seen, out = set(), []
    for t in one_hop + two_hop:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out[:MAX_TRIPLES], len(out) > MAX_TRIPLES


# ── MetaQA: a kb.txt in memory ──────────────────────────────────────────────────────────────────────────
def metaqa_records(questions: list[dict], kb: Path):
    adj = collections.defaultdict(list)
    for line in kb.open(encoding="utf-8"):
        parts = line.rstrip("\n").split("|")
        if len(parts) == 3 and all(parts):
            h, r, t = parts
            adj[h].append((h, r, t))
            adj[t].append((h, r, t))
    for q in questions:
        anchors = list(q["topic_entity"].values())
        one = [t for a in anchors for t in adj.get(a, [])]
        near = {x for h, _, t in one for x in (h, t)} - set(anchors)
        two = [t for n in sorted(near) for t in adj.get(n, [])]
        graph, capped = _cap(one, two)
        yield q, anchors, graph, capped


# ── BioCQ: the SPARQL endpoint of the real graph or of the copy ─────────────────────────────────────────
#: seconds a query may take before it counts as too slow (a hub); GCR_EXPORT_TIMEOUT, 300
TIMEOUT = float(os.getenv("GCR_EXPORT_TIMEOUT", "300"))


def _ask(endpoint: str, query: str) -> list[dict]:
    """One SELECT's bindings; a server error carries its text, and an anytime partial answer is refused."""
    url = endpoint + "?" + urllib.parse.urlencode({"query": query, "format": "application/sparql-results+json"})
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT) as r:
            state = r.headers.get("X-SQL-State")
            if state:
                raise TimeoutError(f"a partial answer (X-SQL-State {state}: {r.headers.get('X-SQL-Message', '')})")
            return json.load(r)["results"]["bindings"]
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")[:300] if e.fp else ""
        raise RuntimeError(f"HTTP {e.code} from {endpoint}: {body}") from e


def _rel(iri: str) -> str:
    return iri.rsplit("/relation/", 1)[-1]


def _named(v: str, literals: bool) -> str:
    """?v's name into ?{v}l: its label; with literals, a literal is its own name (KQA Pro's attribute values)."""
    if not literals:
        return f"?{v} <{LABEL}> ?{v}l ."
    return (f"OPTIONAL {{ ?{v} <{LABEL}> ?{v}_label . }} "
            f"BIND(IF(isLiteral(?{v}), STR(?{v}), ?{v}_label) AS ?{v}l) FILTER(BOUND(?{v}l))")


def _hops(endpoint: str, node: str, literals: bool = False) -> tuple[list, list]:
    """One- and two-hop triples around a node, by name; relations are the graph's own /relation/ predicates."""
    is_rel = 'FILTER(CONTAINS(STR(?{v}), "/relation/"))'
    one, two = [], []
    for direction in ("out", "in"):
        pat = f"<{node}> ?p ?m" if direction == "out" else f"?m ?p <{node}>"
        named_m = _named("m", literals and direction == "out")
        rows = _ask(endpoint, f"SELECT ?p ?m ?ml WHERE {{ GRAPH <{GRAPH}> {{ {pat} . {named_m} "
                              f"{is_rel.format(v='p')} }} }} ORDER BY ?p ?m ?ml LIMIT {MAX_TRIPLES}")
        for r in rows:
            ml, p = r["ml"]["value"], _rel(r["p"]["value"])
            one.append((NAME[node], p, ml) if direction == "out" else (ml, p, NAME[node]))
        for d2 in ("out", "in"):
            pat2 = f"?m ?q ?f" if d2 == "out" else f"?f ?q ?m"
            named_f = _named("f", literals and d2 == "out")
            rows = _ask(endpoint, f"SELECT ?ml ?q ?fl WHERE {{ GRAPH <{GRAPH}> {{ {pat} . {pat2} . "
                                  f"?m <{LABEL}> ?ml . {named_f} {is_rel.format(v='p')} "
                                  f"{is_rel.format(v='q')} FILTER(?f != <{node}>) }} }} "
                                  f"ORDER BY ?m ?ml ?q ?f ?fl LIMIT {MAX_TRIPLES}")
            for r in rows:
                ml, q2, fl = r["ml"]["value"], _rel(r["q"]["value"]), r["fl"]["value"]
                two.append((ml, q2, fl) if d2 == "out" else (fl, q2, ml))
    return one, two


NAME: dict[str, str] = {}


def _questions_cut(questions: list[dict], cut, notes: dict | None, known_slow=None):
    """(q, one-hop, two-hop) per question the graph answers; a slow one goes to notes["missing"]."""
    slow = 0
    for q in questions:
        why = known_slow(q) if (known_slow and notes is not None) else None
        if why:
            notes["missing"][str(q["id"])] = f"no_subgraph: the graph timed out ({why})"
            print(f"[gcr export] {q['id']}: no subgraph, {why}")
            continue
        try:
            one, two = with_retries(cut, q)
        except Exception as e:
            if notes is None or not timed_out(e):
                raise
            notes["missing"][str(q["id"])] = f"no_subgraph: the graph timed out ({type(e).__name__}: {str(e)[:160]})"
            print(f"[gcr export] {q['id']}: no subgraph, the graph timed out twice ({type(e).__name__})")
            slow += 1
            if slow >= SLOW_IN_A_ROW:
                raise RuntimeError(f"{slow} questions in a row timed out at {TIMEOUT:.0f} s: the graph is overloaded; "
                                   "export again with --resume when it answers") from e
            continue
        slow = 0
        yield q, one, two


def biocq_records(questions: list[dict], endpoint: str, literals: bool = False, notes: dict | None = None):
    """BioCQ and KQA Pro (the same layout): literals=True also exports literal tails (KQA Pro's attributes)."""
    def cut(q):
        one, two = [], []
        for short, name in q["topic_entity"].items():
            node = f"{BASE}{KG_OF_PREFIX[short.split(':', 1)[0]]}/node/{short}"
            NAME[node] = name
            o, t = _hops(endpoint, node, literals)
            one += o
            two += t
        return one, two
    for q, one, two in _questions_cut(questions, cut, notes):
        graph, capped = _cap(one, two)
        yield q, list(q["topic_entity"].values()), graph, capped


# ── CWQ, WebQSP: RoG's released subgraphs, from the harness's topic entities ────────────────────────────────
def rog_rows(source: str, split: str):
    """RoG's rows ({"id", "q_entity", "graph"}) for a split: fetch.sh's local copy, or the hub."""
    folder = Path(source)
    if folder.is_dir():
        files = sorted(folder.glob(f"data/{split}-*.parquet"))
        if not files:
            raise SystemExit(f"{source}: no data/{split}-*.parquet (fetch.sh downloads rmanluo/RoG-<set>)")
        import pyarrow.parquet as pq
        for f in files:
            for batch in pq.ParquetFile(f).iter_batches(batch_size=32, columns=["id", "q_entity", "graph"]):
                yield from batch.to_pylist()
    else:
        from datasets import load_dataset
        for r in load_dataset(source, split=split, streaming=True):
            yield {"id": r["id"], "q_entity": r["q_entity"], "graph": r["graph"]}


def rog_records(questions: list[dict], rows, notes: dict):
    """Each question RoG has, its subgraph as released, from the harness's topic entities as RoG names them."""
    wanted = {str(q["id"]): q for q in questions}
    seen = set()
    for r in rows:
        qid = str(r["id"])
        q = wanted.get(qid)
        if q is None or qid in seen:
            continue
        seen.add(qid)
        sub = {"q_entity": list(r.get("q_entity") or []), "graph": [tuple(t) for t in r.get("graph") or []]}
        nodes, absent, mapped = topic_nodes(q.get("topic_entity"), sub)
        notes["topic"][qid] = {"nodes": nodes, "absent": absent, "mapped": mapped}
        yield q, nodes, sub["graph"], False
    for qid in wanted:
        if qid not in seen:
            notes["missing"][qid] = "no_subgraph: not in RoG's released subgraphs"


# ── GrailQA: a live 2-hop export from Freebase ─────────────────────────────────────────────────────────────────
def _fb_kept(v: str) -> str:
    """SPARQL filters: ?v is one of Freebase's own predicates and not one FiDeLiS's builder skips."""
    return (f'FILTER(STRSTARTS(STR(?{v}), "{FB}")) '
            + " ".join(f'FILTER(!CONTAINS(STR(?{v}), "{x}"))' for x in SKIP))


def _fb_value(v: str) -> str:
    """?v is a Freebase node or a plain or English literal; ?{v}l its English name, when it has one."""
    return (f'FILTER(isLiteral(?{v}) || STRSTARTS(STR(?{v}), "{FB}")) '
            f"FILTER(!isLiteral(?{v}) || lang(?{v}) = '' || langMatches(lang(?{v}), 'en')) "
            f"OPTIONAL {{ ?{v} <{FB}type.object.name> ?{v}l . FILTER(langMatches(lang(?{v}l), 'en')) }}")


def freebase_hops(endpoint: str, mid: str, name: str) -> tuple[list, list]:
    """The triples leading out of a Freebase node and out of each of its neighbours, by name, each query sorted."""
    names = {FB + mid: name}

    def named(b: dict, v: str) -> str:
        x = b[v]
        if x["type"] == "literal" or x["type"] == "typed-literal":
            return x["value"]
        label = b.get(v + "l", {}).get("value")
        return names.setdefault(x["value"], label or x["value"].removeprefix(FB))

    one = [(name, b["p"]["value"].removeprefix(FB), named(b, "m")) for b in _ask(endpoint, (
        f"SELECT ?p ?m ?ml WHERE {{ <{FB}{mid}> ?p ?m . {_fb_kept('p')} {_fb_value('m')} }} "
        f"ORDER BY ?p ?m ?ml LIMIT {MAX_TRIPLES}"))]
    two = []
    for b in _ask(endpoint, (
            f"SELECT ?m ?ml ?q ?f ?fl WHERE {{ <{FB}{mid}> ?p ?m . {_fb_kept('p')} FILTER(isIRI(?m)) "
            f"{_fb_value('m')} ?m ?q ?f . {_fb_kept('q')} {_fb_value('f')} FILTER(?f != <{FB}{mid}>) }} "
            f"ORDER BY ?m ?ml ?q ?f ?fl LIMIT {MAX_TRIPLES}")):
        two.append((named(b, "m"), b["q"]["value"].removeprefix(FB), named(b, "f")))
    return one, two


def freebase_records(questions: list[dict], endpoint: str, notes: dict | None = None):
    """The live 2-hop subgraph of each question; a topic node that timed out is not queried again."""
    fails: collections.Counter = collections.Counter()   # mid -> its timeouts in a row

    def cut(q):
        one, two = [], []
        for mid, name in (q.get("topic_entity") or {}).items():
            try:
                o, t = freebase_hops(endpoint, mid, name or mid)
            except Exception as e:
                if timed_out(e):
                    fails[mid] += 1
                raise
            fails[mid] = 0
            one += o
            two += t
        return one, two

    def known_slow(q):
        hubs = [m for m in (q.get("topic_entity") or {}) if fails[m] >= TRIES]
        return f"{', '.join(hubs)} timed out on an earlier question of this export" if hubs else None
    for q, one, two in _questions_cut(questions, cut, notes, known_slow):
        graph, capped = _cap(one, two)
        yield q, [name or mid for mid, name in (q.get("topic_entity") or {}).items()], graph, capped


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--set", required=True, help="cwq, webqsp, grailqa, metaqa, webqsp-unseen(-real), "
                                                "cwq-unseen(-real)")
    p.add_argument("--split", default="test")
    p.add_argument("--kb", type=Path, help="MetaQA: the kb.txt of the graph the set is answered on")
    p.add_argument("--endpoint", help="GrailQA, the Unseen halves, BioCQ, KQA Pro: the SPARQL endpoint of the graph the "
                                      "set is answered on (both Unseen halves: the extended graph's)")
    p.add_argument("--rog", help="CWQ, WebQSP: RoG's released data, fetch.sh's $GCR/hf/RoG-<set> (or rmanluo/RoG-<set>)")
    p.add_argument("--limit", type=int, default=0,
                   help="a smoke: export only the split's first N questions (into a folder of its own, never $GCR/data)")
    p.add_argument("--out", type=Path, required=True, help="a folder; GCR reads <out>/<set>/<split>.jsonl")
    p.add_argument("--resume", action="store_true",
                   help="keep the questions <out>/<set>/<split>.jsonl already holds (a half-written last line is cut) "
                        "and export the rest, appending; a question an earlier run left without a subgraph is tried "
                        "again. Without it the file is written anew")
    a = p.parse_args(argv)
    source = QUESTION_FILE.get((a.set, a.split), a.split)
    questions = question_records(a.set, a.split)
    if a.limit:                        # a smoke: the split's first questions only, into a folder of their own
        questions = questions[:a.limit]
    notes: dict = {"missing": {}, "topic": {}}
    dest = a.out / a.set
    out_file = dest / f"{a.split}.jsonl"
    sidecar = a.out / f"{a.set}.{a.split}.harness.json"
    done: dict = {}                    # id -> (gold inside the subgraph, at the cap), for the rows already written
    if a.resume and out_file.exists():
        _complete_last_line(str(out_file))
        gold = {str(q["id"]): q["answers"] for q in questions}
        for line in out_file.open(encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                names = {x for h, _, t in r["graph"] for x in (h, t)}
                done[str(r["id"])] = (any(g in names for g in gold.get(str(r["id"]), [])), len(r["graph"]) >= MAX_TRIPLES)
        if sidecar.exists():          # the topic notes of the rows kept (RoG's sets)
            before = json.loads(sidecar.read_text(encoding="utf-8"))
            if before.get("endpoint") != a.endpoint:
                p.error(f"{out_file} was cut from {before.get('endpoint')}, not {a.endpoint}: export it anew, "
                        "without --resume")
            notes["topic"].update({k: v for k, v in (before.get("topic") or {}).items() if k in done})
        print(f"[gcr export] {a.set} {a.split}: resuming, {len(done)} questions already exported")
    todo = [q for q in questions if str(q["id"]) not in done]
    if a.set in ROG_SETS:
        if not a.rog:
            p.error(f"--rog (RoG's released rmanluo/RoG-{a.set}, fetch.sh's $GCR/hf/RoG-{a.set}) is needed for {a.set}")
        records = rog_records(todo, rog_rows(a.rog, a.split), notes)
    elif a.set in LIVE_FREEBASE:
        if not a.endpoint:
            p.error(f"--endpoint (the Freebase SPARQL endpoint; for an Unseen half the extended graph's) is needed for {a.set}")
        records = freebase_records(todo, a.endpoint, notes)
    elif a.set.startswith("metaqa"):
        if not a.kb:
            p.error("--kb (the kb.txt of the graph the set is answered on) is needed for MetaQA")
        records = metaqa_records(todo, a.kb)
    elif a.set.startswith(("biocq", "kqapro")):
        if not a.endpoint:
            p.error("--endpoint (the SPARQL endpoint of the graph the set is answered on) is needed")
        records = biocq_records(todo, a.endpoint, literals=a.set.startswith("kqapro"), notes=notes)
    else:
        p.error(f"{a.set}: no GCR export for this set")
    dest.mkdir(parents=True, exist_ok=True)
    with out_file.open("a" if done else "w", encoding="utf-8") as f:
        for q, anchors, graph, capped in records:
            names = {x for h, _, t in graph for x in (h, t)}
            done[str(q["id"])] = (any(g in names for g in q["answers"]), capped)
            f.write(json.dumps({"id": str(q["id"]), "question": q["question"], "q_entity": anchors,
                                "a_entity": [], "answer": [], "choices": [],
                                "graph": [list(t) for t in graph]},
                               ensure_ascii=False) + "\n")
            f.flush()
    exported = [str(q["id"]) for q in questions if str(q["id"]) in done]
    n, reachable, capped_n = len(exported), sum(done[i][0] for i in exported), sum(done[i][1] for i in exported)
    print(f"[gcr export] {a.set} {a.split}: {n} questions -> {out_file}; subgraph capped at "
          f"{MAX_TRIPLES} triples for {capped_n}; a gold answer inside the subgraph for {reachable} "
          f"({reachable / max(n, 1):.0%}); {len(notes['missing'])} without a subgraph, so {reachable} of all "
          f"{len(questions)} ({reachable / max(len(questions), 1):.0%})")
    sidecar.write_text(json.dumps({"set": a.set, "split": a.split, "question_file": f"{source}.json",
                                   "limit": a.limit or None, "endpoint": a.endpoint,
                                   "questions": len(questions), "exported": exported,
                                   "missing": notes["missing"], "topic": notes["topic"]}, ensure_ascii=False, indent=1))
    if notes["topic"]:
        t = notes["topic"].values()
        print(f"[gcr export] {a.set} {a.split}: harness topic entities: {sum(1 for x in t if x['mapped'])} questions "
              f"with one renamed to RoG's node, {sum(1 for x in t if x['absent'])} with one not in the subgraph, "
              f"{sum(1 for x in t if not x['nodes'])} with none in it (GCR then has no path)")
    print(f"[gcr export] {a.set} {a.split}: {len(notes['missing'])} questions not exported (convert.py writes them "
          f"as misses) -> {sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
