"""The one renderer: an estimator state -> (question, evidence), the same at training and at inference.

The estimator reads a text pair: `question`, and `evidence` = the facts retrieved so far, one triple per line
(intermediate sets as ?x1, ?x2), constraint lines, then the candidate answer with its count. Training rows and live
requests both go through this module, so they cannot be written differently.

    from_kora(goal, state, ids, n, query, names, chain=())   a KORA state (the server's input)
    from_path(question, start, hops, constraints, candidate, n)  a generated path (Data A)
    validate(question, evidence) -> list of problems (empty = valid)
"""
from __future__ import annotations

import re

HEAD = "EVIDENCE (facts retrieved so far, as subject | relation | object)"
SHOWN, SHORT = 30, 3                    # names in a full list; names in a constraint line's subject

# ---------------------------------------------------------------- KORA's traversed path
EDGE = re.compile(r"^(.*?) -\[(.*?)\]-> \{(.*)\} \((\d+)\)\s*$")
IDS = re.compile(r"([mg]\.[0-9a-z_]+|[A-Za-z]+KG:[A-Za-z0-9_:.-]+)")


def display(names, total):
    names = list(dict.fromkeys(n for n in names if n))
    s = ", ".join(names[:SHOWN])
    return s + (f", ... (+{total - len(names[:SHOWN])} more)" if total > len(names[:SHOWN]) else "") if s else "(none)"


def short(names, total):
    names = list(dict.fromkeys(n for n in names if n))
    return ", ".join(names[:SHORT]) + (f" (+{total - SHORT} more)" if total > SHORT else "")


def kora_list(text):
    m = re.match(r"^(.*?)(?: … and (\d+) more)?$", text.strip())
    names = [x.strip() for x in m.group(1).split(",") if x.strip()]
    return names, len(names) + int(m.group(2) or 0)


def evidence_block(state):
    if "EVIDENCE" not in state:
        return []
    return state.split("EVIDENCE (the path that was traversed)")[-1].split("CANDIDATE ANSWER")[0].strip().splitlines()


def ids_to_name(state, ids):
    """The ids a KORA state's text needs named: the first SHOWN candidates and every id on its path."""
    ev = state.split("EVIDENCE")[1].split("CANDIDATE")[0] if "EVIDENCE" in state else ""
    return list(dict.fromkeys(list(ids)[:SHOWN] + IDS.findall(ev)))


def _text(goal, lines, total, cand):
    return (f"QUESTION\n{goal or ''}\n\n{HEAD}\n" + ("\n".join(f"  {l}" for l in lines) or "  (nothing retrieved)")
            + f"\n\nCANDIDATE ANSWER ({total} items)\n" + ("; ".join(list(dict.fromkeys(cand))[:SHOWN]) or "(empty)"))


def to_E(state, goal, ids, total, query, names):
    """A KORA state without a parseable query (the root's), from KORA's traversed path."""
    nm = lambda x: names.get(x, x)
    cand = [nm(i) for i in ids]
    lines, k = [], 0
    edges = [e for e in (EDGE.match(l.strip()) for l in evidence_block(state)) if e]
    pats = [tuple(t.split()) for t in re.findall(r"(\S+\s+[\w.]+\s+\S+?)\s*(?:\.|\})", query or "")]
    for j, e in enumerate(edges):
        start, rels, objs = e.group(1), e.group(2), e.group(3)
        chain = [x.strip() for x in rels.split(" / ") if x.strip() and not x.strip().startswith(":")]
        start = ", ".join(nm(x.strip()) for x in start.split(","))
        if j == len(edges) - 1:
            obj_names, obj_total = cand, total
        else:
            ln, lt = kora_list(objs); obj_names, obj_total = [nm(x) for x in ln], lt
        node, used, prev = start, set(), None
        for h, rel in enumerate(chain or ["(unknown)"]):
            if h == len(chain) - 1 or not chain:
                nxt = display(obj_names, obj_total)
            else:
                k += 1; nxt = f"?x{k}"
            d, cur = "out", None
            for pi, pt in enumerate(pats):
                if pi in used or len(pt) != 3 or pt[1] != rel:
                    continue
                s_, _, o_ = pt
                here = (lambda t: (t == prev) if prev else not t.startswith("?"))
                if here(s_):
                    d, cur = "out", o_; used.add(pi); break
                if here(o_):
                    d, cur = "in", s_; used.add(pi); break
            lines.append(f"{node} | {rel} | {nxt}" if d == "out" else f"{nxt} | {rel} | {node}")
            node, prev = nxt, cur
    q = query or ""
    if re.search(r"ORDER\s+BY", q, re.I) and re.search(r"LIMIT\s+1\b", q):
        d = "(largest)" if re.search(r"DESC", q, re.I) else "(smallest)"
        ov = re.search(r"ORDER\s+BY\s+(?:DESC|ASC)?\s*\(?\s*(\?\w+)", q, re.I)
        rel = re.search(rf"\S+\s+(\S+)\s+{re.escape(ov.group(1))}\b", q) if ov else None
        lines.append(f"{short(cand, total)} | {(rel.group(1) if rel else 'order')} | {d}")
    return _text(goal, lines, total, cand)


# ---------------------------------------------------------------- KORA's query
TOK = re.compile(r'"[^"]*"|<[^>]*>|\S+')
URI = re.compile(r"^https?://\S+/node/(\S+)$")
CMP = {":ge": ">=", ":gt": ">", ":lt": "<", ":le": "<=", ":eq": "=", ":ne": "!="}
SET = {":in": "in", ":not_in": "not in"}


def ident(x):
    x = x.strip('"<>')
    m = URI.match(x)
    return m.group(1) if m else x.lstrip("@")


def parse(q):
    """(triples, order, limit, select) of a KORA query; triples include its restriction words."""
    q = q if isinstance(q, str) else ""                   # a root state has none (None, or NaN from a table)
    i, j = q.find("{"), q.rfind("}")
    if i < 0 or j < 0:
        return None
    triples = []
    for part in re.split(r"\s\.\s|\s\.$", " " + q[i + 1:j].strip() + " "):
        t = TOK.findall(part)
        if len(t) == 3:
            triples.append(tuple(t))
    tail = q[j + 1:]
    order = re.search(r"ORDER\s+BY\s+(DESC|ASC)?\s*\(?\s*(\?\w+)\s*\)?\s*(DESC|ASC)?", tail, re.I)
    limit = re.search(r"LIMIT\s+(\d+)", tail, re.I)
    sel = re.search(r"SELECT\s+(?:DISTINCT\s+)?(?:COUNT\s*\(\s*)?(\?\w+)", q, re.I)
    if not triples or not sel:
        return None
    direction = None
    if order:
        direction = (order.group(1) or order.group(3) or "ASC").upper()
    return triples, (direction, order.group(2)) if order else None, int(limit.group(1)) if limit else None, sel.group(1)


def ids_in(q):
    """Constants in the query that the graph can name."""
    p = parse(q)
    if not p:
        return []
    return [ident(x) for s, r, o in p[0] for x in (s, o) if not x.startswith("?") and not r == ":type"]


def compose(triples, order, limit, sel, goal, ids, total, names):
    """Triples -> the estimator's text: the answer variable is the candidate list, other
    variables ?x<k> in order of appearance, a compared value folded into the hop that reached it, ORDER BY ... LIMIT 1
    as (largest) / (smallest)."""
    nm = lambda x: names.get(ident(x), ident(x).strip('"'))
    cand = [names.get(i, i) for i in ids]
    var = {}

    def term(t, full=True):
        if not t.startswith("?"):
            return nm(t)
        if t == sel:
            return display(cand, total) if full else short(cand, total)
        var.setdefault(t, f"?x{len(var) + 1}")
        return var[t]

    reached = {}
    for s, r, o in triples:
        if o.startswith("?") and r != ":type":
            reached.setdefault(o, (s, r))
    folded, cons = set(), []
    for s, r, o in triples:
        if r in CMP and s.startswith("?") and s != sel and s in reached:
            subj, rel = reached[s]; folded.add(s)
            cons.append((subj, rel.lstrip(":"), f"{CMP[r]} {nm(o)}"))
        elif r in CMP:
            cons.append((s, CMP[r], nm(o)))
        elif r in SET:
            cons.append((s, SET[r], "(the set " + o.lstrip("@") + ")" if o.startswith("@T") else nm(o)))
    if order and limit == 1:
        d, v = order
        word = "(largest)" if d == "DESC" else "(smallest)"
        if v != sel and v in reached:
            subj, rel = reached[v]; folded.add(v); cons.append((subj, rel.lstrip(":"), word))
        else:
            cons.append((v, "order", word))
    lines = []
    for s, r, o in triples:
        if r == ":type" or r in CMP or r in SET:
            continue
        if o in folded and reached.get(o) == (s, r):
            continue
        if r.startswith(":"):
            if o.startswith("?"):
                continue
            cons.append((s, r[1:], nm(o)))
            continue
        lines.append(f"{term(s)} | {r} | {term(o)}")
    lines += [f"{term(s, full=False)} | {r} | {v}" for s, r, v in cons]
    return _text(goal, lines, total, cand)


def render(state, goal, ids, total, query, names):
    p = parse(query)
    if not p:
        return to_E(state, goal, ids, total, query, names)
    triples, order, limit, sel = p
    return compose(triples, order, limit, sel, goal, ids, total, names)


def _evidence_lines(text):
    body = text.split(HEAD + "\n", 1)[1].split("\n\nCANDIDATE ANSWER", 1)[0]
    return [l[2:] for l in body.splitlines() if l.startswith("  ") and l.strip() != "(nothing retrieved)"]


def render_chain(chain, state, goal, ids, total, query, names):
    """A leaf that starts from earlier tasks' outputs: their queries first, ?x numbering continued."""
    leaf = render(state, goal, ids, total, query, names)
    if not chain or not parse(query):
        return leaf
    lines, k = [], 0
    for link in chain:
        if not parse(link.get("query")):
            continue
        part = render("", goal, link.get("ids") or [], int(link.get("n") or 0), link["query"], names)
        ev, shift = _evidence_lines(part), k
        ev = [re.sub(r"\?x(\d+)", lambda m: f"?x{int(m.group(1)) + shift}", l) for l in ev]
        k = max([int(x) for l in ev for x in re.findall(r"\?x(\d+)", l)] + [k])
        lines += ev
    own = [re.sub(r"\?x(\d+)", lambda m: f"?x{int(m.group(1)) + k}", l) for l in _evidence_lines(leaf)]
    head, tail = leaf.split(HEAD + "\n", 1)
    return head + HEAD + "\n" + "\n".join(f"  {l}" for l in lines + own) + "\n\nCANDIDATE ANSWER" + tail.split("\n\nCANDIDATE ANSWER", 1)[1]


# ---------------------------------------------------------------- the pair, and its contract
def split_text(text):
    """The live text -> (question, evidence): the QNLI-style pair the model reads."""
    q = text.split("QUESTION\n", 1)[1].split("\n\n" + HEAD, 1)[0]
    return q, HEAD + text.split("\n\n" + HEAD, 1)[1]


def joined(question, evidence):
    """The pair as one block, as the server logs it."""
    return f"QUESTION\n{question}\n\n{evidence}"


def from_kora(goal, state, ids, n, query, names, chain=()):
    """A KORA state, as the server receives it -> (question, evidence)."""
    ids = [str(i) for i in ids]
    query = query if isinstance(query, str) and query.strip() else None
    state = state if isinstance(state, str) else ""
    usable = list(chain or [])
    if usable and not (set(usable[-1].get("ids") or []) & set(ids_in(query))):
        usable = []                                  # the leaf does not start from the last link's output
    text = (render_chain(usable, state, goal, ids, int(n or len(ids)), query, names) if usable
            else render(state, goal, ids, int(n or len(ids)), query, names))
    return split_text(text)


DIRECTION = {">": "out", "<": "in", "out": "out", "in": "in"}
COMPARE = {"<": ":lt", "<=": ":le", ">": ":gt", ">=": ":ge", "=": ":eq", "!=": ":ne"}


def from_path(question, start, hops, constraints, candidate, n):
    """A generated path -> (question, evidence), through the same `compose` as KORA's queries.
    start: the start entity's name (or None when it is unnamed); hops: [(relation, direction)];
    constraints: [(set index, relation, value)] (0 = the start, k = the set reached by hop k; value a name, a
    comparison '>= 1970', or '(largest)' / '(smallest)'); candidate: names; n: the candidate count."""
    q = lambda name: '"' + name.replace('"', "'") + '"'
    terms = [q(start) if start else "?s"]
    triples = []
    for k, (rel, d) in enumerate(hops, 1):
        obj = "?ans" if k == len(hops) else f"?v{k}"
        s = terms[-1]
        triples.append((s, rel, obj) if DIRECTION[d] == "out" else (obj, rel, s))
        terms.append(obj)
    order, limit = None, None
    for j, (at, rel, value) in enumerate(constraints):
        t = terms[min(max(at, 0), len(terms) - 1)]
        value = value.strip()
        if value in ("(largest)", "(smallest)"):
            v = f"?c{j}"; triples.append((t, rel, v))
            order, limit = ("DESC" if value == "(largest)" else "ASC", v), 1
            continue
        op, _, rest = value.partition(" ")
        if op in COMPARE and rest:
            v = f"?c{j}"; triples += [(t, rel, v), (v, COMPARE[op], q(rest))]
            continue
        triples.append((t, rel, q(value)))
    return split_text(compose(triples, order, limit, "?ans", question, list(candidate), int(n), {}))


def validate(question, evidence):
    """What a state must have before the model may score it; the problems found (empty = valid)."""
    problems = []
    if not (question or "").strip():
        problems.append("no question")
    lines = [l for l in evidence.split("\n\nCANDIDATE ANSWER")[0].splitlines()[1:] if l.strip() and l.strip() != "(nothing retrieved)"]
    if not lines:
        problems.append("no evidence line")
    m = re.search(r"\n\nCANDIDATE ANSWER \((\d+) items\)\n(.*)$", evidence, re.S)
    if not m:
        problems.append("no candidate block")
    else:
        names = [x for x in m.group(2).split("; ") if x.strip() and x.strip() != "(empty)"]
        if not names:
            problems.append("no named candidate")
        elif int(m.group(1)) < len(names):
            problems.append("count below the names shown")
    return problems
