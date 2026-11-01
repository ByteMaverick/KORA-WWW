"""KQA Pro adapter for the sampler: a question's KoPL program walked on the KQA Pro graph's endpoint."""
from __future__ import annotations

import os
import re
import threading

from kora.tools.kg import KGError

from ..graph import Rel
from ..rows import Constraint, Hop, Node, Path
from .kqapro_graph import BioKG, token_of

BASE = "http://skill-lab-research.org/kqapro"
LABEL = "http://www.w3.org/2000/01/rdf-schema#label"
TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
N_ENTITIES = 16960          # node ids 1..16960 are entities, the concepts follow
CAP = 300                   # a set over this is not walked (the evidence line would not fit a window either)
AMONG_CAP = 60              # a set a SelectAmong compares
CHUNK = 100                 # subjects per VALUES clause
RELATION_SAMPLE = 50        # nodes `relations` looks at
RELATION_LIMIT = 200
FINAL = ("What", "Count", "QueryAttr", "SelectAmong", "SelectBetween")
FILTERS = ("FilterStr", "FilterNum", "FilterYear", "FilterDate")
WRONG_ATTRIBUTES = 2
WRONG_VALUES = 3          # filters flipped or given another value
_FLIP_OP = {"<": ">", ">": "<", "=": "!=", "!=": "="}


class Unwalkable(Exception):
    """The program cannot be run on the served graph; the message is the reason (the split script counts them)."""


def slug(text: str) -> str:
    """A KQA Pro name as the export writes it into a URI."""
    return re.sub(r"[^A-Za-z0-9_]+", "_", text.strip()).strip("_")


def lit(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r") + '"'


def node_term(node_id: str) -> str:
    return f"<{BASE}/node/{node_id}>"


def value_node(text: str) -> Node:
    return Node(f"value:{text}", text)


def is_value(node: Node) -> bool:
    return node.id.startswith("value:")


def entity_number(node_id: str) -> int:
    try:
        return int(node_id.rsplit(":", 1)[-1])
    except ValueError:
        return 10 ** 9


_NUM = re.compile(r"^\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")
_DATE = re.compile(r"^\s*(-?\d{1,4})(?:[-/](\d{1,2})[-/](\d{1,2}))?")


def number_of(text: str) -> float | None:
    m = _NUM.match(text)
    return float(m.group(1)) if m else None


def date_of(text: str) -> tuple[int, int, int] | None:
    m = _DATE.match(text)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2) or 1), int(m.group(3) or 1)


def order_key(text: str):
    """A value to compare: a number, else a date; None when it is neither."""
    n = number_of(text)
    if n is not None and not re.match(r"^\s*-?\d{1,4}[-/]\d{1,2}[-/]\d{1,2}", text):
        return (0, n)
    d = date_of(text)
    return (1, d) if d else None


def same_value(a: str, b: str) -> bool:
    """The recorded answer against a graph literal: equal text, equal date, or equal number."""
    a, b = a.strip(), b.strip()
    if a.casefold() == b.casefold():
        return True
    da, db = re.match(r"^-?\d+[-/]\d+[-/]\d+$", a), re.match(r"^-?\d+[-/]\d+[-/]\d+$", b)
    if da and db:
        return date_of(a) == date_of(b)
    na, nb = number_of(a), number_of(b)
    if na is not None and nb is not None and not da and not db:
        rest_a, rest_b = _NUM.sub("", a).strip(), _NUM.sub("", b).strip()
        return na == nb and (not rest_a or not rest_b or rest_a.casefold() == rest_b.casefold())
    return False


def compare(op: str, left, right) -> bool:
    return {"=": left == right, "!=": left != right, "<": left < right, ">": left > right}[op]


class _Set:
    """What a program step produced: the nodes, the anchors they started from, the hops walked."""
    __slots__ = ("nodes", "anchors", "hops", "constraints")

    def __init__(self, nodes, anchors, hops, constraints=None):
        self.nodes, self.anchors, self.hops = nodes, anchors, hops
        self.constraints = list(constraints or [])      # the filters and joins applied, as lines


class KQAProGraph:
    """The sampler's Graph over the KQA Pro endpoint."""

    def __init__(self, provider: BioKG, bench: str = "kqapro"):
        self.kg, self.bench = provider, bench
        self._tokens: dict[str, str] | None = None     # token -> relation URI
        self._lock = threading.Lock()
        self._cache: dict[tuple, object] = {}

    # -- the endpoint -----------------------------------------------------------------------------------------

    def _ask(self, sparql: str) -> list[dict]:
        return self.kg.ask(sparql, timeout=60)

    def _memo(self, key: tuple, make):
        if key not in self._cache:
            self._cache[key] = make()
        return self._cache[key]

    def tokens(self) -> dict[str, str]:
        """Every relation and attribute predicate of the graph, by the token the evidence shows."""
        with self._lock:
            if self._tokens is None:
                rows = self._ask(f"SELECT DISTINCT ?p WHERE {{ ?s ?p ?o . FILTER(STRSTARTS(STR(?p), '{BASE}/relation/')) }}"
                                 f" LIMIT 20000")
                found: dict[str, str] = {}
                for r in rows:
                    uri = self.kg.value(r, "p") or ""
                    found.setdefault(token_of(uri), uri)
                self._tokens = found
            return self._tokens

    def predicate(self, name: str) -> tuple[str, str]:
        """(token, URI) of a KoPL relation or attribute name; Unwalkable when the graph has no such predicate."""
        uri = f"{BASE}/relation/{slug(name)}"
        token = token_of(uri)
        if self.tokens().get(token) is None:
            raise Unwalkable("a predicate the served graph lacks")
        return token, self.tokens()[token]

    def name(self, node_id: str) -> str | None:
        return self.kg.label(node_id)

    def _nodes_of(self, ids: list[str]) -> list[Node]:
        names = self.kg.labels(ids) if ids else {}
        return [Node(i, names.get(i)) for i in ids]

    def find(self, name: str, limit: int = 50) -> list[Node]:
        """Every entity (not a concept) labelled `name`, as KoPL's Find returns them."""
        rows = self._memo(("find", name), lambda: self._ask(
            f"SELECT ?e WHERE {{ ?e <{LABEL}> {lit(name)} }} LIMIT {limit}"))
        ids = sorted({(self.kg.value(r, "e") or "").rsplit("/node/", 1)[-1] for r in rows}, key=entity_number)
        return self._nodes_of([i for i in ids if i and entity_number(i) <= N_ENTITIES])

    def same_name(self, node: Node, limit: int = 3) -> list[Node]:
        if not node.name or is_value(node):
            return []
        return [n for n in self.find(node.name, 20) if n.id != node.id][:limit]

    # -- the Graph protocol -----------------------------------------------------------------------------------

    def relations(self, nodes: list[Node]) -> list[Rel]:
        ents = [n for n in nodes if not is_value(n)][:RELATION_SAMPLE]
        if not ents:
            return []
        values = " ".join(node_term(n.id) for n in ents)
        counts: dict[tuple[str, str], int] = {}
        for direction, pattern in (("out", "?s ?p ?o"), ("in", "?o ?p ?s . FILTER(isIRI(?o))")):
            rows = self._ask(f"SELECT ?p (COUNT(*) AS ?n) WHERE {{ VALUES ?s {{ {values} }} {pattern} "
                             f"FILTER(STRSTARTS(STR(?p), '{BASE}/relation/')) }} GROUP BY ?p ORDER BY DESC(?n) "
                             f"LIMIT {RELATION_LIMIT}")
            for r in rows:
                key = (token_of(self.kg.value(r, "p") or ""), direction)
                counts[key] = counts.get(key, 0) + int(self.kg.value(r, "n") or 0)
        return [Rel(t, d, c) for (t, d), c in sorted(counts.items(), key=lambda kv: -kv[1])]

    def follow(self, nodes: list[Node], rel: Rel, limit: int = 200) -> list[Node]:
        uri = self.tokens().get(rel.relation)
        ents = [n for n in nodes if not is_value(n)]
        if uri is None or not ents:
            return []
        objects: list[str] = []; literals: list[str] = []
        for start in range(0, len(ents), CHUNK):
            if len(objects) + len(literals) >= limit:
                break
            values = " ".join(node_term(n.id) for n in ents[start:start + CHUNK])
            pattern = f"?s <{uri}> ?o" if rel.direction == "out" else f"?o <{uri}> ?s"
            rows = self._ask(f"SELECT DISTINCT ?o WHERE {{ VALUES ?s {{ {values} }} {pattern} }} LIMIT {limit}")
            for r in rows:
                b = r.get("o") or {}
                if b.get("type") == "uri":
                    objects.append((b.get("value") or "").rsplit("/node/", 1)[-1])
                elif b.get("value") is not None:
                    literals.append(b["value"])
        objects = list(dict.fromkeys(objects))[:limit]
        literals = list(dict.fromkeys(literals))[:max(limit - len(objects), 0)]
        return self._nodes_of(sorted(objects, key=entity_number)) + [value_node(v) for v in literals]

    def attribute_values(self, nodes: list[Node], uri: str) -> dict[str, list[str]]:
        """The literals each entity holds on one attribute."""
        out: dict[str, list[str]] = {}
        ents = [n for n in nodes if not is_value(n)]
        for start in range(0, len(ents), CHUNK):
            values = " ".join(node_term(n.id) for n in ents[start:start + CHUNK])
            rows = self._ask(f"SELECT ?s ?o WHERE {{ VALUES ?s {{ {values} }} ?s <{uri}> ?o . FILTER(isLiteral(?o)) }} "
                             f"LIMIT {CAP * 20}")
            for r in rows:
                out.setdefault((self.kg.value(r, "s") or "").rsplit("/node/", 1)[-1], []).append(self.kg.value(r, "o") or "")
        return out

    def kinds(self, ids: list[str]) -> dict[str, str]:
        """Each entity's concept (`city`, `film`; the first by name when it has several); a value node is a `value`."""
        out: dict[str, str] = {}
        ents = [i for i in dict.fromkeys(ids) if not i.startswith("value:")]
        for start in range(0, len(ents), CHUNK):
            values = " ".join(node_term(i) for i in ents[start:start + CHUNK])
            try:
                rows = self._ask(f"SELECT ?s ?t WHERE {{ VALUES ?s {{ {values} }} ?s <{TYPE}> ?t }}")
            except KGError:
                continue
            found: dict[str, list[str]] = {}
            for r in rows:
                s_ = (self.kg.value(r, "s") or "").rsplit("/node/", 1)[-1]
                t_ = (self.kg.value(r, "t") or "").rsplit("/type/", 1)[-1]
                if s_ and t_:
                    found.setdefault(s_, []).append(t_.replace("_", " "))
            out.update({k: sorted(v)[0] for k, v in found.items()})
        return {i: ("value" if i.startswith("value:") else out.get(i, "")) for i in ids}

    def with_type(self, nodes: list[Node], concept: str) -> list[Node]:
        ents = [n for n in nodes if not is_value(n)]
        keep: set[str] = set()
        for start in range(0, len(ents), CHUNK):
            values = " ".join(node_term(n.id) for n in ents[start:start + CHUNK])
            rows = self._ask(f"SELECT ?s WHERE {{ VALUES ?s {{ {values} }} ?s <{TYPE}> <{BASE}/type/{slug(concept)}> }}")
            keep |= {(self.kg.value(r, "s") or "").rsplit("/node/", 1)[-1] for r in rows}
        return [n for n in ents if n.id in keep]

    # -- the program ------------------------------------------------------------------------------------------

    def _bound(self, nodes: list[Node], cap: int = CAP) -> None:
        if len(nodes) > cap:
            raise Unwalkable("a set over the cap")

    def _step(self, program: list[dict], i: int, skip: frozenset, memo: dict, flip: int | None = None) -> _Set:
        if i in memo:
            return memo[i]
        step = program[i]
        name, inputs, deps = step["function"], step["inputs"], step["dependencies"]
        if name == "Find":
            nodes = self.find(inputs[0])
            if not nodes:
                raise Unwalkable("an anchor not in the graph")
            out = _Set(nodes, nodes, [])
        elif name in ("And", "Or"):
            a, b = (self._step(program, d, skip, memo, flip) for d in deps)
            main, other = (a, b) if len(a.hops) >= len(b.hops) else (b, a)
            if i in skip:
                out = main
            else:
                ids = {n.id for n in other.nodes}
                if name == "And":
                    nodes = [n for n in main.nodes if n.id in ids]
                else:
                    nodes = main.nodes + [n for n in other.nodes if n.id not in {m.id for m in main.nodes}]
                joined = Constraint(len(main.hops), (other.hops[-1].relation if other.hops else "same as")
                                    + (" (or)" if name == "Or" else ""),
                                    ", ".join(dict.fromkeys(n.name for n in other.anchors if n.named)) or "?")
                out = self._with_nodes(main, nodes, [joined])
        else:
            prev = self._step(program, deps[0], skip, memo, flip)
            if name == "Relate":
                token, _ = self.predicate(inputs[0])
                direction = "out" if inputs[1] == "forward" else "in"
                objs = self.follow(prev.nodes, Rel(token, direction), CAP + 1) if prev.nodes else []
                self._bound(objs)
                out = _Set(objs, prev.anchors, prev.hops + [Hop(prev.nodes, token, direction, objs)], prev.constraints)
            elif i in skip:
                out = prev
            elif name == "FilterConcept":
                out = self._with_nodes(prev, self.with_type(prev.nodes, inputs[0]),
                                       [Constraint(len(prev.hops), "type", inputs[0])])
            elif name in FILTERS:
                used = list(inputs)
                if flip == i:
                    used = self._wrong(prev.nodes, name, inputs)
                    if used is None:
                        raise Unwalkable("no wrong value for this filter")
                token = self.predicate(used[0])[0]
                value = used[1] if name == "FilterStr" else f"{used[2]} {used[1]}"
                out = self._with_nodes(prev, self._filter(prev.nodes, name, used),
                                       [Constraint(len(prev.hops), token, value)])
            else:
                raise Unwalkable(f"a step the walk does not read ({name})")
        memo[i] = out
        return out

    @staticmethod
    def _with_nodes(s: _Set, nodes: list[Node], added: list[Constraint] | None = None) -> _Set:
        """The same chain ending on `nodes` (a narrowed or merged last set), with the lines of what narrowed it."""
        lines = s.constraints + list(added or [])
        if s.hops:
            hops = s.hops[:-1] + [Hop(s.hops[-1].subjects, s.hops[-1].relation, s.hops[-1].direction, nodes)]
            return _Set(nodes, s.anchors, hops, lines)
        return _Set(nodes, nodes, [], lines)

    def _wrong(self, nodes: list[Node], name: str, inputs: list[str]) -> list[str] | None:
        """A filter's inputs with a wrong value (a comparison reversed, or another value the set holds); None if none."""
        if name != "FilterStr":
            op = inputs[2]
            return [inputs[0], inputs[1], _FLIP_OP[op]] + list(inputs[3:]) if op in _FLIP_OP else None
        _, uri = self.predicate(inputs[0])
        values = self.attribute_values(nodes, uri)
        other = next((v for vs in values.values() for v in vs if v != inputs[1]), None)
        return [inputs[0], other] + list(inputs[2:]) if other else None

    def _filter(self, nodes: list[Node], name: str, inputs: list[str]) -> list[Node]:
        _, uri = self.predicate(inputs[0])
        values = self.attribute_values(nodes, uri)
        if name == "FilterStr":
            return [n for n in nodes if inputs[1] in values.get(n.id, [])]
        op = inputs[2]
        if name == "FilterNum":
            want = number_of(inputs[1])
            test = lambda v: (number_of(v) is not None and want is not None and compare(op, number_of(v), want))
        elif name == "FilterYear":
            want = date_of(inputs[1])
            test = lambda v: (date_of(v) is not None and want is not None and compare(op, date_of(v)[0], want[0]))
        else:
            want = date_of(inputs[1])
            test = lambda v: (date_of(v) is not None and want is not None and compare(op, date_of(v), want))
        return [n for n in nodes if any(test(v) for v in values.get(n.id, []))]

    def walk(self, question: dict, skip: frozenset = frozenset(), *, wrong: int | None = None,
             flip: int | None = None) -> Path:
        """The question's program executed on the graph: one Path, source "spine"; `skip` steps left out."""
        program = question["program"]
        functions = [s["function"] for s in program]
        final = functions[-1]
        if final not in FINAL:
            raise Unwalkable("a question type the sets do not hold")
        if "FindAll" in functions:
            raise Unwalkable("a FindAll")
        if any("Qualifier" in f or f.startswith("QFilter") for f in functions):
            raise Unwalkable("a qualifier step")
        memo: dict = {}
        last, inputs, deps = len(program) - 1, program[-1]["inputs"], program[-1]["dependencies"]
        if final == "SelectBetween":
            return self._compare(program, deps, inputs, skip, memo, wrong, flip)
        s = self._step(program, deps[0], skip, memo, flip)
        if final == "What":
            if not s.nodes:
                raise Unwalkable("an empty answer set")
            return Path(s.anchors, s.hops, "spine", extra={"kind": final}, constraints=s.constraints)
        if final == "Count":
            self._bound(s.nodes)
            n = Node(f"value:{len(s.nodes)}", str(len(s.nodes)))
            return Path(s.anchors, s.hops + [Hop(s.nodes, "count", "out", [n])], "spine", extra={"kind": final},
                        constraints=s.constraints)
        if final == "QueryAttr":
            token, uri = self.predicate(inputs[0])
            vals = self.attribute_values(s.nodes, uri)
            found = list(dict.fromkeys(v for n in s.nodes for v in vals.get(n.id, [])))
            if not found:
                raise Unwalkable("the entity has no such attribute")
            holders = [n for n in s.nodes if vals.get(n.id)]
            return Path(s.anchors, s.hops + [Hop(holders, token, "out", [value_node(v) for v in found])], "spine",
                        extra={"kind": final}, constraints=s.constraints)
        return self._among(s, inputs, wrong)

    def _ranked(self, nodes: list[Node], uri: str) -> list[tuple[Node, str, tuple]]:
        """(node, literal, sort key) per node that has an orderable value, in the set's order."""
        values = self.attribute_values(nodes, uri)
        out = []
        for n in nodes:
            keyed = sorted(((order_key(v), v) for v in values.get(n.id, []) if order_key(v) is not None))
            if keyed:
                out.append((n, keyed[0][1], keyed[0][0]))
        return out

    def _pick(self, ranked: list, largest: bool, token: str, anchors: list[Node], hops: list[Hop],
              constraints: list[Constraint] | None = None) -> Path:
        kinds = {k[0] for _, _, k in ranked}
        if len(ranked) < 2 or len(kinds) != 1:
            raise Unwalkable("nothing to compare")
        best = (max if largest else min)(k for _, _, k in ranked)
        winners = [n for n, _, k in ranked if k == best]
        nodes = [n for n, _, _ in ranked]
        values = [value_node(v) for _, v, _ in ranked]
        op = "largest" if largest else "smallest"
        evidence = hops + [Hop(nodes, token, "out", values), Hop(values, f"{token} ({op})", "out", winners)]
        return Path(anchors, evidence, "spine", extra={"kind": "compare", "ranked": [(n.id, v) for n, v, _ in ranked]},
                    constraints=list(constraints or []))

    def _among(self, s: _Set, inputs: list[str], wrong: int | None) -> Path:
        self._bound(s.nodes, AMONG_CAP)
        token, uri = self._attribute(inputs[0], s.nodes, wrong)
        return self._pick(self._ranked(s.nodes, uri), inputs[1] == "largest", token, s.anchors, s.hops, s.constraints)

    def _compare(self, program, deps, inputs, skip, memo, wrong, flip=None) -> Path:
        sides = [self._step(program, d, skip, memo, flip) for d in deps]
        chosen = []
        for side in sides:
            self._bound(side.nodes, AMONG_CAP)
            if not side.nodes:
                raise Unwalkable("an empty side")
            chosen.append(side)
        nodes = [n for side in chosen for n in side.nodes]
        token, uri = self._attribute(inputs[0], nodes, wrong)
        ranked = self._ranked(nodes, uri)
        # Each side must hold exactly one entity with the attribute; KoPL compares one entity against one.
        for side in chosen:
            if sum(1 for n, _, _ in ranked if n.id in {m.id for m in side.nodes}) != 1:
                raise Unwalkable("a side without one comparable entity")
        # The compared entities become the anchors, so their filters sit on the anchors (position 0).
        lines = [Constraint(0, c.relation, c.value) for side in chosen for c in side.constraints]
        return self._pick(ranked, inputs[1] == "greater", token, nodes, [], lines)

    def _attribute(self, name: str, nodes: list[Node], wrong: int | None) -> tuple[str, str]:
        """The attribute the question compares on; with `wrong` the n-th other attribute the nodes share."""
        if wrong is None:
            return self.predicate(name)
        gold = self.predicate(name)[0]
        shared = []
        for rel in self.relations(nodes):
            if rel.direction == "out" and rel.relation != gold and rel.count >= len(nodes):
                uri = self.tokens()[rel.relation]
                if len(self._ranked(nodes, uri)) >= 2:
                    shared.append((rel.relation, uri))
        if wrong >= len(shared):
            raise Unwalkable("no other attribute to compare on")
        return shared[wrong]

    # -- spine and variants -----------------------------------------------------------------------------------

    def spine(self, question: dict) -> Path | None:
        try:
            return self.walk(question)
        except (Unwalkable, KGError):
            return None

    def spine_variants(self, question: dict, spine: Path) -> list[Path]:
        """The program with one filter (or And/Or branch) left out; for comparisons, the other entity and attribute."""
        out: list[Path] = []
        seen = {spine.key()}
        program = question["program"]
        for i, step in enumerate(program):
            if step["function"] in FILTERS + ("FilterConcept", "And", "Or"):
                try:
                    p = self.walk(question, frozenset({i}))
                except (Unwalkable, KGError):
                    continue
                if p.key() not in seen:
                    seen.add(p.key()); p.source = "drop_constraint"; out.append(p)
        flips = 0
        for i, step in enumerate(program):
            if flips >= WRONG_VALUES or step["function"] not in FILTERS:
                continue
            try:
                p = self.walk(question, flip=i)
            except (Unwalkable, KGError):
                continue
            # a wrong value that empties a set on the way (no anchor left, a hop reaching nothing) is not a row anyone
            # would judge: a count of nothing, an attribute of nobody
            if p.key() not in seen and p.candidate and p.anchors and all(h.objects for h in p.hops):
                seen.add(p.key()); p.source = "wrong_value"; out.append(p); flips += 1
        if spine.extra.get("kind") == "compare":
            last = spine.hops[-1]
            ranked = spine.extra["ranked"]
            losers = [n for n in spine.hops[-2].subjects if n.id not in {w.id for w in last.objects}]
            if losers:
                other = Path(spine.anchors, spine.hops[:-1] + [Hop(last.subjects, last.relation, "out", losers[:1])],
                             "sibling", extra={"kind": "compare", "ranked": ranked}, constraints=list(spine.constraints))
                if other.key() not in seen:
                    seen.add(other.key()); out.append(other)
            for k in range(WRONG_ATTRIBUTES):
                try:
                    p = self.walk(question, wrong=k)
                except (Unwalkable, KGError):
                    break
                # Only a wrong attribute that changes the answer is a negative; one that picks the same entity is not.
                if p.key() not in seen and {n.id for n in p.candidate} != {n.id for n in spine.candidate}:
                    seen.add(p.key()); p.source = "sibling"; out.append(p)
        return out


def endpoint() -> str:
    return os.environ.get("KQAPRO_SPARQL", "http://localhost:9091/sparql")


def make_graph(bench: str = "kqapro") -> KQAProGraph:
    return KQAProGraph(BioKG(endpoint(), graphs=("kqapro",), schema="kqapro"), bench)
