"""Two filters, and the distinction between them is the heart of the design."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from kora.modules.decomposition.structural.graph import (
    ConstraintGraph,
    Cut,
    Leaf,
    components_of,
)

__all__ = ["Verdict", "executable", "validate"]

CARRIED = ("identity", "intersect", "union", "select", "project")


class Verdict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    ok: bool
    problems: list[str] = Field(default_factory=list)

    def __bool__(self) -> bool:
        return self.ok


def validate(cut: Cut, graph: ConstraintGraph,
             requirements: list[str] | None = None) -> Verdict:
    """Semantic validity: does this cut still mean the question?"""
    problems: list[str] = []
    known = {c.id for c in graph.constraints}
    covered = [cid for leaf in cut.leaves for cid in leaf.constraint_ids]

    missing = known - set(covered)
    if missing:
        problems.append(f"drops {', '.join(sorted(missing))}")
    invented = set(covered) - known
    if invented:
        problems.append(f"invents {', '.join(sorted(invented))}")

    bound = set(graph.grounded) | set(graph.literals)
    for leaf in cut.leaves:
        for name in leaf.inputs:
            if not name.startswith("?") and name not in bound:
                problems.append(f"{leaf.id} names {name!r}, which nothing bound")

    produced: dict[str, set[str]] = {}
    for leaf in cut.leaves:
        if leaf.output:
            produced.setdefault(leaf.output, set()).add(leaf.id)
    earlier = cut.runs_before()
    for leaf in cut.leaves:
        for name in leaf.inputs:
            if not name.startswith("?"):
                continue
            if name in graph.domains or produced.get(name, set()) & earlier[leaf.id]:
                continue
            problems.append(f"{leaf.id} needs {name}, which nothing before it produces")

    if _cyclic(cut):
        problems.append("the edges form a cycle")

    # The outputs, under the recomposition, must rebuild the answer.
    if cut.recomposition not in CARRIED:
        problems.append(f"recomposition {cut.recomposition!r} needs tuples the state "
                        f"cannot hold")
    elif graph.answer.variable and not any(
            leaf.output == graph.answer.variable for leaf in cut.leaves):
        problems.append(f"nothing produces the answer variable {graph.answer.variable}")

    if len(cut.leaves) > 1:
        for leaf in cut.leaves:
            if set(leaf.constraint_ids) == known and len(known) > 1:
                problems.append(f"{leaf.id} carries every constraint: it restates the parent")

    if requirements is not None:
        claimed = {r for leaf in cut.leaves for r in leaf.satisfies}
        stray = claimed - set(requirements)
        if stray:
            problems.append(f"claims {', '.join(sorted(stray))}, not this task's")
        unclaimed = set(requirements) - claimed
        if unclaimed and claimed:
            problems.append(f"no leaf claims {', '.join(sorted(unclaimed))}")

    return Verdict(ok=not problems, problems=problems)


def executable(cut: Cut, graph: ConstraintGraph) -> Verdict:
    """Can each leaf actually be run from where the state now stands?"""
    problems: list[str] = []
    upstream: dict[str, set[str]] = {leaf.id: set() for leaf in cut.leaves}
    for before, then in cut.edges:
        upstream.setdefault(then, set()).add(before)
    produced = {leaf.id: leaf.output for leaf in cut.leaves}

    for leaf in cut.leaves:
        fed = {produced[b] for b in upstream.get(leaf.id, set()) if produced.get(b)}
        for component in components_of([graph.constraint(c) for c in leaf.constraint_ids]):
            if not anchors(component, graph, leaf, fed):
                names = ", ".join(graph.constraint(c).render() for c in component)
                problems.append(f"{leaf.id} has nothing to stand on for: {names}")
    return Verdict(ok=not problems, problems=problems)


def anchors(component: list[str], graph: ConstraintGraph, leaf: Leaf,
            fed: set[str]) -> list[str]:
    """Where a component of a leaf could start a query from."""
    found: list[str] = []
    for constraint_id in component:
        constraint = graph.constraint(constraint_id)
        for argument in constraint.arguments:
            if argument in graph.grounded or argument in fed or graph.domains.get(argument):
                found.append(argument)
    return found


def _cyclic(cut: Cut) -> bool:
    after: dict[str, list[str]] = {}
    for before, then in cut.edges:
        after.setdefault(before, []).append(then)
    colour: dict[str, int] = {}

    def visit(node: str) -> bool:
        if colour.get(node) == 1:
            return True
        if colour.get(node) == 2:
            return False
        colour[node] = 1
        for other in after.get(node, []):
            if visit(other):
                return True
        colour[node] = 2
        return False

    return any(visit(leaf.id) for leaf in cut.leaves)
