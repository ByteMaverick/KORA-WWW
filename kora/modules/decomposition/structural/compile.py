"""Adapter compiling algebraic Hypertree DAGs and Cuts into DecompositionResult."""

from __future__ import annotations

from typing import TYPE_CHECKING

from kora.provenance.models import Spend
from kora.modules.decomposition.decomposer import (
    DecompositionResult,
    Dependency,
    OutputSpec,
    Subtask,
)
from kora.modules.decomposition.structural.graph import (
    Constraint,
    ConstraintGraph,
    Cut,
    Leaf,
)

if TYPE_CHECKING:
    from kora.modules.decomposition.structural.hypertree import HypertreeDAG

__all__ = [
    "COMBINE",
    "compile_cut",
    "compile_hypertree_dag",
    "not_applicable",
    "to_moves",
]

COMBINE = {"intersect": "intersect", "union": "union", "select": "select", "project": "project", "identity": "identity"}


def compile_hypertree_dag(
    dag: HypertreeDAG,
    graph: ConstraintGraph,
    task_id: str = "",
    spend: Spend | None = None,
    requirements: dict | None = None,
) -> DecompositionResult:
    """Compiles an algebraic HypertreeDAG into the Controller's DecompositionResult."""
    requirements = requirements or {}
    if not dag.nodes:
        return not_applicable(task_id, "empty hypergraph", spend=spend)

    ordered_node_ids = _topological_sort_hypertree(dag)
    names = {nid: f"c{n}" for n, nid in enumerate(ordered_node_ids, start=1)}

    producer_of: dict[str, str] = {}
    for nid in ordered_node_ids:
        node = dag.nodes[nid]
        if node.output_variable:
            producer_of[node.output_variable] = names[nid]

    subtasks: list[Subtask] = []
    for nid in ordered_node_ids:
        node = dag.nodes[nid]
        subtask_id = names[nid]

        inputs: list[str] = []
        for child_id in node.child_ids:
            child_node = dag.nodes[child_id]
            if child_node.output_variable:
                inputs.append(f"output:{names[child_id]}")

        for c in node.assigned_clauses:
            for arg in c.arguments:
                if arg.startswith("?"):
                    if arg in graph.domains and arg not in node.separator_variables:
                        inputs.append(arg.lstrip("?"))
                elif arg in graph.grounded or arg in graph.literals:
                    inputs.append(arg)

        unique_inputs = list(dict.fromkeys(inputs))

        leaf = Leaf(
            id=nid,
            constraint_ids=list(node.clause_ids),
            inputs=unique_inputs,
            output=node.output_variable,
            satisfies=[c.source_requirement for c in node.assigned_clauses if c.source_requirement],
        )

        subtasks.append(
            Subtask(
                id=subtask_id,
                goal=_goal(leaf, graph, requirements),
                required_inputs=unique_inputs,
                satisfies=list(dict.fromkeys(leaf.satisfies)),
                output=_output(leaf, graph, requirements, subtask_id),
            )
        )

    edges: list[Dependency] = []
    for nid in ordered_node_ids:
        node = dag.nodes[nid]
        for child_id in node.child_ids:
            edges.append(Dependency(**{"from": names[child_id], "to": names[nid]}))

    completion = [names[dag.root_id]] if dag.root_id in names else [names[ordered_node_ids[-1]]]

    recomp = "intersect" if all(
        n.output_variable == graph.answer.variable for n in dag.nodes.values()
    ) else ("identity" if len(dag.nodes) == 1 else "project")

    return DecompositionResult(
        applicable=True,
        task_id=task_id,
        subtasks=subtasks,
        dependencies=edges,
        completion=completion,
        combine=COMBINE.get(recomp, "union"),
        recomposition=recomp,
        rationale=f"algebraic decomposition with width {dag.width} and {len(dag.nodes)} bags ({dag.fingerprint})",
        spend=spend or Spend(),
    )


def compile_cut(
    cut: Cut,
    graph: ConstraintGraph,
    task_id: str = "",
    spend: Spend | None = None,
    requirements: dict | None = None,
) -> DecompositionResult:
    """Compiles a Cut into DecompositionResult (backwards compatible)."""
    requirements = requirements or {}
    names = {leaf.id: f"c{n}" for n, leaf in enumerate(cut.leaves, start=1)}
    earlier = cut.runs_before()

    subtasks: list[Subtask] = []
    for leaf in cut.leaves:
        produced = {
            other.output: names[other.id]
            for other in cut.leaves
            if other.output and other.id in earlier[leaf.id]
        }
        inputs: list[str] = []
        for name in leaf.inputs:
            if name.startswith("?"):
                producer = produced.get(name)
                if producer and producer != names[leaf.id]:
                    inputs.append(f"output:{producer}")
                elif graph.domains.get(name):
                    inputs.append(name.lstrip("?"))
            else:
                inputs.append(name)
        subtasks.append(
            Subtask(
                id=names[leaf.id],
                goal=_goal(leaf, graph, requirements),
                required_inputs=list(dict.fromkeys(inputs)),
                satisfies=list(leaf.satisfies),
                output=_output(leaf, graph, requirements, names[leaf.id]),
            )
        )

    edges = [Dependency(**{"from": names[a], "to": names[b]}) for a, b in cut.edges]
    completion = _completion(cut, names)
    return DecompositionResult(
        applicable=True,
        task_id=task_id,
        subtasks=subtasks,
        dependencies=edges,
        completion=completion,
        combine=COMBINE.get(cut.recomposition, "union"),
        recomposition=cut.recomposition,
        rationale=_why(cut, graph),
        spend=spend or Spend(),
    )


def not_applicable(
    task_id: str,
    reason: str,
    spend: Spend | None = None,
) -> DecompositionResult:
    """No simplifying factorisation exists."""
    return DecompositionResult(
        applicable=False,
        task_id=task_id,
        subtasks=[],
        rationale=reason,
        spend=spend or Spend(),
    )


def _topological_sort_hypertree(dag: HypertreeDAG) -> list[str]:
    visited: set[str] = set()
    order: list[str] = []

    def post_order(nid: str) -> None:
        if nid in visited:
            return
        visited.add(nid)
        node = dag.nodes[nid]
        for cid in sorted(node.child_ids):
            if cid in dag.nodes:
                post_order(cid)
        order.append(nid)

    if dag.root_id in dag.nodes:
        post_order(dag.root_id)

    for nid in sorted(dag.nodes):
        if nid not in visited:
            post_order(nid)

    return order


def _goal(leaf: Leaf, graph: ConstraintGraph, requirements: dict) -> str:
    written = [
        requirements[r].goal
        for r in _claimed(leaf, graph)
        if r in requirements and requirements[r].goal
    ]
    if written:
        return " Then: ".join(dict.fromkeys(written))
    return " and ".join(_readable(graph.constraint(c), leaf) for c in leaf.constraint_ids)


def _readable(constraint: Constraint, leaf: Leaf) -> str:
    text = constraint.template or constraint.render()
    for argument in constraint.arguments:
        text = text.replace("{" + argument + "}", _phrase(argument))
    wanted = _phrase(leaf.output) if leaf.output else ""
    text = text.replace("{", "").replace("}", "")
    return f"determine {wanted}: {text}" if wanted else text


def _phrase(name: str) -> str:
    if not name.startswith("?"):
        return name
    return "the " + name.lstrip("?").replace("_", " ")


def _output(
    leaf: Leaf,
    graph: ConstraintGraph,
    requirements: dict,
    fallback_name: str,
) -> OutputSpec:
    name = (leaf.output or fallback_name).lstrip("?")
    for requirement_id in _claimed(leaf, graph):
        found = requirements.get(requirement_id)
        if found is not None and found.expected_output:
            return OutputSpec(
                name=name,
                type=found.expected_output,
                entity_type=found.expected_entity_type or None,
            )
    return OutputSpec(name=name, type="value" if _is_value(name) else "entity_set")


_VALUES = ("year", "date", "count", "number", "amount", "percentage", "total")


def _is_value(name: str) -> bool:
    lowered = name.lower()
    return any(word in lowered for word in _VALUES)


def _claimed(leaf: Leaf, graph: ConstraintGraph) -> list[str]:
    found: list[str] = []
    for constraint_id in leaf.constraint_ids:
        try:
            source = graph.constraint(constraint_id).source_requirement
            if source and source not in found:
                found.append(source)
        except KeyError:
            continue
    return found


def _completion(cut: Cut, names: dict[str, str]) -> list[str]:
    if cut.recomposition in ("intersect", "union"):
        return [names[leaf.id] for leaf in cut.leaves]
    consumed = {a for a, _ in cut.edges}
    last = [leaf for leaf in cut.leaves if leaf.id not in consumed]
    return [names[leaf.id] for leaf in (last or cut.leaves[-1:])]


def _why(cut: Cut, graph: ConstraintGraph) -> str:
    if cut.kind == "intersection":
        return (
            f"{len(cut.leaves)} constraints restrict {graph.answer.variable} "
            f"independently, so they run separately and intersect"
        )
    if cut.kind == "chain":
        return (
            "each step needs what the last produced: "
            + " then ".join(leaf.output for leaf in cut.leaves)
        )
    if cut.kind == "whole":
        return "one lookup settles every constraint"
    return f"{len(cut.leaves)} leaves, recomposed by {cut.recomposition}"


def to_moves(result: DecompositionResult, state, task):
    from kora.modules.controller.moves import Ask, Moves
    by_role = {n.lower(): [e.id for e in refs] for n, refs in task.bindings.roles.items()}
    by_label = {e.label.lower(): [e.id] for e in task.bindings.entities() if e.label}
    after: dict[str, list[str]] = {}
    for d in result.dependencies:
        after.setdefault(d.to_id, []).append(d.from_id)
    pieces = [Ask(id=s.id, goal=s.goal,
                  anchors=[i for name in s.roles for i in (by_role.get(name.lower()) or by_label.get(name.lower()) or [])],
                  after=list(dict.fromkeys(after.get(s.id, []) + s.consumes)),
                  expects_kind=s.output.entity_type or "", why=result.rationale)
              for s in result.subtasks]
    combine = {"intersect": "intersect", "union": "union"}.get(result.combine, "last")
    return Moves(pieces=pieces, combine=combine, rationale=result.rationale)
