"""The semantic constraint hypergraph and algebraic planning engine."""

from __future__ import annotations

import itertools
import re
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from kora.provenance.models import EntityRef

if TYPE_CHECKING:
    from kora.modules.decomposition.structural.hypertree import HypertreeDAG

__all__ = [
    "PROJECTIONS",
    "AnswerSpec",
    "Constraint",
    "ConstraintGraph",
    "Cut",
    "IncidenceMatrix",
    "Leaf",
    "SubQueryFailureContext",
    "UngroundedQuerySkeleton",
    "components_of",
    "cut_fingerprint",
    "failure_cut_fingerprints",
    "is_variable",
    "is_unknown_predicate_failure",
    "plan_algebraic_decomposition",
]

PROJECTIONS = ("identity", "count", "argmax", "argmin")

_VAR = re.compile(r"^\?[A-Za-z_][A-Za-z0-9_]*$")


def is_variable(name: str) -> bool:
    return bool(_VAR.match(name or ""))


class SubQueryFailureContext(BaseModel):
    """Context passed back when a sub-query execution fails, driving bounded replanning."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    subtask_id: str
    failed_node_id: str = ""
    failure_type: str = ""
    diagnosis: str = ""
    attempt_count: int = 1
    cut_fingerprint: str = ""
    failed_clauses: list[str] = Field(default_factory=list)
    prior_cuts: list[str] = Field(default_factory=list)


def failure_cut_fingerprints(context: SubQueryFailureContext | None) -> set[str]:
    """Every separator already known to fail, including the current one."""
    if context is None:
        return set()
    return {
        value for value in [*context.prior_cuts, context.cut_fingerprint]
        if value
    }


_UNKNOWN_PREDICATE = re.compile(
    r"(?:unknown|missing|does not exist|not (?:in|listed in|present in)(?: the)?)"
    r"[^.;\n]{0,100}(?:predicate|relation)|"
    r"(?:predicate|relation)[^.;\n]{0,100}"
    r"(?:unknown|missing|does not exist|not (?:in|listed in|present in)(?: the)?)",
    re.IGNORECASE,
)


def is_unknown_predicate_failure(context: SubQueryFailureContext | None) -> bool:
    """Whether execution identified a schema-absent predicate/relation."""
    return bool(context and _UNKNOWN_PREDICATE.search(context.diagnosis or ""))


class Constraint(BaseModel):
    """One semantic restriction, over variables and constants."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    source_requirement: str = ""
    arguments: list[str] = Field(default_factory=list)
    template: str = ""

    @property
    def variables(self) -> list[str]:
        seen: set[str] = set()
        res: list[str] = []
        for a in self.arguments:
            if is_variable(a) and a not in seen:
                seen.add(a)
                res.append(a)
        return res

    @property
    def constants(self) -> list[str]:
        seen: set[str] = set()
        res: list[str] = []
        for a in self.arguments:
            if not is_variable(a) and a not in seen:
                seen.add(a)
                res.append(a)
        return res

    def render(self) -> str:
        return self.template or f"{self.id}({', '.join(self.arguments)})"


class AnswerSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    projection: str = "identity"
    variable: str = ""
    key: str = ""


class UngroundedQuerySkeleton(BaseModel):
    """Normalized algebraic query representation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    clause_ids: list[str] = Field(default_factory=list)
    clauses: list[Constraint] = Field(default_factory=list)
    target_variables: list[str] = Field(default_factory=list)
    grounded: dict[str, EntityRef] = Field(default_factory=dict)
    literals: list[str] = Field(default_factory=list)
    domains: dict[str, list[EntityRef]] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def all_variables(self) -> list[str]:
        seen: set[str] = set()
        res: list[str] = []
        for c in self.clauses:
            for v in c.variables:
                if v not in seen:
                    seen.add(v)
                    res.append(v)
        return res

    def sub_skeleton(self, clause_subset: list[str], target_variables: list[str]) -> UngroundedQuerySkeleton:
        subset_set = set(clause_subset)
        kept_clauses = [c for c in self.clauses if c.id in subset_set]
        kept_vars = {v for c in kept_clauses for v in c.variables}
        return UngroundedQuerySkeleton(
            clause_ids=[c.id for c in kept_clauses],
            clauses=kept_clauses,
            target_variables=[v for v in target_variables if v in kept_vars] or (list(kept_vars)[:1] if kept_vars else []),
            grounded=dict(self.grounded),
            literals=list(self.literals),
            domains={v: refs for v, refs in self.domains.items() if v in kept_vars},
            metadata=dict(self.metadata),
        )


class IncidenceMatrix(BaseModel):
    """Binary incidence matrix B in {0, 1}^(|V| x |E|)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    var_names: list[str] = Field(default_factory=list)
    clause_names: list[str] = Field(default_factory=list)
    matrix: list[list[int]] = Field(default_factory=list)
    var_to_idx: dict[str, int] = Field(default_factory=dict)
    clause_to_idx: dict[str, int] = Field(default_factory=dict)

    @classmethod
    def from_clauses(cls, clauses: list[Constraint], var_order: list[str] | None = None) -> IncidenceMatrix:
        clause_names = [c.id for c in clauses]
        if var_order is None:
            seen_vars: set[str] = set()
            var_names = []
            for c in clauses:
                for v in c.variables:
                    if v not in seen_vars:
                        seen_vars.add(v)
                        var_names.append(v)
        else:
            var_names = list(var_order)

        var_to_idx = {v: i for i, v in enumerate(var_names)}
        clause_to_idx = {c: j for j, c in enumerate(clause_names)}

        num_vars = len(var_names)
        num_clauses = len(clause_names)
        mat = [[0] * num_clauses for _ in range(num_vars)]

        for j, c in enumerate(clauses):
            for v in c.variables:
                if v in var_to_idx:
                    mat[var_to_idx[v]][j] = 1

        return cls(
            var_names=var_names,
            clause_names=clause_names,
            matrix=mat,
            var_to_idx=var_to_idx,
            clause_to_idx=clause_to_idx,
        )

    @classmethod
    def from_skeleton(cls, skeleton: UngroundedQuerySkeleton) -> IncidenceMatrix:
        return cls.from_clauses(skeleton.clauses, var_order=skeleton.all_variables())

    def remove_variable(self, v: str) -> tuple[IncidenceMatrix, list[str]]:
        if v not in self.var_to_idx:
            return self, list(self.var_names)
        v_idx = self.var_to_idx[v]
        new_vars = [var for i, var in enumerate(self.var_names) if i != v_idx]
        new_mat = [row for i, row in enumerate(self.matrix) if i != v_idx]
        var_to_idx = {var: i for i, var in enumerate(new_vars)}
        return (
            IncidenceMatrix(
                var_names=new_vars,
                clause_names=list(self.clause_names),
                matrix=new_mat,
                var_to_idx=var_to_idx,
                clause_to_idx=dict(self.clause_to_idx),
            ),
            new_vars,
        )

    def clause_adjacency(self, without_var: str | None = None) -> list[list[int]]:
        mat = self.matrix
        if without_var is not None and without_var in self.var_to_idx:
            b_sub, _ = self.remove_variable(without_var)
            mat = b_sub.matrix

        num_vars = len(mat)
        num_clauses = len(self.clause_names)
        adj = [[0] * num_clauses for _ in range(num_clauses)]

        for i in range(num_clauses):
            for j in range(num_clauses):
                if i == j:
                    adj[i][j] = 0
                else:
                    dot = sum(mat[r][i] * mat[r][j] for r in range(num_vars))
                    adj[i][j] = 1 if dot > 0 else 0
        return adj

    def find_connected_clause_components(self, without_var: str | None = None) -> list[list[str]]:
        num_clauses = len(self.clause_names)
        if num_clauses == 0:
            return []
        if num_clauses == 1:
            return [[self.clause_names[0]]]

        adj = self.clause_adjacency(without_var=without_var)
        visited = [False] * num_clauses
        components: list[list[str]] = []

        for start in range(num_clauses):
            if not visited[start]:
                comp: list[str] = []
                queue = [start]
                visited[start] = True
                while queue:
                    curr = queue.pop(0)
                    comp.append(self.clause_names[curr])
                    for nxt in range(num_clauses):
                        if adj[curr][nxt] > 0 and not visited[nxt]:
                            visited[nxt] = True
                            queue.append(nxt)
                components.append(comp)
        return components

    def find_cut_variables(self, target_variables: set[str]) -> list[str]:
        valid: list[str] = []
        for v in self.var_names:
            if v in target_variables:
                continue
            comps = self.find_connected_clause_components(without_var=v)
            if len(comps) >= 2:
                valid.append(v)
        return valid

    def detach_components(
        self,
        cut_var: str,
        target_variables: set[str],
        clauses: list[Constraint],
    ) -> tuple[list[list[str]], list[str]] | None:
        comps = self.find_connected_clause_components(without_var=cut_var)
        if len(comps) < 2:
            return None

        clause_map = {c.id: c for c in clauses}
        target_bearing: list[list[str]] = []
        detached: list[list[str]] = []

        for comp in comps:
            comp_vars = {v for cid in comp for v in clause_map[cid].variables}
            if comp_vars & target_variables:
                target_bearing.append(comp)
            else:
                detached.append(comp)

        if not target_bearing and detached:
            target_bearing.append(detached.pop())

        if len(target_bearing) != 1 or not detached:
            return None

        rem_clauses = set().union(*target_bearing, *(d for d in detached))
        for comp in detached:
            comp_vars = {v for cid in comp for v in clause_map[cid].variables}
            other_clauses = rem_clauses - set(comp)
            other_vars = {v for cid in other_clauses for v in clause_map[cid].variables}
            boundary = comp_vars & other_vars
            if not boundary.issubset({cut_var}):
                return None

        return detached, target_bearing[0]


class Leaf(BaseModel):
    """One child of a factorisation: some constraints, and what it produces."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    constraint_ids: list[str] = Field(default_factory=list)
    inputs: list[str] = Field(default_factory=list)
    output: str = ""
    satisfies: list[str] = Field(default_factory=list)


class Cut(BaseModel):
    """One factorisation of a hypergraph."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    leaves: list[Leaf] = Field(default_factory=list)
    edges: list[tuple[str, str]] = Field(default_factory=list)
    recomposition: str = "identity"
    kind: str = ""  # "whole" | "chain" | "intersection" | "narrow" | "ghd" | "detachment"

    @property
    def depth(self) -> int:
        after: dict[str, list[str]] = {}
        for before, then in self.edges:
            after.setdefault(before, []).append(then)
        best = 1

        def walk(node: str, seen: frozenset[str]) -> int:
            if node in seen:
                return 0
            return 1 + max((walk(n, seen | {node}) for n in after.get(node, [])), default=0)

        for leaf in self.leaves:
            best = max(best, walk(leaf.id, frozenset()))
        return best

    def runs_before(self) -> dict[str, set[str]]:
        before: dict[str, set[str]] = {leaf.id: set() for leaf in self.leaves}
        for _ in self.edges:
            for producer, consumer in self.edges:
                if consumer in before:
                    before[consumer] |= {producer} | before.get(producer, set())
        return before


def cut_fingerprint(cut: Cut) -> str:
    """Stable identity of a separator, independent of cut representation."""
    leaves = {leaf.id: leaf for leaf in cut.leaves}
    for producer, _consumer in cut.edges:
        output = leaves.get(producer).output if producer in leaves else ""
        if output.startswith("?"):
            return f"cut_{output}"
    groups = ["+".join(sorted(leaf.constraint_ids)) for leaf in cut.leaves]
    return f"{cut.kind or 'cut'}:{'|'.join(groups)}"


class ConstraintGraph(BaseModel):
    """H = (V, E, B, A)."""

    model_config = ConfigDict(extra="forbid")

    variables: list[str] = Field(default_factory=list)
    constraints: list[Constraint] = Field(default_factory=list)
    domains: dict[str, list[EntityRef]] = Field(default_factory=dict)
    grounded: dict[str, EntityRef] = Field(default_factory=dict)
    literals: list[str] = Field(default_factory=list)
    answer: AnswerSpec = Field(default_factory=AnswerSpec)
    resolved_requirements: list[str] = Field(default_factory=list)

    def constraint(self, constraint_id: str) -> Constraint:
        for found in self.constraints:
            if found.id == constraint_id:
                return found
        raise KeyError(constraint_id)

    def over(self, variable: str) -> list[Constraint]:
        return [c for c in self.constraints if variable in c.variables]

    def separators(self) -> list[str]:
        produced, consumed = set(), set()
        for constraint in self.constraints:
            for v in constraint.variables:
                (produced if v not in produced else consumed).add(v)
        shared = produced & consumed
        return [v for v in self.variables if v in shared and v != self.answer.variable]

    def scoped_to(self, requirements: list[str] | None) -> ConstraintGraph:
        wanted = set(requirements or [])
        if not wanted:
            return self
        kept = [c for c in self.constraints if c.source_requirement in wanted]
        if not kept or len(kept) == len(self.constraints):
            return self
        alive = {v for c in kept for v in c.variables}
        return ConstraintGraph(
            variables=[v for v in self.variables if v in alive],
            constraints=kept,
            domains={v: refs for v, refs in self.domains.items() if v in alive},
            grounded=dict(self.grounded),
            literals=list(self.literals),
            answer=self.answer,
            resolved_requirements=list(self.resolved_requirements),
        )

    def residualise(
        self,
        domains: dict[str, list[EntityRef]] | None = None,
        resolved: list[str] | None = None,
    ) -> ConstraintGraph:
        domains = {**self.domains, **(domains or {})}
        closed = set(self.resolved_requirements) | set(resolved or [])
        kept = [c for c in self.constraints if c.source_requirement not in closed]

        grounded = dict(self.grounded)
        singletons = {v: refs[0] for v, refs in domains.items() if len(refs) == 1}
        rewritten: list[Constraint] = []
        for constraint in kept:
            if not any(v in singletons for v in constraint.variables):
                rewritten.append(constraint)
                continue
            arguments, template = [], constraint.template
            for argument in constraint.arguments:
                if argument in singletons:
                    entity = singletons[argument]
                    key = entity.label or entity.id
                    grounded[key] = entity
                    arguments.append(key)
                    template = template.replace(f"{{{argument}}}", f"{{{key}}}")
                else:
                    arguments.append(argument)
            rewritten.append(
                constraint.model_copy(update={"arguments": arguments, "template": template})
            )

        return ConstraintGraph(
            variables=[v for v in self.variables if v not in singletons],
            constraints=rewritten,
            domains={v: refs for v, refs in domains.items() if len(refs) > 1},
            grounded=grounded,
            literals=list(self.literals),
            answer=self.answer,
            resolved_requirements=sorted(closed),
        )

    def prune_constraints(self, constraint_ids: list[str]) -> ConstraintGraph:
        rejected = set(constraint_ids)
        kept = [constraint for constraint in self.constraints
                if constraint.id not in rejected]
        alive = {variable for constraint in kept for variable in constraint.variables}
        return ConstraintGraph(
            variables=[variable for variable in self.variables if variable in alive],
            constraints=kept,
            domains={variable: refs for variable, refs in self.domains.items()
                     if variable in alive},
            grounded=dict(self.grounded),
            literals=list(self.literals),
            answer=self.answer,
            resolved_requirements=list(self.resolved_requirements),
        )

    def to_skeleton(self) -> UngroundedQuerySkeleton:
        targets = [self.answer.variable] if self.answer.variable else []
        return UngroundedQuerySkeleton(
            clause_ids=[c.id for c in self.constraints],
            clauses=list(self.constraints),
            target_variables=targets,
            grounded=dict(self.grounded),
            literals=list(self.literals),
            domains=dict(self.domains),
        )

    def factorisations(self, forbidden_cuts: set[str] | None = None) -> list[Cut]:
        forbidden_cuts = forbidden_cuts or set()
        found: list[Cut] = []
        for build in (self._whole, self._intersection, self._chain, self._narrow, self._algebraic_cut, self._per_constraint):
            cut = build()
            if (cut is not None
                    and cut_fingerprint(cut) not in forbidden_cuts
                    and not any(_same(cut, seen) for seen in found)):
                found.append(cut)
        return found

    def _algebraic_cut(self) -> Cut | None:
        if len(self.constraints) < 2:
            return None
        skeleton = self.to_skeleton()
        try:
            dag = plan_algebraic_decomposition(skeleton)
        except Exception:
            return None
        if len(dag.nodes) <= 1:
            return None

        leaves: list[Leaf] = []
        edges: list[tuple[str, str]] = []
        node_to_leaf: dict[str, str] = {}

        for i, (nid, node) in enumerate(dag.nodes.items(), start=1):
            lid = f"l{i}"
            node_to_leaf[nid] = lid
            inputs = sorted(
                set(node.separator_variables)
                | {a for c in node.assigned_clauses for a in c.constants if a in self.grounded or a in self.literals}
            )
            satisfies = sorted(
                {c.source_requirement for c in node.assigned_clauses if c.source_requirement}
            )
            leaves.append(
                Leaf(
                    id=lid,
                    constraint_ids=list(node.clause_ids),
                    inputs=inputs,
                    output=node.output_variable,
                    satisfies=satisfies,
                )
            )

        for nid, node in dag.nodes.items():
            for cid in node.child_ids:
                edges.append((node_to_leaf[cid], node_to_leaf[nid]))

        recomposition = "intersect" if all(l.output == self.answer.variable for l in leaves) else "project"
        return Cut(
            leaves=leaves,
            edges=edges,
            recomposition=recomposition,
            kind="detachment" if any(n.origin == "detachment" for n in dag.nodes.values()) else "ghd",
        )

    def _whole(self) -> Cut | None:
        if not self.constraints:
            return None
        return Cut(
            kind="whole",
            recomposition="identity",
            leaves=[
                Leaf(
                    id="l1",
                    constraint_ids=[c.id for c in self.constraints],
                    inputs=sorted({a for c in self.constraints for a in c.constants}),
                    output=self.answer.variable,
                    satisfies=sorted(
                        {c.source_requirement for c in self.constraints if c.source_requirement}
                    ),
                )
            ],
        )

    def _intersection(self) -> Cut | None:
        target = self.answer.variable
        if not target:
            return None
        on_target = [c for c in self.constraints if target in c.variables]
        if len(on_target) < 2 or len(on_target) != len(self.constraints):
            return None
        available = set(self.grounded) | {v for v, d in self.domains.items() if d}
        leaves = []
        for n, constraint in enumerate(on_target, start=1):
            others = [v for v in constraint.variables if v != target]
            if any(v not in available for v in others):
                return None
            leaves.append(
                Leaf(
                    id=f"l{n}",
                    constraint_ids=[constraint.id],
                    inputs=constraint.constants,
                    output=target,
                    satisfies=[constraint.source_requirement] if constraint.source_requirement else [],
                )
            )
        return Cut(kind="intersection", recomposition="intersect", leaves=leaves)

    def _narrow(self) -> Cut | None:
        target = self.answer.variable
        if not target:
            return None
        on_target = [c for c in self.constraints if target in c.variables]
        if len(on_target) < 2 or len(on_target) != len(self.constraints):
            return None
        available = set(self.grounded) | {v for v, d in self.domains.items() if d}
        now = [c for c in on_target if any(a in available for a in c.arguments)]
        later = [c for c in on_target if c not in now]
        if not now or not later:
            return None
        return Cut(
            kind="narrow",
            recomposition="select",
            leaves=[
                Leaf(
                    id="l1",
                    constraint_ids=[c.id for c in now],
                    inputs=sorted({a for c in now for a in c.constants if a in available}),
                    output=target,
                    satisfies=sorted({c.source_requirement for c in now if c.source_requirement}),
                ),
                Leaf(
                    id="l2",
                    constraint_ids=[c.id for c in later],
                    inputs=sorted({target} | {a for c in later for a in c.constants}),
                    output=target,
                    satisfies=sorted({c.source_requirement for c in later if c.source_requirement}),
                ),
            ],
            edges=[("l1", "l2")],
        )

    def _chain(self) -> Cut | None:
        order = self.separators()
        if not order:
            return None
        leaves, edges, previous = [], [], None
        used: set[str] = set()
        available = set(self.grounded) | {v for v, d in self.domains.items() if d}
        for n, variable in enumerate([*order, self.answer.variable], start=1):
            if not variable:
                continue
            producing = [
                c for c in self.constraints
                if variable in c.variables and c.id not in used
                and all(v in available or v == variable for v in c.variables)
            ]
            if not producing:
                continue
            used.update(c.id for c in producing)
            leaf = Leaf(
                id=f"l{n}",
                constraint_ids=[c.id for c in producing],
                inputs=sorted(
                    ({v for c in producing for v in c.variables} - {variable})
                    | {a for c in producing for a in c.constants}
                ),
                output=variable,
                satisfies=sorted({c.source_requirement for c in producing if c.source_requirement}),
            )
            leaves.append(leaf)
            available.add(variable)
            if previous is not None:
                edges.append((previous, leaf.id))
            previous = leaf.id
        if len(leaves) < 2 or len(used) != len(self.constraints):
            return None
        return Cut(kind="chain", recomposition="project", leaves=leaves, edges=edges)

    def _per_constraint(self) -> Cut | None:
        if len(self.constraints) < 2:
            return None
        leaves, edges = [], []
        produced: dict[str, str] = {}
        for n, constraint in enumerate(self.constraints, start=1):
            target = next(
                (v for v in constraint.variables if v not in produced),
                constraint.variables[-1] if constraint.variables else "",
            )
            leaf = Leaf(
                id=f"l{n}",
                constraint_ids=[constraint.id],
                inputs=[v for v in constraint.variables if v != target] + constraint.constants,
                output=target,
                satisfies=[constraint.source_requirement] if constraint.source_requirement else [],
            )
            for variable in leaf.inputs:
                if variable in produced:
                    edges.append((produced[variable], leaf.id))
            produced.setdefault(target, leaf.id)
            leaves.append(leaf)
        recomposition = "intersect" if all(
            leaf.output == self.answer.variable for leaf in leaves
        ) else "project"
        return Cut(
            kind="per-constraint",
            recomposition=recomposition,
            leaves=leaves,
            edges=edges,
        )


def plan_algebraic_decomposition(
    skeleton: UngroundedQuerySkeleton,
    failure_context: SubQueryFailureContext | None = None,
) -> HypertreeDAG:
    """Primary algebraic planner implementing Wong-Youssefi Detachment and GHD."""
    from kora.modules.decomposition.structural.hypertree import (
        HypertreeDAG,
        HypertreeNode,
        build_ghd,
    )

    clauses = skeleton.clauses
    if not clauses:
        return HypertreeDAG(nodes={}, root_id="", target_variables=skeleton.target_variables, width=1)

    if len(clauses) == 1:
        c = clauses[0]
        node = HypertreeNode(
            id="node_1",
            assigned_clauses=[c],
            clause_ids=[c.id],
            bag_variables=list(c.variables),
            covered_hyperedges=[c.id],
            separator_variables=[],
            output_variable=skeleton.target_variables[0] if skeleton.target_variables else (c.variables[0] if c.variables else ""),
            child_ids=[],
            is_root=True,
            origin="detachment",
        )
        return HypertreeDAG(
            nodes={"node_1": node},
            root_id="node_1",
            target_variables=skeleton.target_variables,
            width=1,
            fingerprint="detachment_single",
        )

    forbidden_cuts = failure_cut_fingerprints(failure_context)

    matrix = IncidenceMatrix.from_skeleton(skeleton)
    target_set = set(skeleton.target_variables)
    cut_candidates = matrix.find_cut_variables(target_set)

    chosen_cut: str | None = None
    detached_result: tuple[list[list[str]], list[str]] | None = None

    for v in cut_candidates:
        if f"cut_{v}" in forbidden_cuts:
            continue
        res = matrix.detach_components(v, target_set, clauses)
        if res is not None:
            chosen_cut = v
            detached_result = res
            break

    if chosen_cut is None or detached_result is None:
        return build_ghd(skeleton)

    detached_comps, target_comp = detached_result
    nodes: dict[str, HypertreeNode] = {}
    child_node_ids: list[str] = []

    for i, comp_cids in enumerate(detached_comps, start=1):
        sub_skel = skeleton.sub_skeleton(comp_cids, target_variables=[chosen_cut])
        sub_dag = plan_algebraic_decomposition(sub_skel, failure_context=failure_context)
        for nid, node in sub_dag.nodes.items():
            prefixed_id = f"d{i}_{nid}"
            prefixed_children = [f"d{i}_{cid}" for cid in node.child_ids]
            nodes[prefixed_id] = HypertreeNode(
                id=prefixed_id,
                assigned_clauses=node.assigned_clauses,
                clause_ids=node.clause_ids,
                bag_variables=node.bag_variables,
                covered_hyperedges=node.covered_hyperedges,
                separator_variables=node.separator_variables or ([chosen_cut] if node.is_root else []),
                output_variable=node.output_variable,
                child_ids=prefixed_children,
                is_root=False,
                origin="detachment",
            )
        child_node_ids.append(f"d{i}_{sub_dag.root_id}")

    tgt_skel = skeleton.sub_skeleton(target_comp, target_variables=skeleton.target_variables)
    tgt_dag = plan_algebraic_decomposition(tgt_skel, failure_context=failure_context)

    attach_target_node_id = None
    for nid, node in tgt_dag.nodes.items():
        if chosen_cut in node.bag_variables:
            attach_target_node_id = f"t_{nid}"
            break
    if attach_target_node_id is None:
        attach_target_node_id = f"t_{tgt_dag.root_id}"

    for nid, node in tgt_dag.nodes.items():
        prefixed_id = f"t_{nid}"
        prefixed_children = [f"t_{cid}" for cid in node.child_ids]
        if prefixed_id == attach_target_node_id:
            prefixed_children.extend(child_node_ids)
        nodes[prefixed_id] = HypertreeNode(
            id=prefixed_id,
            assigned_clauses=node.assigned_clauses,
            clause_ids=node.clause_ids,
            bag_variables=node.bag_variables,
            covered_hyperedges=node.covered_hyperedges,
            separator_variables=node.separator_variables,
            output_variable=node.output_variable,
            child_ids=prefixed_children,
            is_root=node.is_root,
            origin="detachment",
        )

    root_id = f"t_{tgt_dag.root_id}"
    dag = HypertreeDAG(
        nodes=nodes,
        root_id=root_id,
        target_variables=skeleton.target_variables,
        width=max(tgt_dag.width, 1),
        fingerprint=f"detachment_cut_{chosen_cut}",
    )
    dag.validate_invariants(clauses)
    return dag


def components_of(constraints: list[Constraint]) -> list[list[str]]:
    """Constraint ids grouped by shared variables."""
    parent = {c.id: c.id for c in constraints}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in itertools.combinations(constraints, 2):
        if set(a.variables) & set(b.variables):
            ra, rb = find(a.id), find(b.id)
            if ra != rb:
                parent[ra] = rb
    groups: dict[str, list[str]] = {}
    for c in constraints:
        groups.setdefault(find(c.id), []).append(c.id)
    return list(groups.values())


def _same(a: Cut, b: Cut) -> bool:
    return (
        sorted(tuple(sorted(l.constraint_ids)) for l in a.leaves)
        == sorted(tuple(sorted(l.constraint_ids)) for l in b.leaves)
        and sorted(a.edges) == sorted(b.edges)
    )
