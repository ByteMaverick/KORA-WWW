"""Generalized Hypertree Decomposition (GHD) for irreducible cyclic blocks."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from kora.modules.decomposition.structural.graph import UngroundedQuerySkeleton

__all__ = [
    "DEFAULT_MAX_GHD_WIDTH",
    "HypertreeDAG",
    "HypertreeError",
    "HypertreeNode",
    "build_ghd",
    "single_bag_decomposition",
]

DEFAULT_MAX_GHD_WIDTH = 3


class HypertreeError(RuntimeError):
    """Raised when hypertree decomposition violates structural invariants."""


class HypertreeNode(BaseModel):
    """A node (bag) in the Hypertree DAG."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    assigned_clauses: list[Any] = Field(default_factory=list)
    clause_ids: list[str] = Field(default_factory=list)
    bag_variables: list[str] = Field(default_factory=list)
    covered_hyperedges: list[str] = Field(default_factory=list)
    separator_variables: list[str] = Field(default_factory=list)
    output_variable: str = ""
    child_ids: list[str] = Field(default_factory=list)
    is_root: bool = False
    origin: str = "ghd"  # "detachment" | "ghd" | "fallback"


class HypertreeDAG(BaseModel):
    """Typed DAG / rooted tree resulting from algebraic planning."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    nodes: dict[str, HypertreeNode] = Field(default_factory=dict)
    root_id: str = ""
    target_variables: list[str] = Field(default_factory=list)
    width: int = 1
    fingerprint: str = ""

    def validate_invariants(self, all_clauses: list[Any]) -> None:
        if not self.nodes:
            if all_clauses:
                raise HypertreeError("Empty HypertreeDAG for non-empty query")
            return

        if self.root_id not in self.nodes:
            raise HypertreeError(f"Root id {self.root_id!r} not in nodes")

        root = self.nodes[self.root_id]
        if not root.is_root:
            raise HypertreeError(f"Node {self.root_id} is marked root_id but is_root is False")

        visited: dict[str, int] = {}

        def check_cycle(node_id: str) -> None:
            visited[node_id] = 1
            for child_id in self.nodes[node_id].child_ids:
                if child_id not in self.nodes:
                    raise HypertreeError(f"Node {node_id} references missing child {child_id}")
                if visited.get(child_id) == 1:
                    raise HypertreeError(f"Directed cycle detected involving {node_id} -> {child_id}")
                if visited.get(child_id) == 0:
                    check_cycle(child_id)
            visited[node_id] = 2

        for nid in self.nodes:
            visited[nid] = 0
        check_cycle(self.root_id)

        assigned_cids = [cid for node in self.nodes.values() for cid in node.clause_ids]
        expected_cids = [c.id for c in all_clauses]
        if sorted(assigned_cids) != sorted(expected_cids):
            missing = set(expected_cids) - set(assigned_cids)
            invented = set(assigned_cids) - set(expected_cids)
            duplicates = [cid for cid in assigned_cids if assigned_cids.count(cid) > 1]
            raise HypertreeError(
                f"Clause coverage invariant violated: missing={missing}, "
                f"invented={invented}, duplicates={duplicates}"
            )

        clause_map = {c.id: c for c in all_clauses}
        for node in self.nodes.values():
            covered_vars = {v for hid in node.covered_hyperedges if hid in clause_map
                            for v in clause_map[hid].variables}
            bag_vset = set(node.bag_variables)
            if not bag_vset.issubset(covered_vars) and node.covered_hyperedges:
                uncovered = bag_vset - covered_vars
                raise HypertreeError(f"Node {node.id} bag variables {uncovered} not covered by {node.covered_hyperedges}")

        all_vars = sorted({v for c in all_clauses for v in c.variables})
        for var in all_vars:
            containing_nodes = [nid for nid, node in self.nodes.items() if var in node.bag_variables]
            if len(containing_nodes) > 1:
                adj: dict[str, set[str]] = {nid: set() for nid in containing_nodes}
                for nid in containing_nodes:
                    node = self.nodes[nid]
                    for cid in node.child_ids:
                        if cid in adj:
                            adj[nid].add(cid)
                            adj[cid].add(nid)
                start = containing_nodes[0]
                seen_rip = {start}
                queue = [start]
                while queue:
                    curr = queue.pop(0)
                    for nxt in adj[curr]:
                        if nxt not in seen_rip:
                            seen_rip.add(nxt)
                            queue.append(nxt)
                if len(seen_rip) != len(containing_nodes):
                    raise HypertreeError(f"Running Intersection Property violated for variable {var}")

        if self.target_variables and all_vars:
            primary_target = self.target_variables[0]
            if primary_target in all_vars and primary_target not in root.bag_variables:
                raise HypertreeError(
                    f"Root completeness violated: target {primary_target} not in root bag {root.bag_variables}"
                )


def build_ghd(
    skeleton: UngroundedQuerySkeleton,
    max_width: int = DEFAULT_MAX_GHD_WIDTH,
) -> HypertreeDAG:
    """Constructs a Generalized Hypertree Decomposition for an irreducible query block."""
    clauses = skeleton.clauses
    if not clauses:
        return HypertreeDAG(nodes={}, root_id="", target_variables=skeleton.target_variables, width=1)

    if len(clauses) == 1:
        c = clauses[0]
        node = HypertreeNode(
            id="bag_1",
            assigned_clauses=[c],
            clause_ids=[c.id],
            bag_variables=list(c.variables),
            covered_hyperedges=[c.id],
            separator_variables=[],
            output_variable=skeleton.target_variables[0] if skeleton.target_variables else (c.variables[0] if c.variables else ""),
            child_ids=[],
            is_root=True,
            origin="ghd",
        )
        dag = HypertreeDAG(
            nodes={"bag_1": node},
            root_id="bag_1",
            target_variables=skeleton.target_variables,
            width=1,
            fingerprint="ghd_single_clause",
        )
        dag.validate_invariants(clauses)
        return dag

    variables = skeleton.all_variables()
    if not variables:
        return single_bag_decomposition(skeleton, origin="ghd")

    primal_adj: dict[str, set[str]] = {v: set() for v in variables}
    for c in clauses:
        for u, v in itertools.combinations(c.variables, 2):
            primal_adj[u].add(v)
            primal_adj[v].add(u)

    candidates_orderings = [
        _min_degree_ordering(primal_adj, variables),
        _min_fill_ordering(primal_adj, variables),
        list(variables),  # original order
    ]

    best_dag: HypertreeDAG | None = None
    min_found_width = 999

    for order in candidates_orderings:
        cliques, chordal_adj = _eliminate_and_find_cliques(primal_adj, order)
        if not cliques:
            continue
        join_tree = _build_join_tree(cliques)
        bag_covers: list[list[str]] = []
        possible = True
        for bag in cliques:
            cover = _min_hyperedge_cover(bag, clauses)
            if not cover:
                possible = False
                break
            bag_covers.append(cover)
        if not possible:
            continue

        width = max(len(cover) for cover in bag_covers)
        if width < min_found_width:
            min_found_width = width
            if width <= max_width:
                candidate_dag = _assemble_ghd_dag(
                    cliques=cliques,
                    bag_covers=bag_covers,
                    join_tree_edges=join_tree,
                    skeleton=skeleton,
                    width=width,
                )
                try:
                    candidate_dag.validate_invariants(clauses)
                    best_dag = candidate_dag
                    break
                except HypertreeError:
                    continue

    if best_dag is not None:
        return best_dag

    return single_bag_decomposition(skeleton, origin="ghd_bounded_fallback")


def single_bag_decomposition(
    skeleton: UngroundedQuerySkeleton,
    origin: str = "fallback",
) -> HypertreeDAG:
    """Produces a valid single-bag HypertreeDAG covering all clauses."""
    clauses = skeleton.clauses
    all_vars = skeleton.all_variables()
    all_cids = [c.id for c in clauses]
    target = skeleton.target_variables[0] if skeleton.target_variables else (all_vars[0] if all_vars else "")

    node = HypertreeNode(
        id="bag_1",
        assigned_clauses=list(clauses),
        clause_ids=all_cids,
        bag_variables=all_vars,
        covered_hyperedges=all_cids,
        separator_variables=[],
        output_variable=target,
        child_ids=[],
        is_root=True,
        origin=origin,
    )
    dag = HypertreeDAG(
        nodes={"bag_1": node},
        root_id="bag_1",
        target_variables=skeleton.target_variables,
        width=len(clauses) or 1,
        fingerprint=f"single_bag_{len(clauses)}",
    )
    dag.validate_invariants(clauses)
    return dag


def _min_degree_ordering(adj: dict[str, set[str]], variables: list[str]) -> list[str]:
    curr_adj = {v: set(neighbors) for v, neighbors in adj.items()}
    remaining = list(variables)
    ordering: list[str] = []
    while remaining:
        best_v = min(remaining, key=lambda v: (len(curr_adj[v]), variables.index(v)))
        ordering.append(best_v)
        remaining.remove(best_v)
        nbrs = [n for n in curr_adj[best_v] if n in curr_adj]
        for u, w in itertools.combinations(nbrs, 2):
            curr_adj[u].add(w)
            curr_adj[w].add(u)
        del curr_adj[best_v]
        for n in nbrs:
            curr_adj[n].discard(best_v)
    return ordering


def _min_fill_ordering(adj: dict[str, set[str]], variables: list[str]) -> list[str]:
    curr_adj = {v: set(neighbors) for v, neighbors in adj.items()}
    remaining = list(variables)
    ordering: list[str] = []

    def count_fill(v: str) -> int:
        nbrs = [n for n in curr_adj[v] if n in curr_adj]
        fill = 0
        for u, w in itertools.combinations(nbrs, 2):
            if w not in curr_adj[u]:
                fill += 1
        return fill

    while remaining:
        best_v = min(remaining, key=lambda v: (count_fill(v), len(curr_adj[v]), variables.index(v)))
        ordering.append(best_v)
        remaining.remove(best_v)
        nbrs = [n for n in curr_adj[best_v] if n in curr_adj]
        for u, w in itertools.combinations(nbrs, 2):
            curr_adj[u].add(w)
            curr_adj[w].add(u)
        del curr_adj[best_v]
        for n in nbrs:
            curr_adj[n].discard(best_v)
    return ordering


def _eliminate_and_find_cliques(
    adj: dict[str, set[str]],
    order: list[str],
) -> tuple[list[list[str]], dict[str, set[str]]]:
    chordal_adj = {v: set(neighbors) for v, neighbors in adj.items()}
    cliques_raw: list[set[str]] = []
    active = set(order)

    for v in order:
        higher_nbrs = {n for n in chordal_adj[v] if n in active}
        clique = {v} | higher_nbrs
        cliques_raw.append(clique)
        for u, w in itertools.combinations(higher_nbrs, 2):
            chordal_adj[u].add(w)
            chordal_adj[w].add(u)
        active.remove(v)

    maximal: list[list[str]] = []
    sorted_cliques = sorted(cliques_raw, key=lambda c: len(c), reverse=True)
    for c in sorted_cliques:
        if not any(c < set(m) for m in maximal):
            maximal.append([v for v in order if v in c])

    maximal.sort(key=lambda clq: (len(clq), [order.index(v) for v in clq]))
    return maximal, chordal_adj


def _build_join_tree(cliques: list[list[str]]) -> list[tuple[int, int]]:
    k = len(cliques)
    if k <= 1:
        return []

    edges_with_weight: list[tuple[int, int, int]] = []
    for i in range(k):
        for j in range(i + 1, k):
            weight = len(set(cliques[i]) & set(cliques[j]))
            if weight > 0:
                edges_with_weight.append((weight, i, j))

    edges_with_weight.sort(key=lambda x: (-x[0], x[1], x[2]))

    parent = list(range(k))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    tree_edges: list[tuple[int, int]] = []
    for weight, u, v in edges_with_weight:
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[ru] = rv
            tree_edges.append((u, v))
            if len(tree_edges) == k - 1:
                break

    if len(tree_edges) < k - 1:
        for i in range(1, k):
            if find(0) != find(i):
                parent[find(i)] = find(0)
                tree_edges.append((0, i))

    return tree_edges


def _min_hyperedge_cover(bag: list[str], clauses: list[Any]) -> list[str]:
    bag_set = set(bag)
    if not bag_set:
        return []

    candidate_clauses = [c for c in clauses if set(c.variables) & bag_set]
    if not candidate_clauses:
        return []

    sorted_candidates = sorted(
        candidate_clauses,
        key=lambda c: (-len(set(c.variables) & bag_set), c.id),
    )

    for r in range(1, len(sorted_candidates) + 1):
        for combo in itertools.combinations(sorted_candidates, r):
            covered = set().union(*(set(c.variables) for c in combo))
            if bag_set.issubset(covered):
                return sorted(c.id for c in combo)
    return []


def _assemble_ghd_dag(
    cliques: list[list[str]],
    bag_covers: list[list[str]],
    join_tree_edges: list[tuple[int, int]],
    skeleton: UngroundedQuerySkeleton,
    width: int,
) -> HypertreeDAG:
    k = len(cliques)
    target_var = skeleton.target_variables[0] if skeleton.target_variables else ""

    adj: dict[int, list[int]] = {i: [] for i in range(k)}
    for u, v in join_tree_edges:
        adj[u].append(v)
        adj[v].append(u)

    root_idx = 0
    if target_var:
        for i, bag in enumerate(cliques):
            if target_var in bag:
                root_idx = i
                break

    parent_map: dict[int, int | None] = {root_idx: None}
    children_map: dict[int, list[int]] = {i: [] for i in range(k)}
    queue = [root_idx]
    bfs_order = [root_idx]

    while queue:
        curr = queue.pop(0)
        for nxt in sorted(adj[curr]):
            if nxt not in parent_map:
                parent_map[nxt] = curr
                children_map[curr].append(nxt)
                queue.append(nxt)
                bfs_order.append(nxt)

    assigned_clauses: dict[int, list[Any]] = {i: [] for i in range(k)}
    unassigned = list(skeleton.clauses)

    for node_idx in bfs_order:
        bag_vset = set(cliques[node_idx])
        new_unassigned = []
        for c in unassigned:
            if set(c.variables).issubset(bag_vset):
                assigned_clauses[node_idx].append(c)
            else:
                new_unassigned.append(c)
        unassigned = new_unassigned

    for c in unassigned:
        c_vset = set(c.variables)
        best_bag_idx = max(range(k), key=lambda i: (len(set(cliques[i]) & c_vset), -i))
        assigned_clauses[best_bag_idx].append(c)

    nodes: dict[str, HypertreeNode] = {}
    for i in range(k):
        node_id = f"bag_{i + 1}"
        p_idx = parent_map[i]
        bag_vars = cliques[i]
        sep_vars = sorted(set(bag_vars) & set(cliques[p_idx])) if p_idx is not None else []
        if i == root_idx:
            output_var = target_var if target_var in bag_vars else (bag_vars[0] if bag_vars else "")
        else:
            output_var = sep_vars[0] if sep_vars else (bag_vars[0] if bag_vars else "")

        child_node_ids = [f"bag_{c + 1}" for c in sorted(children_map[i])]
        nodes[node_id] = HypertreeNode(
            id=node_id,
            assigned_clauses=assigned_clauses[i],
            clause_ids=[c.id for c in assigned_clauses[i]],
            bag_variables=bag_vars,
            covered_hyperedges=bag_covers[i],
            separator_variables=sep_vars,
            output_variable=output_var,
            child_ids=child_node_ids,
            is_root=(i == root_idx),
            origin="ghd",
        )

    root_id = f"bag_{root_idx + 1}"
    return HypertreeDAG(
        nodes=nodes,
        root_id=root_id,
        target_variables=skeleton.target_variables,
        width=width,
        fingerprint=f"ghd_{k}_bags_w{width}",
    )
