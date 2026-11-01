"""The structural decomposer: algebraic planning entry point."""

from __future__ import annotations

from kora.provenance.models import EntityRef, Spend
from kora.modules.decomposition.decomposer import DecompositionError, DecompositionResult
from kora.modules.decomposition.structural.compile import (
    compile_cut,
    compile_hypertree_dag,
    not_applicable,
)
from kora.modules.decomposition.structural.graph import (
    ConstraintGraph,
    IncidenceMatrix,
    SubQueryFailureContext,
    failure_cut_fingerprints,
    is_unknown_predicate_failure,
    plan_algebraic_decomposition,
)
from kora.modules.decomposition.structural.select import (
    CapabilityThreshold,
    Solvability,
    TieBreakPolicy,
    choose,
    score_all,
)
from kora.modules.decomposition.structural.validity import executable, validate
from kora.modules.decomposition.view import DecompositionView

__all__ = ["StructuralDecomposer"]


class StructuralDecomposer:
    def __init__(
        self,
        llm,
        *,
        solvability: Solvability | None = None,
        threshold: CapabilityThreshold | None = None,
        tie_break: TieBreakPolicy | None = None,
    ):
        self.solvability = solvability
        self.threshold = threshold or CapabilityThreshold()
        self.tie_break = tie_break or TieBreakPolicy()

    def decompose(
        self,
        request,
        view: DecompositionView | None = None,
        graph: ConstraintGraph | None = None,
        failure_context: SubQueryFailureContext | None = None,
    ) -> DecompositionResult:
        if view is None:
            raise DecompositionError("the structural decomposer needs a DecompositionView")
        spend = Spend()
        if graph is None or not graph.variables:
            raise DecompositionError("the question's structure was not read: no plan, or a plan with no variables")

        residual = graph.scoped_to(view.claims).residualise(
            domains=self._domains(view),
            resolved=[r for r in graph.resolved_requirements] + [rid for rid in view.resolved],
        )
        pruned: list[str] = []
        if is_unknown_predicate_failure(failure_context):
            pruned = [constraint.id for constraint in residual.constraints
                      if constraint.id in set(failure_context.failed_clauses)]
            if pruned:
                residual = residual.prune_constraints(pruned)
        if not residual.constraints:
            reason = (
                "all failed hyperedges were pruned after an unknown-predicate error: "
                + ", ".join(pruned)
                if pruned else "every constraint of this task is already settled"
            )
            return not_applicable(
                request.task_id, reason, spend
            )

        skeleton = residual.to_skeleton()
        forbidden = failure_cut_fingerprints(failure_context)
        matrix = IncidenceMatrix.from_skeleton(skeleton)
        candidate_fingerprints = {
            f"cut_{variable}"
            for variable in matrix.find_cut_variables(set(skeleton.target_variables))
        }
        if (failure_context is not None and candidate_fingerprints
                and candidate_fingerprints <= forbidden):
            return not_applicable(
                request.task_id,
                "all candidate vertex cuts exhausted: "
                + ", ".join(sorted(candidate_fingerprints)),
                spend,
            )
        dag = plan_algebraic_decomposition(skeleton, failure_context=failure_context)

        rows = [
            (
                cut,
                validate(cut, residual, view.open_requirement_ids or None),
                executable(cut, residual),
            )
            for cut in residual.factorisations(forbidden_cuts=forbidden)
        ]
        runnable = [cut for cut, ok, can in rows if ok.ok and can.ok]
        splits = [cut for cut in runnable if len(cut.leaves) > 1]

        if not splits:
            if len(dag.nodes) > 1 and dag.width < len(skeleton.clauses):
                return compile_hypertree_dag(
                    dag,
                    residual,
                    task_id=request.task_id,
                    spend=spend,
                    requirements={r.id: r for r in view.requirements},
                )
            why = self._why_not(rows)
            if len(dag.nodes) > 1 and dag.width >= len(skeleton.clauses):
                why = (f"residual tree-width {dag.width} cannot reduce "
                       f"{len(skeleton.clauses)} hyperedges; {why}")
            if pruned:
                why = f"pruned unknown-predicate hyperedges {pruned}; {why}"
            return not_applicable(request.task_id, why, spend)

        scored = score_all(splits, residual, self.solvability, view.statistics)
        best = choose(scored, self.threshold, self.tie_break)
        return compile_cut(
            best.cut,
            residual,
            task_id=request.task_id,
            spend=spend,
            requirements={r.id: r for r in view.requirements},
        )

    @staticmethod
    def _domains(view: DecompositionView) -> dict[str, list[EntityRef]]:
        return {f"?{role}": list(refs) for role, refs in view.bindings.items() if refs}

    @staticmethod
    def _why_not(rows) -> str:
        valid_splits = [cut for cut, ok, _ in rows if ok.ok and len(cut.leaves) > 1]
        if not valid_splits:
            return "no valid factorisation splits this goal: it is one restriction"
        blocked = [can for cut, ok, can in rows if ok.ok and len(cut.leaves) > 1 and not can.ok]
        detail = "; ".join(p for verdict in blocked for p in verdict.problems[:2])
        return (
            f"{len(valid_splits)} valid factorisation(s), none executable from here: "
            f"{detail[:220]}"
        )
