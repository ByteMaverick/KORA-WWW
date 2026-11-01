"""The plan's operations among the cut's pieces."""

from __future__ import annotations

from kora.provenance.models import EntityRef, Plan
from kora.modules.decomposition.structural.graph import ConstraintGraph, is_variable
from kora.modules.decomposition.view import DecompositionView
from kora.modules.reduction.models import Operate

__all__ = ["placed", "relations"]

NAMED = "named"
PROJECTED = ("argmax", "argmin", "count")


def relations(plan: Plan | None, graph: ConstraintGraph | None,
              view: DecompositionView) -> tuple[ConstraintGraph | None, DecompositionView]:
    """What the cut factorises: the plan's constraints without an operation, and the claims without them."""
    ops = {c.id for c in plan.constraints if c.op} if plan is not None else set()
    if not ops or graph is None:
        return graph, view
    view = view.model_copy(update={"requirements": [r for r in view.requirements if r.id not in ops],
                                   "claims": [c for c in view.claims if c not in ops]})
    return graph.prune_constraints(sorted(ops)), view


def placed(moves, plan: Plan | None, outputs: dict[str, str], named: dict[str, EntityRef],
           claims: list[str] = ()):
    """The cut's pieces with the operations the task claims, the projection last."""
    from kora.modules.controller.moves import Ask, Moves
    if plan is None:
        return moves
    kinds = {v.name: v.kind for v in plan.variables}
    pieces = [p.model_copy(update={"expects_kind": kinds.get(outputs.get(p.id, ""), "")})
              if isinstance(p, Ask) and not p.expects_kind else p for p in moves.pieces]
    holder = {outputs[p.id]: p.id for p in pieces if isinstance(p, Ask) and p.id in outputs}
    whole = not claims or {c.id for c in plan.constraints} <= set(claims)
    ops = [(c.over, Operate(id=c.id, op=c.op, prop=c.about, cmp=c.cmp, value=c.value, expects_kind=kinds.get(c.over, ""),
                            named=[named[a] for a in c.args if a in named] if c.over == NAMED else [],
                            why=f"{c.id}: {c.op} {c.about}".strip()))
           for c in plan.constraints if c.op and (whole or c.id in claims)]
    if whole and (projected := _projection(plan)) is not None:
        ops.append(projected)
    combine, skipped = moves.combine, []
    answering = [p.id for p in pieces if isinstance(p, Ask) and outputs.get(p.id) == plan.answer.variable]
    if combine == "intersect" and len(answering) > 1 and any(v == plan.answer.variable for v, _ in ops):
        meet = Operate(id=f"{answering[-1]}.meet", op="intersect", over=answering, after=answering,
                       why="the leaves meet before the operation")
        pieces, holder[plan.answer.variable], combine = [*pieces, meet], meet.id, "last"
    for variable, op in ops:
        if op.named:
            pieces.append(op)
            continue
        source = holder.get(variable)
        if source is None:
            skipped.append(op.id or op.op)
            continue
        op = op.model_copy(update={"id": op.id or f"{source}.{op.op}", "over": [source], "after": [source]})
        at = next(n for n, p in enumerate(pieces) if p.id == source) + 1
        later = [p.model_copy(update={"after": [op.id if a == source else a for a in p.after]})
                 if isinstance(p, Ask) and source in p.after else p for p in pieces[at:]]
        pieces, holder[variable] = [*pieces[:at], op, *later], op.id
    rationale = moves.rationale + (f"; no piece holds the set of {', '.join(skipped)}" if skipped else "")
    return Moves(pieces=pieces, combine=combine, rationale=rationale)


def _projection(plan: Plan) -> tuple[str, Operate] | None:
    projection, key = plan.answer.projection, plan.answer.key.strip()
    if projection not in PROJECTED or any(c.op == projection for c in plan.constraints):
        return None
    holding = next((c for c in plan.constraints if key.startswith("?") and key in c.args), None)
    ordered = next((a for a in (holding.args if holding else []) if is_variable(a) and a != key), "")
    variable = ordered or plan.answer.variable
    return variable, Operate(op=projection, prop="" if projection == "count" else key.lstrip("?").replace("_", " "),
                             expects_kind=plan.answer.kind if variable == plan.answer.variable else "",
                             why=f"the plan projects {projection}({key or variable})")
