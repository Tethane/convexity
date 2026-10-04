import numpy as np
import cvxpy as cp
from ... import _decision as v
from ..interfaces import TreeControlResult


def solve(tree, a, b, q, r, terminal, initial, lo, hi, state_lo, state_hi, solver):
    nodes, n, m = len(tree.parents), a.shape[0], b.shape[1]
    x, u = cp.Variable((nodes, n)), cp.Variable((nodes, m))
    cons = [x[0] == initial]
    mass, objective = tree.masses(), 0
    for i in range(nodes):
        leaf = not np.any(tree.parents == i)
        objective += mass[i] * cp.quad_form(x[i], cp.psd_wrap(terminal if leaf else q))
        if leaf:
            cons.append(u[i] == 0)
        else:
            objective += mass[i] * cp.quad_form(u[i], cp.psd_wrap(r))
            if np.isfinite(lo).any():
                k = np.isfinite(lo)
                cons.append(u[i, k] >= lo[k])
            if np.isfinite(hi).any():
                k = np.isfinite(hi)
                cons.append(u[i, k] <= hi[k])
        for bounds, lower in [(state_lo, True), (state_hi, False)]:
            k = np.isfinite(bounds)
            if k.any():
                cons.append(x[i, k] >= bounds[k] if lower else x[i, k] <= bounds[k])
        if i:
            parent = tree.parents[i]
            cons.append(x[i] == a @ x[parent] + b @ u[parent] + tree.disturbances[i])
    problem = cp.Problem(cp.Minimize(objective), cons)
    problem.solve(solver=solver)
    if problem.status != "optimal":
        raise RuntimeError(f"scenario-tree solve: {problem.status}")
    if x.value is None or u.value is None:
        raise RuntimeError("solver returned no primal solution")
    violation = max(float(np.max(abs(c.violation()))) for c in cons)
    if violation > 1e-5:
        raise RuntimeError(
            f"scenario-tree solution violates constraints by {violation}"
        )
    return TreeControlResult(
        x.value,
        u.value,
        v.scalar(problem.value, "objective"),
        violation,
        problem.solver_stats.solver_name,
    )
