import cvxpy as cp
import numpy as np
from ...optimization._backends.cvxpy import result
from ...optimization.interfaces import SolveResult


def _constraints(x, bounds, budget):
    cons = {"budget": cp.sum(x) == budget}
    lo, hi = np.isfinite(bounds.lower_bounds), np.isfinite(bounds.upper_bounds)
    if lo.any():
        cons["lower"] = x[lo] >= bounds.lower_bounds[lo]
    if hi.any():
        cons["upper"] = x[hi] <= bounds.upper_bounds[hi]
    return cons


def minimax(a, b, bounds, budget, solver):
    x, t = cp.Variable(a.shape[1]), cp.Variable()
    cons = _constraints(x, bounds, budget)
    cons["scenarios"] = a @ x + b <= t
    p = cp.Problem(cp.Minimize(t), list(cons.values()))
    p.solve(solver=solver)
    return result(p, x, cons)


def mean_cvar(r, w, aversion, level, bounds, budget, solver, ambiguity=None):
    x, t, z = cp.Variable(r.shape[1]), cp.Variable(), cp.Variable(len(r), nonneg=True)
    cons = _constraints(x, bounds, budget)
    cons["tail"] = -r @ x - t <= z
    objective = -(w @ r) @ x + aversion * (t + w @ z / (1 - level))
    if ambiguity is not None:
        objective += ambiguity.radius * cp.norm(ambiguity.shape.T @ x)

    p = cp.Problem(cp.Minimize(objective), list(cons.values()))
    p.solve(solver=solver)
    return result(p, x, cons)


def allocation(m, c, ambiguity, aversion, bounds, budget, solver):
    x = cp.Variable(len(m))
    cons = _constraints(x, bounds, budget)
    obj = (
        -m @ x
        + ambiguity.radius * cp.norm(ambiguity.shape.T @ x)
        + 0.5 * aversion * cp.quad_form(x, cp.psd_wrap(c))
    )
    p = cp.Problem(cp.Minimize(obj), list(cons.values()))
    p.solve(solver=solver)
    return result(p, x, cons)


def kl_worst_case(x, w, radius, solver):
    if radius == 0:
        return SolveResult(w, float(w @ x), "optimal", "analytic:KL-zero", 0.0, gap=0.0)
    keep = w > 0
    q = cp.Variable(int(keep.sum()), nonneg=True)
    cons = {"mass": cp.sum(q) == 1, "KL": cp.sum(cp.rel_entr(q, w[keep])) <= radius}
    p = cp.Problem(cp.Maximize(x[keep] @ q), list(cons.values()))
    p.solve(solver=solver)
    r = result(p, q, cons)
    full = None
    if r.solution is not None:
        full = np.zeros_like(w)
        full[keep] = r.solution
    return SolveResult(
        full,
        r.objective,
        r.status,
        r.solver,
        r.max_violation,
        r.duals,
        r.gap,
        r.iterations,
    )
