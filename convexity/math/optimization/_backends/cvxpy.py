"""CVXPY construction and solver diagnostics, hidden from ordinary callers."""

import numpy as np
import cvxpy as cp
from dataclasses import replace
from ..interfaces import SolveResult


def result(problem, variables, named, *, tolerance=1e-6):
    value = variables.value
    status = str(problem.status)
    violation = None
    if value is not None:
        value = np.asarray(value).ravel()
        violation = max(
            [0.0] + [float(np.max(np.abs(c.violation()))) for c in problem.constraints]
        )
        if status == "optimal" and violation > tolerance:
            status = "infeasible_solution"
    stats = problem.solver_stats
    extra = stats.extra_stats
    gap = (
        extra.get("mip_gap")
        if isinstance(extra, dict)
        else getattr(extra, "mip_gap", None)
    )
    duals = {
        name: c.dual_value for name, c in named.items() if c.dual_value is not None
    }
    objective = (
        float(problem.value)
        if value is not None
        and problem.value is not None
        and np.isfinite(problem.value)
        else None
    )
    return SolveResult(
        value,
        objective,
        status,
        stats.solver_name,
        violation,
        duals,
        gap,
        stats.num_iters,
    )


def solve(problem, solver, options, tolerance):
    n = len(problem.linear)
    discrete = any(d != "continuous" for d in problem.domains)
    # Scalar attributes avoid CVXPY's version-dependent sparse integer-index
    # convention and work with mixed domains on CVXPY 1.6 as well as 1.9.
    x = (
        cp.hstack(
            [
                cp.Variable(integer=d == "integer", boolean=d == "binary")
                for d in problem.domains
            ]
        )
        if discrete
        else cp.Variable(n)
    )
    named = {}
    if problem.inequality is not None:
        named["inequality"] = problem.inequality @ x <= problem.upper
    if problem.equality is not None:
        named["equality"] = problem.equality @ x == problem.target
    lo, hi = np.isfinite(problem.lower_bounds), np.isfinite(problem.upper_bounds)
    if lo.any():
        named["lower_bounds"] = x[lo] >= problem.lower_bounds[lo]
    if hi.any():
        named["upper_bounds"] = x[hi] <= problem.upper_bounds[hi]
    for i, k in enumerate(problem.cones):
        named[f"cone_{i}"] = cp.norm(k.matrix @ x + k.offset) <= k.linear @ x + k.bound
    objective = problem.linear @ x
    if np.any(problem.quadratic):
        objective += 0.5 * cp.quad_form(x, cp.psd_wrap(problem.quadratic))
    p = cp.Problem(cp.Minimize(objective), list(named.values()))
    # Clarabel's interior-point tolerances are suitable for the default public
    # accuracy contract; CVXPY otherwise selects OSQP with looser QP defaults.
    selected = solver or (None if discrete else "CLARABEL")
    p.solve(solver=selected, **options)
    r = result(p, x, named, tolerance=tolerance)
    if discrete and r.solution is not None:
        violations = [r.max_violation or 0.0]
        for i, domain in enumerate(problem.domains):
            if domain != "continuous":
                violations.append(abs(r.solution[i] - round(r.solution[i])))
            if domain == "binary":
                violations.extend([-r.solution[i], r.solution[i] - 1])
        violation = float(max(violations))
        r = replace(
            r,
            max_violation=violation,
            status="infeasible_solution"
            if r.status == "optimal" and violation > tolerance
            else r.status,
        )
    return r
