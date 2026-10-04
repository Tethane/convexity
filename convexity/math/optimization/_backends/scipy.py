"""Local nonlinear optimization; convergence is never a global certificate."""

import numpy as np
from scipy.optimize import minimize, Bounds
from ... import _decision as v
from ..interfaces import SolveResult


def solve(p, tolerance, maxiter):
    constraints = []
    if p.inequalities is not None:
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda x: -v.array(p.inequalities(x), "inequalities"),
            }
        )
    if p.equalities is not None:
        constraints.append(
            {"type": "eq", "fun": lambda x: v.array(p.equalities(x), "equalities")}
        )
    r = minimize(
        lambda x: v.scalar(p.objective(x), "objective"),
        p.initial,
        jac=p.gradient,
        method="SLSQP",
        bounds=Bounds(p.lower_bounds, p.upper_bounds),
        constraints=constraints,
        options={"ftol": tolerance, "maxiter": maxiter},
    )
    violation = max(
        0.0, float(np.max(p.lower_bounds - r.x)), float(np.max(r.x - p.upper_bounds))
    )
    if p.inequalities is not None:
        violation = max(
            violation, float(np.max(v.array(p.inequalities(r.x), "inequalities")))
        )
    if p.equalities is not None:
        violation = max(
            violation, float(np.max(abs(v.array(p.equalities(r.x), "equalities"))))
        )
    status = "local_optimum" if r.success and violation <= tolerance * 10 else "failed"
    return SolveResult(
        r.x,
        float(r.fun),
        status,
        "scipy:SLSQP",
        violation,
        iterations=r.nit,
        message=str(r.message),
    )
