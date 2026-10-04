import cvxpy as cp
import numpy as np
from ... import _decision as v
from ..interfaces import LiquidationResult


def solve(
    initial, dt, spread, impact, covariance, aversion, maximum_rate, monotone, solver
):
    t, n = len(dt), len(initial)
    q, trades = cp.Variable((t + 1, n)), cp.Variable((t, n))
    cons = [q[0] == initial, q[-1] == 0, q[1:] == q[:-1] - trades]
    cost, risk = cp.Constant(0.0), cp.Constant(0.0)
    for k in range(t):
        cost += (
            spread @ cp.abs(trades[k])
            + cp.quad_form(trades[k], cp.psd_wrap(impact)) / dt[k]
        )
        risk += aversion * dt[k] * cp.quad_form(q[k + 1], cp.psd_wrap(covariance))
        finite = np.isfinite(maximum_rate)
        if finite.any():
            cons.append(cp.abs(trades[k, finite]) <= maximum_rate[finite] * dt[k])
        if monotone:
            positive, negative, zero = initial > 0, initial < 0, initial == 0
            if positive.any():
                cons.append(trades[k, positive] >= 0)
            if negative.any():
                cons.append(trades[k, negative] <= 0)
            if zero.any():
                cons.append(trades[k, zero] == 0)
    problem = cp.Problem(cp.Minimize(cost + risk), cons)
    problem.solve(solver=solver)
    if problem.status != "optimal":
        raise RuntimeError(f"liquidation solve: {problem.status}")
    if trades.value is None or q.value is None:
        raise RuntimeError("solver returned no primal solution")
    violation = max(float(np.max(abs(c.violation()))) for c in cons)
    if violation > 1e-5:
        raise RuntimeError(f"liquidation constraint violation: {violation}")
    return LiquidationResult(
        trades.value,
        q.value,
        v.scalar(cost.value, "cost"),
        v.scalar(risk.value, "risk"),
        v.scalar(problem.value, "objective"),
        problem.solver_stats.solver_name,
        violation,
    )
