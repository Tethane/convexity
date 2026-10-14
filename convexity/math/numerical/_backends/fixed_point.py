import numpy as np
from scipy import optimize
from .. import _validation as v
from ..interfaces.base import ArrayResult, Diagnostics


def solve(problem, *, method, xtol, residual_tol, maxiter):
    evaluations = 0

    def evaluate(x):
        nonlocal evaluations
        evaluations += 1
        y = v.array(problem.function(x), "fixed-point function output", real=True)
        if y.shape != np.shape(problem.x0):
            raise ValueError("fixed-point function must preserve shape")
        return y

    x = optimize.fixed_point(
        evaluate, problem.x0, xtol=xtol, maxiter=maxiter, method=method
    )
    residual = float(np.linalg.norm((evaluate(x) - x).ravel()))
    return ArrayResult(
        np.asarray(x),
        Diagnostics(
            converged=residual <= residual_tol,
            residual_norm=residual,
            evaluations=evaluations,
            message="fixed-point residual check",
        ),
    )
