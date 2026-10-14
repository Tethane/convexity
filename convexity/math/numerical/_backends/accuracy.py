import numpy as np
from scipy.linalg import norm
from ..interfaces.accuracy import MachinePrecision, ResidualResult


def compare(problem):
    return np.asarray(
        np.isclose(
            problem.actual,
            problem.reference,
            atol=problem.tolerances.atol,
            rtol=problem.tolerances.rtol,
            equal_nan=problem.equal_nan,
        )
    )


def precision(dtype):
    info = np.finfo(dtype)
    # Preserve the queried dtype: casting extended limits to Python float can
    # turn a finite maximum into infinity or a nonzero minimum into zero.
    return MachinePrecision(
        str(np.dtype(dtype)), info.eps, info.tiny, info.smallest_subnormal, info.max
    )


def residual(problem):
    value = problem.a @ problem.x - problem.b
    absolute, scale = float(norm(value)), float(norm(problem.b))
    relative = absolute / scale if scale else (0.0 if absolute == 0 else np.inf)
    return ResidualResult(value, absolute, relative)
