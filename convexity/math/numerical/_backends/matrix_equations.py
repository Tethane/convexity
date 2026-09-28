"""SciPy matrix equation adapters with explicit equation residuals."""

import warnings

import numpy as np
from scipy import linalg

from ..interfaces import ArrayResult, Diagnostics, LyapunovEquation, RiccatiEquation


def lyapunov(problem: LyapunovEquation) -> ArrayResult:
    a, q = np.asarray(problem.a), np.asarray(problem.q)
    with warnings.catch_warnings():
        warnings.simplefilter("error", linalg.LinAlgWarning)
        if problem.time == "continuous":
            x = linalg.solve_continuous_lyapunov(a, -q)
            residual = a @ x + x @ a.conj().T + q
        else:
            x = linalg.solve_discrete_lyapunov(a, q)
            residual = a @ x @ a.conj().T - x + q
    return ArrayResult(x, Diagnostics(residual_norm=float(np.linalg.norm(residual))))


def riccati(problem: RiccatiEquation) -> ArrayResult:
    a, b, q, r = (np.asarray(value) for value in (problem.a, problem.b, problem.q, problem.r))
    if problem.time == "continuous":
        x = linalg.solve_continuous_are(a, b, q, r)
        residual = a.conj().T @ x + x @ a - x @ b @ linalg.solve(r, b.conj().T @ x) + q
    else:
        x = linalg.solve_discrete_are(a, b, q, r)
        residual = (a.conj().T @ x @ a - x + q -
                    a.conj().T @ x @ b @ linalg.solve(r + b.conj().T @ x @ b, b.conj().T @ x @ a))
    return ArrayResult(x, Diagnostics(residual_norm=float(np.linalg.norm(residual))))
