"""NumPy/SciPy dense factorizations and matrix analysis."""
import numpy as np
from typing import Any, cast
from scipy import linalg

from ..interfaces.base import ArrayResult, Diagnostics
from ..interfaces.linear_algebra import (
    ConditionResult, FactorizationResult, GeneralizedEigenResult, LeastSquaresResult, LogDetResult,
)


def least_squares(problem, *, rcond):
    x, _, rank, singular = linalg.lstsq(problem.a, problem.b, cond=rcond, lapack_driver="gelsd")
    residual = float(linalg.norm(np.asarray(problem.a) @ x - problem.b))
    return LeastSquaresResult(x, rank, singular, Diagnostics(residual_norm=residual))


def generalized_eigen(problem):
    if problem.hermitian:
        values, vectors = linalg.eigh(problem.a, problem.b)
        return GeneralizedEigenResult(values, np.ones_like(values), vectors)
    values, vectors = linalg.eig(problem.a, problem.b, homogeneous_eigvals=True)
    return GeneralizedEigenResult(values[0], values[1], vectors)


def factorize(problem):
    a = np.asarray(problem.a)
    if problem.kind == "qr":
        if problem.pivoting:
            q, r, p = cast(tuple[Any, Any, Any], linalg.qr(a, mode="economic", pivoting=True))
        else:
            q, r = cast(tuple[Any, Any], linalg.qr(a, mode="economic"))
            p = np.arange(a.shape[1])
        return FactorizationResult({"q": q, "r": r}, p)
    if problem.kind == "lu":
        p, l, u = cast(tuple[Any, Any, Any], linalg.lu(a))
        return FactorizationResult({"p": p, "l": l, "u": u})
    if problem.kind == "cholesky":
        return FactorizationResult({"l": linalg.cholesky(a, lower=True)})
    t, z = cast(tuple[Any, Any], linalg.schur(a, output="complex" if np.iscomplexobj(a) else "real"))
    return FactorizationResult({"t": t, "z": z})


def matrix_property(problem):
    a, op = np.asarray(problem.a), problem.operation
    if op == "rank":
        return ArrayResult(np.linalg.matrix_rank(a, tol=problem.tolerance))
    if op == "condition":
        return ConditionResult(float(np.linalg.cond(a, p=problem.order)))
    if op == "norm":
        return ArrayResult(linalg.norm(a, ord=problem.order))
    if op == "pinv":
        return ArrayResult(linalg.pinv(a, rtol=problem.tolerance))
    if op == "det":
        return ArrayResult(linalg.det(a))
    if op == "slogdet":
        sign, value = np.linalg.slogdet(a)
        return LogDetResult(sign, float(value))
    if op == "logm":
        if np.linalg.slogdet(a)[0] == 0:
            raise np.linalg.LinAlgError("a singular matrix has no finite matrix logarithm")
        value = linalg.logm(a)
        residual = linalg.norm(linalg.expm(value)-a)
    else:
        value = linalg.sqrtm(a)
        residual = linalg.norm(value @ value-a)
    return ArrayResult(value, Diagnostics(residual_norm=float(residual)))
