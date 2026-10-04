"""SciPy linear algebra adapters; no Convexity numerical algorithms."""
from __future__ import annotations
from typing import TypeIs

import warnings
from typing import Any

import numpy as np
from scipy import linalg, sparse
from scipy.sparse import linalg as spla

from ..interfaces import (
    ArrayResult, Diagnostics, Eigenproblem, EigenResult, LinearSystem,
    MatrixExponential, SVDProblem, SVDResult,
)


def is_sparse(value: object) -> TypeIs[sparse.spmatrix | sparse.sparray]:
    return sparse.issparse(value)


def is_operator(value: object) -> TypeIs[spla.LinearOperator]:
    return isinstance(value, spla.LinearOperator)


def sparse_matrix(data: Any, *, format: str, shape: tuple[int, int] | None) -> Any:
    constructor = sparse.csr_array if format == "csr" else sparse.csc_array
    result = constructor(data, shape=shape, copy=True)
    if result.shape is None or result.ndim != 2 or not all(result.shape):
        raise ValueError("sparse matrix must be nonempty and two-dimensional")
    if result.dtype.kind not in "iufc" or not np.all(np.isfinite(result.data)):
        raise ValueError("sparse matrix must contain finite numeric values")
    result.sum_duplicates()
    result.sort_indices()
    return result.astype(np.complex128 if np.iscomplexobj(result.data) else np.float64)


def dense_solve(problem: LinearSystem) -> ArrayResult:
    if is_sparse(problem.a) or is_operator(problem.a):
        raise ValueError("DenseSolve requires a dense matrix; use SparseSolve")
    with warnings.catch_warnings():
        warnings.simplefilter("error", linalg.LinAlgWarning)
        x = linalg.solve(problem.a, problem.b, check_finite=False)
    residual = float(np.linalg.norm(problem.a @ x - problem.b))
    return ArrayResult(x, Diagnostics(residual_norm=residual))


def sparse_solve(problem: LinearSystem, *, method: str, atol: float,
                 rtol: float, maxiter: int, preconditioner=None) -> ArrayResult:
    if not (is_sparse(problem.a) or is_operator(problem.a)):
        raise ValueError("SparseSolve requires a sparse matrix or LinearOperator")
    a, b = problem.a, np.asarray(problem.b)
    if preconditioner is not None and preconditioner.shape != a.shape:
        raise ValueError("preconditioner must have the shape of a")
    iterations = 0
    info = 0
    if method == "direct":
        if is_operator(a):
            raise ValueError("direct solve requires an explicit sparse matrix")
        with warnings.catch_warnings():
            warnings.simplefilter("error", spla.MatrixRankWarning)
            try:
                x = np.asarray(spla.spsolve(a, b)).reshape(b.shape)
            except spla.MatrixRankWarning as exc:
                raise np.linalg.LinAlgError(str(exc)) from exc
    else:
        if b.ndim != 1:
            raise ValueError("iterative sparse solvers require b with shape (n,)")
        if method == "cg" and not is_operator(a):
            difference = a - a.conj().T
            if difference.nnz and np.max(np.abs(difference.data)) > 1e-12 * max(1., float(abs(a).max())):
                raise ValueError("conjugate gradient requires a Hermitian positive-definite matrix")
        def count(_):
            nonlocal iterations
            iterations += 1

        if method == "cg":
            x, info = spla.cg(a, b, atol=atol, rtol=rtol, maxiter=maxiter, callback=count, M=preconditioner)
        elif method == "bicgstab":
            x, info = spla.bicgstab(a, b, atol=atol, rtol=rtol, maxiter=maxiter, callback=count, M=preconditioner)
        else:
            x, info = spla.gmres(a, b, atol=atol, rtol=rtol, maxiter=maxiter,
                                 callback=count, callback_type="legacy", M=preconditioner)
    residual = float(np.linalg.norm(np.asarray(a @ x) - b))
    success = bool(info == 0 and np.all(np.isfinite(x)) and np.isfinite(residual))
    return ArrayResult(x, Diagnostics(converged=success,
                                      iterations=None if method == "direct" else iterations,
                                      residual_norm=residual if np.isfinite(residual) else None,
                                      message="" if success else f"{method} failed (info={info}, finite={np.isfinite(residual)})"))


def eigen(problem: Eigenproblem) -> EigenResult:
    values, vectors = (linalg.eigh if problem.hermitian else linalg.eig)(problem.a)
    return EigenResult(values, vectors)


def svd(problem: SVDProblem) -> SVDResult:
    return SVDResult(*linalg.svd(problem.a, full_matrices=problem.full_matrices))


def expm(problem: MatrixExponential) -> ArrayResult:
    return ArrayResult(linalg.expm(problem.a))
