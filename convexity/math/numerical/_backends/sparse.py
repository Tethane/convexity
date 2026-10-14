import numpy as np
from typing import Any, cast
from scipy.sparse.linalg import LinearOperator, eigs, eigsh, svds

from .. import _validation as v
from ..interfaces.linalg import EigenResult, SVDResult


def operator(problem):
    def checked(function, size, x):
        value = v.array(function(np.asarray(x).reshape(-1)), "operator output", ndim=1)
        if value.shape != (size,):
            raise ValueError("operator callback returned the wrong shape")
        if (
            np.iscomplexobj(value)
            and problem.dtype.kind != "c"
            and not np.iscomplexobj(x)
        ):
            raise ValueError("real operator callback returned complex values")
        return value.astype(np.result_type(problem.dtype, np.asarray(x).dtype))

    # LinearOperator is a public factory; its __new__ signature is wider than
    # its base __init__ annotation in some SciPy versions.
    return cast(Any, LinearOperator)(
        problem.shape,
        matvec=lambda x: checked(problem.matvec, problem.shape[0], x),
        rmatvec=None
        if problem.rmatvec is None
        else lambda x: checked(problem.rmatvec, problem.shape[1], x),
        dtype=problem.dtype,
    )


def spectrum(problem, *, tolerance, maxiter):
    if problem.kind == "svd":
        u, s, vh = svds(
            problem.a,
            k=problem.k,
            which=problem.which,
            tol=tolerance,
            maxiter=maxiter,
            solver="arpack",
            rng=np.random.default_rng(0),
        )
        assert u is not None and vh is not None
        order = np.argsort(s)[::-1]
        return SVDResult(u[:, order], s[order], vh[order])
    operation = eigsh if problem.hermitian else eigs
    values, vectors = operation(
        problem.a,
        k=problem.k,
        which=problem.which,
        tol=tolerance,
        maxiter=maxiter,
        v0=np.random.default_rng(0).standard_normal(problem.a.shape[0]),
    )
    return EigenResult(values, vectors)
