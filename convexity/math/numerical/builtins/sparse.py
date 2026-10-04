from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import ArrayLike, DTypeLike
from scipy.sparse.linalg import LinearOperator

from ..._typing import MatrixLike, NumericArray
from .. import _validation as v
from .._backends import sparse as backend
from ..interfaces import (
    EigenResult,
    NumericalTool,
    OperatorProblem,
    SparseSpectrumProblem,
    SVDResult,
)
from ..interfaces.base import run


@dataclass(frozen=True)
class MatrixFreeOperator:
    """Wrap a shape and matrix-vector action without materializing a dense matrix."""

    def __call__(self, problem: OperatorProblem):
        return backend.operator(problem)


@dataclass(frozen=True)
class SparseSpectral:
    """Compute selected sparse eigenpairs with explicit selection and convergence settings."""

    tolerance: float = 1e-10
    maxiter: int = 1000

    def __post_init__(self) -> None:
        v.positive(self.tolerance, "tolerance", zero=True)
        v.integer(self.maxiter, "maxiter")

    def __call__(self, problem: SparseSpectrumProblem):
        return backend.spectrum(problem, tolerance=self.tolerance, maxiter=self.maxiter)


def linear_operator(
    shape: tuple[int, int],
    matvec: Callable[[NumericArray], ArrayLike],
    *,
    rmatvec: Callable[[NumericArray], ArrayLike] | None = None,
    dtype: DTypeLike = np.float64,
    tool: NumericalTool[OperatorProblem, LinearOperator] | None = None,
) -> LinearOperator:
    """rmatvec applies the conjugate transpose; required by sparse SVD."""
    return run(
        OperatorProblem(shape, matvec, rmatvec, dtype),
        tool if tool is not None else MatrixFreeOperator(),
    )


def sparse_eig(
    a: MatrixLike,
    k: int = 1,
    *,
    hermitian: bool = True,
    which: str = "LM",
    tool: NumericalTool[SparseSpectrumProblem, EigenResult] | None = None,
) -> EigenResult:
    """Compute k eigenpairs of a sparse matrix or linear operator.

    Eigenvectors are columns. which selects spectral location; hermitian=True
    requires a self-adjoint operator. Convergence failures are reported explicitly."""
    return cast(
        EigenResult,
        run(
            SparseSpectrumProblem(a, k, "eigen", hermitian, which),
            tool if tool is not None else SparseSpectral(),
        ),
    )


def sparse_svd(
    a: MatrixLike,
    k: int = 1,
    *,
    which: str = "LM",
    tool: NumericalTool[SparseSpectrumProblem, SVDResult] | None = None,
) -> SVDResult:
    """Compute k singular triplets of a sparse matrix or linear operator.

    Return u, nonnegative s, vh; truncation is an approximation, not full factorization."""
    return cast(
        SVDResult,
        run(
            SparseSpectrumProblem(a, k, "svd", False, which),
            tool if tool is not None else SparseSpectral(),
        ),
    )


__all__ = [
    "MatrixFreeOperator",
    "SparseSpectral",
    "linear_operator",
    "sparse_eig",
    "sparse_svd",
]
