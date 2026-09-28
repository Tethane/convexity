from dataclasses import dataclass
from typing import cast
import numpy as np
from numpy.typing import DTypeLike

from .. import _validation as v
from .._backends import sparse as backend
from ..interfaces.base import run
from ..interfaces.sparse import OperatorProblem, SparseSpectrumProblem
from ..interfaces.linalg import EigenResult, SVDResult


@dataclass(frozen=True)
class MatrixFreeOperator:
    def __call__(self, problem: OperatorProblem):
        return backend.operator(problem)


@dataclass(frozen=True)
class SparseSpectral:
    tolerance: float = 1e-10
    maxiter: int = 1000

    def __post_init__(self):
        v.positive(self.tolerance, "tolerance", zero=True)
        v.integer(self.maxiter, "maxiter")

    def __call__(self, problem: SparseSpectrumProblem):
        return backend.spectrum(problem, tolerance=self.tolerance, maxiter=self.maxiter)


def linear_operator(shape, matvec, *, rmatvec=None, dtype: DTypeLike = np.float64, tool=None):
    """rmatvec applies the conjugate transpose; required by sparse SVD."""
    return run(OperatorProblem(shape, matvec, rmatvec, dtype), tool if tool is not None else MatrixFreeOperator())


def sparse_eig(a, k=1, *, hermitian=True, which="LM", tool=None) -> EigenResult:
    return cast(EigenResult, run(SparseSpectrumProblem(a, k, "eigen", hermitian, which), tool if tool is not None else SparseSpectral()))


def sparse_svd(a, k=1, *, which="LM", tool=None) -> SVDResult:
    return cast(SVDResult, run(SparseSpectrumProblem(a, k, "svd", False, which), tool if tool is not None else SparseSpectral()))


__all__ = ["MatrixFreeOperator", "SparseSpectral", "linear_operator", "sparse_eig", "sparse_svd"]
