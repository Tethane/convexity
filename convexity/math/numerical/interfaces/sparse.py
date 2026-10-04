"""Sparse spectral and matrix-free contracts; native SciPy containers interoperate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, DTypeLike
from scipy.sparse import sparray, spmatrix
from scipy.sparse.linalg import LinearOperator

from ..._typing import Matrix, MatrixLike, NumericArray
from .. import _validation as v
from .linalg import EigenResult, SVDResult


@dataclass(frozen=True)
class OperatorProblem:
    """Shape and matrix-vector callbacks defining a linear operator; adjoint action is optional."""

    shape: tuple[int, int]
    matvec: Callable[[NumericArray], ArrayLike]
    rmatvec: Callable[[NumericArray], ArrayLike] | None = None
    dtype: DTypeLike = np.float64

    def __post_init__(self) -> None:
        if len(self.shape) != 2:
            raise ValueError("operator shape must be (rows, columns)")
        for n in self.shape:
            v.integer(n, "operator dimension")
        v.function(self.matvec, "matvec")
        if self.rmatvec is not None:
            v.function(self.rmatvec, "rmatvec")
        dtype = np.dtype(self.dtype)
        if dtype.kind not in "fc":
            raise ValueError("operator dtype must be floating-point or complex")
        object.__setattr__(self, "dtype", dtype)

    def validate_result(self, result: object) -> None:
        if (
            not isinstance(result, LinearOperator)
            or result.shape != self.shape
            or result.dtype != self.dtype
        ):
            raise ValueError(
                "tool must return a LinearOperator of the declared shape/dtype"
            )


@dataclass(frozen=True)
class SparseSpectrumProblem:
    """Selected eigenpairs of a sparse matrix or linear operator, with Hermitian structure declared explicitly."""

    a: Matrix
    k: int
    kind: str = "eigen"
    hermitian: bool = True
    which: str = "LM"

    def __post_init__(self) -> None:
        v.choice(self.kind, ("eigen", "svd"), "kind")
        v.boolean(self.hermitian, "hermitian")
        k = v.integer(self.k, "k")
        a = self.a
        if (
            not isinstance(a, (spmatrix, sparray, LinearOperator))
            or a.shape is None
            or len(a.shape) != 2
        ):
            raise TypeError("a must be a sparse matrix or LinearOperator")
        if (
            not all(a.shape)
            or a.dtype is None
            or np.dtype(str(a.dtype)).kind not in "iufc"
        ):
            raise ValueError("a must be a nonempty numeric matrix/operator")
        if not isinstance(a, LinearOperator):
            a = a.tocsr(copy=True).astype(
                np.complex128 if np.iscomplexobj(a.data) else np.float64
            )
            if not np.all(np.isfinite(a.data)):
                raise ValueError("sparse data must be finite")
        assert a.shape is not None
        if self.kind == "eigen":
            if a.shape[0] != a.shape[1]:
                raise ValueError("eigenproblems require square a")
            # Complex Hermitian eigsh delegates to eigs, also requiring k < n-1.
            limit = (
                a.shape[0]
                if self.hermitian and np.dtype(str(a.dtype)).kind != "c"
                else a.shape[0] - 1
            )
            choices = (
                ("LM", "SM", "LA", "SA")
                if self.hermitian
                else ("LM", "SM", "LR", "SR", "LI", "SI")
            )
            if self.hermitian and not isinstance(a, LinearOperator):
                difference = a - a.conj().T
                if difference.nnz and np.max(np.abs(difference.data)) > 1e-12 * max(
                    1.0, float(abs(a).max())
                ):
                    raise ValueError("a must be Hermitian")
        else:
            limit, choices = min(a.shape), ("LM", "SM")
        if k >= limit:
            raise ValueError(f"ARPACK requires k < {limit} for this problem")
        v.choice(self.which, choices, "which")
        object.__setattr__(self, "a", a)

    def validate_result(self, result: object) -> None:
        assert self.a.shape is not None
        m, n = self.a.shape
        if self.kind == "eigen":
            if not isinstance(result, EigenResult):
                raise TypeError("tool must return EigenResult")
            if result.values.shape != (self.k,) or result.vectors.shape != (n, self.k):
                raise ValueError("invalid sparse eigenpair shapes")
        else:
            if not isinstance(result, SVDResult):
                raise TypeError("tool must return SVDResult")
            if (
                result.u.shape != (m, self.k)
                or result.s.shape != (self.k,)
                or result.vh.shape != (self.k, n)
            ):
                raise ValueError("invalid sparse SVD shapes")

    def __init__(
        self,
        a: MatrixLike,
        k: int,
        kind: str = "eigen",
        hermitian: bool = True,
        which: str = "LM",
    ) -> None:
        object.__setattr__(self, "a", a)
        object.__setattr__(self, "k", k)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "hermitian", hermitian)
        object.__setattr__(self, "which", which)
        self.__post_init__()
