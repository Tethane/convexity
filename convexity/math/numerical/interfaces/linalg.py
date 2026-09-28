"""Dense and sparse linear algebra problem and result schemas."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .. import _validation as v
from .base import check_array_result


@dataclass(frozen=True)
class LinearSystem:
    """A x = b: square dense/sparse A; b has shape (n,) or (n, k)."""

    a: Any
    b: ArrayLike

    def __post_init__(self) -> None:
        a, b = v.matrix(self.a, "a"), v.array(self.b, "b")
        if b.ndim not in (1, 2) or b.shape[0] != a.shape[0]:
            raise ValueError("b must have shape (n,) or (n, k), matching a")
        object.__setattr__(self, "a", a)
        object.__setattr__(self, "b", b)

    def validate_result(self, result: Any) -> None:
        check_array_result(result, np.shape(self.b))


@dataclass(frozen=True, init=False)
class EigenResult:
    """Right eigenvectors are columns: A @ vectors = vectors * values."""

    values: NDArray[Any]
    vectors: NDArray[Any]

    def __init__(self, values: ArrayLike, vectors: ArrayLike):
        object.__setattr__(self, "values", v.array(values, "values", ndim=1))
        object.__setattr__(self, "vectors", v.array(vectors, "vectors", ndim=2))


@dataclass(frozen=True)
class Eigenproblem:
    a: ArrayLike
    hermitian: bool = False

    def __post_init__(self) -> None:
        v.boolean(self.hermitian, "hermitian")
        a = v.square(self.a, "a")
        if self.hermitian:
            v.hermitian(a, "a")
        object.__setattr__(self, "a", a)

    def validate_result(self, result: Any) -> None:
        n = np.shape(self.a)[0]
        if not isinstance(result, EigenResult):
            raise TypeError("tool must return EigenResult")
        if result.values.shape != (n,) or result.vectors.shape != (n, n):
            raise ValueError("eigenvalues/eigenvectors have incompatible shapes")
        if self.hermitian and np.iscomplexobj(result.values):
            raise ValueError("Hermitian eigenvalues must be real")


@dataclass(frozen=True, init=False)
class SVDResult:
    """A = U[:, :k] @ diag(s) @ Vh[:k, :], k = min(m, n)."""

    u: NDArray[Any]
    s: NDArray[Any]
    vh: NDArray[Any]

    def __init__(self, u: ArrayLike, s: ArrayLike, vh: ArrayLike):
        object.__setattr__(self, "u", v.array(u, "u", ndim=2))
        object.__setattr__(self, "s", v.array(s, "s", ndim=1, real=True))
        object.__setattr__(self, "vh", v.array(vh, "vh", ndim=2))
        if np.any(self.s < 0):
            raise ValueError("singular values must be nonnegative")


@dataclass(frozen=True)
class SVDProblem:
    a: ArrayLike
    full_matrices: bool = False

    def __post_init__(self) -> None:
        v.boolean(self.full_matrices, "full_matrices")
        object.__setattr__(self, "a", v.array(self.a, "a", ndim=2))

    def validate_result(self, result: Any) -> None:
        m, n = np.shape(self.a)
        k = min(m, n)
        if not isinstance(result, SVDResult):
            raise TypeError("tool must return SVDResult")
        shapes = ((m, m), (n, n)) if self.full_matrices else ((m, k), (k, n))
        if (result.u.shape, result.vh.shape) != shapes or result.s.shape != (k,):
            raise ValueError("SVD factors have incompatible shapes")


@dataclass(frozen=True)
class MatrixExponential:
    a: ArrayLike

    def __post_init__(self) -> None:
        object.__setattr__(self, "a", v.square(self.a, "a"))

    def validate_result(self, result: Any) -> None:
        check_array_result(result, np.shape(self.a))
