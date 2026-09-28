"""Additional dense linear algebra contracts."""
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
from numpy.typing import ArrayLike

from .. import _validation as v
from .base import ArrayResult, Diagnostics, check_array_result


@dataclass(frozen=True)
class LeastSquaresProblem:
    a: ArrayLike
    b: ArrayLike

    def __post_init__(self):
        a, b = v.array(self.a, "a", ndim=2), v.array(self.b, "b")
        if b.ndim not in (1, 2) or b.shape[0] != a.shape[0]:
            raise ValueError("b must have shape (m,) or (m,k), matching a: (m,n)")
        object.__setattr__(self, "a", a)
        object.__setattr__(self, "b", b)

    def validate_result(self, result):
        if not isinstance(result, LeastSquaresResult):
            raise TypeError("tool must return LeastSquaresResult")
        expected = (np.shape(self.a)[1],) + np.shape(self.b)[1:]
        if result.value.shape != expected or result.singular_values.shape != (min(np.shape(self.a)),) or not 0 <= result.rank <= min(np.shape(self.a)):
            raise ValueError("invalid least-squares result shape or rank")


@dataclass(frozen=True, init=False)
class LeastSquaresResult(ArrayResult):
    rank: int
    singular_values: Any

    def __init__(self, value, rank, singular_values, diagnostics=None):
        super().__init__(value, diagnostics)
        if not isinstance(rank, (int, np.integer)) or rank < 0:
            raise ValueError("rank must be a nonnegative integer")
        s = v.array(singular_values, "singular_values", ndim=1, real=True)
        if np.any(s < 0):
            raise ValueError("singular values must be nonnegative")
        object.__setattr__(self, "rank", int(rank))
        object.__setattr__(self, "singular_values", s)


@dataclass(frozen=True)
class GeneralizedEigenproblem:
    a: ArrayLike
    b: ArrayLike
    hermitian: bool = False

    def __post_init__(self):
        a, b = v.square(self.a, "a"), v.square(self.b, "b")
        v.boolean(self.hermitian, "hermitian")
        if a.shape != b.shape:
            raise ValueError("a and b must have the same shape")
        if self.hermitian:
            v.hermitian(a, "a")
            v.hermitian(b, "b")
        object.__setattr__(self, "a", a)
        object.__setattr__(self, "b", b)

    def validate_result(self, result):
        if not isinstance(result, GeneralizedEigenResult):
            raise TypeError("tool must return GeneralizedEigenResult")
        n = np.shape(self.a)[0]
        if result.alpha.shape != (n,) or result.beta.shape != (n,) or result.vectors.shape != (n, n):
            raise ValueError("invalid generalized eigenpair shapes")


@dataclass(frozen=True)
class GeneralizedEigenResult:
    """beta * (A @ vectors) = alpha * (B @ vectors), including infinite eigenvalues."""
    alpha: Any
    beta: Any
    vectors: Any

    def __post_init__(self):
        for name, ndim in (("alpha", 1), ("beta", 1), ("vectors", 2)):
            object.__setattr__(self, name, v.array(getattr(self, name), name, ndim=ndim))

    @property
    def values(self):
        # alpha=beta=0 is an indeterminate eigenvalue, not a finite eigenvalue.
        result = np.full(self.alpha.shape, np.inf, dtype=np.result_type(self.alpha, self.beta))
        np.divide(self.alpha, self.beta, out=result, where=self.beta != 0)
        result[(self.alpha == 0) & (self.beta == 0)] = np.nan
        return result


@dataclass(frozen=True)
class FactorizationProblem:
    a: ArrayLike
    kind: str = "qr"
    pivoting: bool = False

    def __post_init__(self):
        v.choice(self.kind, ("qr", "lu", "cholesky", "schur"), "kind")
        v.boolean(self.pivoting, "pivoting")
        if self.pivoting and self.kind != "qr":
            raise ValueError("pivoting is a QR option; LU always uses row pivoting")
        a = v.array(self.a, "a", ndim=2)
        if self.kind in ("cholesky", "schur"):
            a = v.square(a, "a")
        if self.kind == "cholesky":
            v.hermitian(a, "a")
        object.__setattr__(self, "a", a)

    def validate_result(self, result):
        if not isinstance(result, FactorizationResult):
            raise TypeError("tool must return FactorizationResult")
        m, n = np.shape(self.a)
        k = min(m, n)
        shapes = {
            "qr": {"q": (m, k), "r": (k, n)},
            "lu": {"p": (m, m), "l": (m, k), "u": (k, n)},
            "cholesky": {"l": (n, n)}, "schur": {"t": (n, n), "z": (n, n)},
        }[self.kind]
        if set(result.factors) != set(shapes) or any(result.factors[key].shape != shape for key, shape in shapes.items()):
            raise ValueError("factor names/shapes do not match the requested decomposition")
        if self.kind == "qr":
            if result.permutation is None or not np.array_equal(np.sort(result.permutation), np.arange(n)):
                raise ValueError("QR must return a column permutation")


@dataclass(frozen=True)
class FactorizationResult:
    factors: Mapping[str, Any]
    permutation: Any = None

    def __post_init__(self):
        object.__setattr__(self, "factors", MappingProxyType({k: v.array(a, k, ndim=2) for k, a in self.factors.items()}))
        if self.permutation is not None:
            p = np.asarray(self.permutation)
            if p.ndim != 1 or p.dtype.kind not in "iu":
                raise ValueError("permutation must be an integer vector")
            p = p.copy()
            p.setflags(write=False)
            object.__setattr__(self, "permutation", p)


@dataclass(frozen=True)
class MatrixPropertyProblem:
    a: ArrayLike
    operation: str
    order: Any = None
    tolerance: float | None = None

    def __post_init__(self):
        v.choice(self.operation, ("rank", "condition", "norm", "pinv", "det", "slogdet", "logm", "sqrtm"), "operation")
        a = v.array(self.a, "a", ndim=None if self.operation == "norm" else 2)
        if a.ndim not in (1, 2):
            raise ValueError("norm accepts vectors or matrices")
        if self.operation in ("det", "slogdet", "logm", "sqrtm"):
            a = v.square(a, "a")
        if self.tolerance is not None:
            v.positive(self.tolerance, "tolerance", zero=True)
        if self.order not in (None, "fro", "nuc", 1, -1, 2, -2, np.inf, -np.inf):
            raise ValueError("unsupported matrix norm order")
        object.__setattr__(self, "a", a)

    def validate_result(self, result):
        if self.operation == "slogdet":
            if not isinstance(result, LogDetResult):
                raise TypeError("tool must return LogDetResult")
        elif self.operation == "condition":
            if not isinstance(result, ConditionResult):
                raise TypeError("tool must return ConditionResult")
        else:
            shape = np.shape(self.a)
            if self.operation == "pinv":
                shape = shape[::-1]
            elif self.operation not in ("logm", "sqrtm"):
                shape = ()
            check_array_result(result, shape)


@dataclass(frozen=True)
class ConditionResult:
    """Infinity is a legitimate condition number for a singular matrix."""
    value: float

    def __post_init__(self):
        value = v.scalar(self.value, "condition", finite=False)
        if value < 0:
            raise ValueError("condition number must be nonnegative")
        object.__setattr__(self, "value", value)


@dataclass(frozen=True)
class LogDetResult:
    sign: Any
    logabsdet: float

    def __post_init__(self):
        sign = v.array(self.sign, "sign", ndim=0).item()
        logabsdet = v.scalar(self.logabsdet, "logabsdet", finite=False)
        if not (sign == 0 and logabsdet == -np.inf or np.isclose(abs(sign), 1) and np.isfinite(logabsdet)):
            raise ValueError("invalid determinant sign/log magnitude")
        object.__setattr__(self, "sign", sign)
        object.__setattr__(self, "logabsdet", logabsdet)
