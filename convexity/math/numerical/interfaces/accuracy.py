from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.sparse import csr_array, csr_matrix
from scipy.sparse.linalg import LinearOperator

from ..._typing import (
    Matrix,
    MatrixLike,
    NumericArray,
)
from .. import _validation as v


@dataclass(frozen=True)
class Tolerances:
    """Componentwise |actual-reference| <= atol + rtol*|reference|."""

    atol: float = 1e-8
    rtol: float = 1e-5

    def __post_init__(self) -> None:
        v.positive(self.atol, "atol", zero=True)
        v.positive(self.rtol, "rtol", zero=True)


@dataclass(frozen=True)
class ComparisonProblem:
    """Actual and reference arrays with tolerance and NaN-equality conventions; shapes are validated."""

    actual: NumericArray
    reference: NumericArray
    tolerances: Tolerances = Tolerances()
    equal_nan: bool = False

    def __post_init__(self) -> None:
        v.boolean(self.equal_nan, "equal_nan")
        if not isinstance(self.tolerances, Tolerances):
            raise TypeError("tolerances must be Tolerances")
        actual, reference = (
            v.array(self.actual, "actual", finite=False),
            v.array(self.reference, "reference", finite=False),
        )
        if actual.shape != reference.shape:
            raise ValueError(
                "comparison shapes must match; broadcasting is not implicit"
            )
        object.__setattr__(self, "actual", actual)
        object.__setattr__(self, "reference", reference)

    def validate_result(self, result: object) -> None:
        if (
            not isinstance(result, np.ndarray)
            or result.dtype != bool
            or result.shape != self.actual.shape
        ):
            raise ValueError(
                "comparison tool must return a boolean array with the input shape"
            )

    def __init__(
        self,
        actual: ArrayLike,
        reference: ArrayLike,
        tolerances: Tolerances = Tolerances(atol=1e-08, rtol=1e-05),
        equal_nan: bool = False,
    ) -> None:
        object.__setattr__(self, "actual", actual)
        object.__setattr__(self, "reference", reference)
        object.__setattr__(self, "tolerances", tolerances)
        object.__setattr__(self, "equal_nan", equal_nan)
        self.__post_init__()


@dataclass(frozen=True)
class MachinePrecision:
    """Floating dtype limits: machine epsilon, smallest normal/subnormal and largest finite magnitude."""

    dtype: str
    epsilon: float | np.floating
    smallest_normal: float | np.floating
    smallest_subnormal: float | np.floating
    largest: float | np.floating


@dataclass(frozen=True)
class ResidualProblem:
    """Linear system A x = b whose residual is to be measured."""

    a: Matrix
    x: NumericArray
    b: NumericArray

    def __post_init__(self) -> None:
        a = self.a
        if isinstance(a, LinearOperator):
            if a.dtype is None or np.dtype(str(a.dtype)).kind not in "fc":
                raise ValueError(
                    "operator must declare a floating-point or complex dtype"
                )
        elif isinstance(a, (csr_matrix, csr_array)):
            a = a.tocsr(copy=True)
            if not np.all(np.isfinite(a.data)):
                raise ValueError("a must contain finite values")
        else:
            a = v.array(a, "a", ndim=2)
        x, b = v.array(self.x, "x"), v.array(self.b, "b")
        if (
            a.shape is None
            or len(a.shape) != 2
            or x.ndim not in (1, 2)
            or b.ndim != x.ndim
            or x.shape[0] != a.shape[1]
            or b.shape != (a.shape[0],) + x.shape[1:]
        ):
            raise ValueError("incompatible shapes for A @ x - b")
        for name, value in (("a", a), ("x", x), ("b", b)):
            object.__setattr__(self, name, value)

    def validate_result(self, result: object) -> None:
        if not isinstance(result, ResidualResult) or result.value.shape != self.b.shape:
            raise ValueError("invalid residual result")

    def __init__(
        self,
        a: MatrixLike,
        x: ArrayLike,
        b: ArrayLike,
    ) -> None:
        object.__setattr__(self, "a", a)
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "b", b)
        self.__post_init__()


@dataclass(frozen=True)
class ResidualResult:
    """Residual array with absolute and scaled relative norms."""

    value: NumericArray
    absolute: float
    relative: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", v.array(self.value, "residual"))
        v.positive(self.absolute, "absolute residual", zero=True)
        if v.scalar(self.relative, "relative residual", finite=False) < 0:
            raise ValueError("relative residual must be nonnegative")

    def __init__(
        self,
        value: ArrayLike,
        absolute: float,
        relative: float,
    ) -> None:
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "absolute", absolute)
        object.__setattr__(self, "relative", relative)
        self.__post_init__()
