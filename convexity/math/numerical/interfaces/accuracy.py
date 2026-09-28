from dataclasses import dataclass
from typing import Any
import numpy as np
from scipy.sparse import issparse
from scipy.sparse.linalg import LinearOperator
from .. import _validation as v


@dataclass(frozen=True)
class Tolerances:
    """Componentwise |actual-reference| <= atol + rtol*|reference|."""
    atol: float = 1e-8
    rtol: float = 1e-5

    def __post_init__(self):
        v.positive(self.atol, "atol", zero=True)
        v.positive(self.rtol, "rtol", zero=True)


@dataclass(frozen=True)
class ComparisonProblem:
    actual: Any
    reference: Any
    tolerances: Tolerances = Tolerances()
    equal_nan: bool = False

    def __post_init__(self):
        v.boolean(self.equal_nan, "equal_nan")
        if not isinstance(self.tolerances, Tolerances):
            raise TypeError("tolerances must be Tolerances")
        actual, reference = v.array(self.actual, "actual", finite=False), v.array(self.reference, "reference", finite=False)
        if actual.shape != reference.shape:
            raise ValueError("comparison shapes must match; broadcasting is not implicit")
        object.__setattr__(self, "actual", actual)
        object.__setattr__(self, "reference", reference)

    def validate_result(self, result):
        if not isinstance(result, np.ndarray) or result.dtype != bool or result.shape != self.actual.shape:
            raise ValueError("comparison tool must return a boolean array with the input shape")


@dataclass(frozen=True)
class MachinePrecision:
    dtype: str
    epsilon: Any
    smallest_normal: Any
    smallest_subnormal: Any
    largest: Any


@dataclass(frozen=True)
class ResidualProblem:
    a: Any
    x: Any
    b: Any

    def __post_init__(self):
        a = self.a
        if isinstance(a, LinearOperator):
            if a.dtype is None or np.dtype(str(a.dtype)).kind not in "fc":
                raise ValueError("operator must declare a floating-point or complex dtype")
        elif issparse(a):
            a = a.tocsr(copy=True)
            if not np.all(np.isfinite(a.data)):
                raise ValueError("a must contain finite values")
        else:
            a = v.array(a, "a", ndim=2)
        x, b = v.array(self.x, "x"), v.array(self.b, "b")
        if len(a.shape) != 2 or x.ndim not in (1, 2) or b.ndim != x.ndim or x.shape[0] != a.shape[1] or b.shape != (a.shape[0],)+x.shape[1:]:
            raise ValueError("incompatible shapes for A @ x - b")
        for name, value in (("a", a), ("x", x), ("b", b)):
            object.__setattr__(self, name, value)

    def validate_result(self, result):
        if not isinstance(result, ResidualResult) or result.value.shape != self.b.shape:
            raise ValueError("invalid residual result")


@dataclass(frozen=True)
class ResidualResult:
    value: Any
    absolute: float
    relative: float

    def __post_init__(self):
        object.__setattr__(self, "value", v.array(self.value, "residual"))
        v.positive(self.absolute, "absolute residual", zero=True)
        if v.scalar(self.relative, "relative residual", finite=False) < 0:
            raise ValueError("relative residual must be nonnegative")
