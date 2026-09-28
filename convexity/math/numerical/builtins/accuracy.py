from dataclasses import dataclass
import numpy as np
from numpy.typing import DTypeLike
from .. import _validation as v
from .._backends import accuracy as backend
from ..interfaces.base import run
from ..interfaces.accuracy import ComparisonProblem, ResidualProblem, Tolerances


@dataclass(frozen=True)
class FloatingComparison:
    def __call__(self, problem: ComparisonProblem):
        return backend.compare(problem)


@dataclass(frozen=True)
class Residual:
    def __call__(self, problem: ResidualProblem):
        return backend.residual(problem)


def isclose(actual, reference, *, tolerances=Tolerances(), equal_nan=False, tool=None):
    return run(ComparisonProblem(actual, reference, tolerances, equal_nan),
               tool if tool is not None else FloatingComparison())


def allclose(actual, reference, *, tolerances=Tolerances(), equal_nan=False, tool=None):
    return bool(np.all(isclose(actual, reference, tolerances=tolerances, equal_nan=equal_nan, tool=tool)))


def machine_precision(dtype: DTypeLike = np.float64):
    dtype = np.dtype(dtype)
    if dtype.kind not in "fc":
        raise ValueError("machine precision requires a floating-point or complex dtype")
    return backend.precision(dtype)


def require_finite(values):
    """Validate and snapshot numeric data, rejecting NaN and Inf rather than replacing them."""
    return v.array(values, "values")


def residual(a, x, b, *, tool=None):
    """Absolute Euclidean/Frobenius residual and residual relative to ||b||."""
    return run(ResidualProblem(a, x, b), tool if tool is not None else Residual())


__all__ = ["FloatingComparison", "Residual", "isclose", "allclose", "machine_precision", "require_finite", "residual"]
