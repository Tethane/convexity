from __future__ import annotations

from typing import TYPE_CHECKING

from numpy.typing import ArrayLike, DTypeLike

from ..._typing import BoolArray, NumericArray

if TYPE_CHECKING:
    from .. import MachinePrecision

if TYPE_CHECKING:
    from .. import ComparisonProblem, NumericalTool


from dataclasses import dataclass

import numpy as np

from ..._typing import MatrixLike
from .. import _validation as v
from .._backends import accuracy as backend
from ..interfaces import (
    ComparisonProblem,
    NumericalTool,
    ResidualProblem,
    ResidualResult,
)
from ..interfaces.accuracy import Tolerances
from ..interfaces.base import run


@dataclass(frozen=True)
class FloatingComparison:
    """Compare arrays using explicit absolute/relative tolerances and optional equal-NaN semantics."""

    def __call__(self, problem: ComparisonProblem):
        return backend.compare(problem)


@dataclass(frozen=True)
class Residual:
    """Evaluate a linear-system residual and its absolute and relative norms."""

    def __call__(self, problem: ResidualProblem):
        return backend.residual(problem)


def isclose(
    actual: ArrayLike,
    reference: ArrayLike,
    *,
    tolerances: Tolerances = Tolerances(),
    equal_nan: bool = False,
    tool: NumericalTool[ComparisonProblem, BoolArray] | None = None,
) -> BoolArray:
    """Compare equal-shaped arrays using |actual-reference| <= atol+rtol*|reference|.

    Return a Boolean array; broadcasting is deliberately disallowed. NaNs match
    only when equal_nan=True."""
    return run(
        ComparisonProblem(actual, reference, tolerances, equal_nan),
        tool if tool is not None else FloatingComparison(),
    )


def allclose(
    actual: ArrayLike,
    reference: ArrayLike,
    *,
    tolerances: Tolerances = Tolerances(),
    equal_nan: bool = False,
    tool: NumericalTool[ComparisonProblem, BoolArray] | None = None,
) -> bool:
    """Return whether every coordinate passes isclose with the supplied Tolerances."""
    return bool(
        np.all(
            isclose(
                actual, reference, tolerances=tolerances, equal_nan=equal_nan, tool=tool
            )
        )
    )


def machine_precision(dtype: DTypeLike = np.float64) -> MachinePrecision:
    """Return floating-point limits for dtype, including epsilon and normal/subnormal minima."""
    dtype = np.dtype(dtype)
    if dtype.kind not in "fc":
        raise ValueError("machine precision requires a floating-point or complex dtype")
    return backend.precision(dtype)


def require_finite(values: ArrayLike) -> NumericArray:
    """Validate and snapshot numeric data, rejecting NaN and Inf rather than replacing them."""
    return v.array(values, "values")


def residual(
    a: MatrixLike,
    x: ArrayLike,
    b: ArrayLike,
    *,
    tool: NumericalTool[ResidualProblem, ResidualResult] | None = None,
) -> ResidualResult:
    """Absolute Euclidean/Frobenius residual and residual relative to ||b||."""
    return run(ResidualProblem(a, x, b), tool if tool is not None else Residual())


__all__ = [
    "FloatingComparison",
    "Residual",
    "isclose",
    "allclose",
    "machine_precision",
    "require_finite",
    "residual",
]
