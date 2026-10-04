from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from numpy.typing import ArrayLike

from .. import _validation as v
from .._backends import approximation as backend
from ..interfaces import (
    Interpolant,
    InterpolationProblem,
    MultivariateInterpolant,
    MultivariateInterpolationProblem,
    NumericalTool,
    PolynomialProblem,
    PolynomialResult,
)
from ..interfaces.base import run


@dataclass(frozen=True)
class Spline:
    """Construct a one-dimensional spline with explicit boundary and extrapolation conventions."""

    degree: int = 3
    boundary: str = "not-a-knot"

    def __post_init__(self) -> None:
        v.integer(self.degree, "degree")
        if self.degree not in (1, 3, 5):
            raise ValueError("supported spline degrees are 1, 3, and 5")
        v.choice(self.boundary, ("not-a-knot", "natural", "clamped"), "boundary")
        if self.boundary != "not-a-knot" and self.degree != 3:
            raise ValueError("natural/clamped boundaries require cubic splines")

    def __call__(self, problem: InterpolationProblem):
        return backend.spline(problem, degree=self.degree, boundary=self.boundary)


@dataclass(frozen=True)
class MultivariateInterpolation:
    """Interpolate regular-grid or scattered observations with explicit extrapolation behavior."""

    method: str = "linear"

    def __post_init__(self) -> None:
        v.choice(self.method, ("linear", "nearest", "cubic"), "method")

    def __call__(self, problem: MultivariateInterpolationProblem):
        return backend.multivariate(problem, method=self.method)


@dataclass(frozen=True)
class PolynomialApproximation:
    """Fit a polynomial of the requested degree to supplied observations."""

    basis: str = "chebyshev"

    def __post_init__(self) -> None:
        v.choice(self.basis, ("polynomial", "chebyshev"), "basis")

    def __call__(self, problem: PolynomialProblem):
        return backend.polynomial(problem, basis=self.basis)


def spline(
    x: ArrayLike,
    y: ArrayLike,
    *,
    extrapolate: bool = False,
    tool: NumericalTool[InterpolationProblem, Interpolant] | None = None,
) -> Interpolant:
    """Interpolate values y with leading axis len(x) on a strictly increasing x grid.

    Return a callable interpolant; queries outside the domain require extrapolate=True."""
    return run(
        InterpolationProblem(x, y, extrapolate), tool if tool is not None else Spline()
    )


def regular_grid(
    axes: Sequence[ArrayLike],
    values: ArrayLike,
    *,
    extrapolation: str = "raise",
    tool: NumericalTool[MultivariateInterpolationProblem, MultivariateInterpolant]
    | None = None,
) -> MultivariateInterpolant:
    """Interpolate a tensor grid with one increasing axis per coordinate.

    Leading values dimensions match axis lengths; query shape is (...,dimension).
    Trailing value axes survive. Extrapolation must be selected explicitly."""
    return run(
        MultivariateInterpolationProblem(tuple(axes), values, "regular", extrapolation),
        tool if tool is not None else MultivariateInterpolation(),
    )


def scattered(
    points: ArrayLike,
    values: ArrayLike,
    *,
    extrapolation: str = "raise",
    tool: NumericalTool[MultivariateInterpolationProblem, MultivariateInterpolant]
    | None = None,
) -> MultivariateInterpolant:
    """Interpolate points (n,dimension) and values (n,...), dimension>=2.

    Queries have shape (...,dimension). Outside-hull behavior follows extrapolation."""
    return run(
        MultivariateInterpolationProblem(points, values, "scattered", extrapolation),
        tool if tool is not None else MultivariateInterpolation(),
    )


def polynomial_approximation(
    x: ArrayLike,
    y: ArrayLike,
    degree: int,
    *,
    extrapolate: bool = False,
    tool: NumericalTool[PolynomialProblem, PolynomialResult] | None = None,
    check_convergence: bool = True,
) -> PolynomialResult:
    """Fit degree+1 coefficients to scalar y on increasing sample coordinates x.

    Return a callable approximation and coefficients in increasing ordinary powers
    of x. Rank loss is reflected in convergence diagnostics."""
    return run(
        PolynomialProblem(x, y, extrapolate, degree),
        tool if tool is not None else PolynomialApproximation(),
        check_convergence=check_convergence,
    )


__all__ = [
    "Spline",
    "MultivariateInterpolation",
    "PolynomialApproximation",
    "spline",
    "regular_grid",
    "scattered",
    "polynomial_approximation",
]
