from dataclasses import dataclass
from .. import _validation as v
from .._backends import approximation as backend
from ..interfaces.base import run
from ..interfaces.interpolation import InterpolationProblem
from ..interfaces.approximation import MultivariateInterpolationProblem, PolynomialProblem


@dataclass(frozen=True)
class Spline:
    degree: int = 3
    boundary: str = "not-a-knot"

    def __post_init__(self):
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
    method: str = "linear"

    def __post_init__(self):
        v.choice(self.method, ("linear", "nearest", "cubic"), "method")

    def __call__(self, problem: MultivariateInterpolationProblem):
        return backend.multivariate(problem, method=self.method)


@dataclass(frozen=True)
class PolynomialApproximation:
    basis: str = "chebyshev"

    def __post_init__(self):
        v.choice(self.basis, ("polynomial", "chebyshev"), "basis")

    def __call__(self, problem: PolynomialProblem):
        return backend.polynomial(problem, basis=self.basis)


def spline(x, y, *, extrapolate=False, tool=None):
    return run(InterpolationProblem(x, y, extrapolate), tool if tool is not None else Spline())


def regular_grid(axes, values, *, extrapolation="raise", tool=None):
    return run(MultivariateInterpolationProblem(tuple(axes), values, "regular", extrapolation),
               tool if tool is not None else MultivariateInterpolation())


def scattered(points, values, *, extrapolation="raise", tool=None):
    return run(MultivariateInterpolationProblem(points, values, "scattered", extrapolation),
               tool if tool is not None else MultivariateInterpolation())


def polynomial_approximation(x, y, degree, *, extrapolate=False, tool=None, check_convergence=True):
    return run(PolynomialProblem(x, y, extrapolate, degree),
               tool if tool is not None else PolynomialApproximation(), check_convergence=check_convergence)


__all__ = ["Spline", "MultivariateInterpolation", "PolynomialApproximation", "spline", "regular_grid",
           "scattered", "polynomial_approximation"]
