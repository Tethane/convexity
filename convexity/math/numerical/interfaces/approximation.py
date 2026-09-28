from dataclasses import dataclass, field
from typing import Any, Callable
import numpy as np
from .. import _validation as v
from .base import Diagnostics
from .interpolation import InterpolationProblem, Interpolant


@dataclass(frozen=True)
class MultivariateInterpolationProblem:
    points: Any
    values: Any
    layout: str = "regular"
    extrapolation: str = "raise"

    def __post_init__(self):
        v.choice(self.layout, ("regular", "scattered"), "layout")
        v.choice(self.extrapolation, ("raise", "nan", "extrapolate"), "extrapolation")
        values = v.array(self.values, "values", real=True)
        if self.layout == "regular":
            points = tuple(v.array(x, "axis", ndim=1, real=True) for x in self.points)
            if not points or any(x.size < 2 or np.any(np.diff(x) <= 0) for x in points):
                raise ValueError("regular axes must each contain at least two increasing points")
            shape = tuple(x.size for x in points)
            if values.shape[:len(shape)] != shape:
                raise ValueError("leading value dimensions must match the grid axes")
        else:
            points = v.array(self.points, "points", ndim=2, real=True)
            if points.shape[1] < 2 or points.shape[0] < points.shape[1]+1 or values.ndim < 1 or values.shape[0] != points.shape[0]:
                raise ValueError("scattered points need (N,d), d>=2, N>=d+1; values must have N samples")
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "values", values)

    @property
    def dimension(self):
        return len(self.points) if self.layout == "regular" else self.points.shape[1]

    @property
    def output_shape(self):
        return self.values.shape[self.dimension:] if self.layout == "regular" else self.values.shape[1:]

    def validate_result(self, result):
        if not isinstance(result, MultivariateInterpolant):
            raise TypeError("tool must return MultivariateInterpolant")
        if result.dimension != self.dimension or result.output_shape != self.output_shape or result.extrapolation != self.extrapolation:
            raise ValueError("multivariate interpolant contract mismatch")


@dataclass(frozen=True)
class MultivariateInterpolant:
    _evaluate: Callable = field(repr=False)
    dimension: int
    output_shape: tuple[int, ...] = ()
    extrapolation: str = "raise"

    def __post_init__(self):
        v.function(self._evaluate)
        v.integer(self.dimension, "dimension")
        v.choice(self.extrapolation, ("raise", "nan", "extrapolate"), "extrapolation")
        for size in self.output_shape:
            v.integer(size, "output dimension")

    def __call__(self, points):
        points = v.array(points, "query points", real=True)
        if points.ndim < 1 or points.shape[-1] != self.dimension:
            raise ValueError("queries must have shape (..., dimension)")
        shape = points.shape[:-1] + self.output_shape
        output = v.array(self._evaluate(points.reshape(-1, self.dimension)), "interpolated values",
                         real=True, finite=self.extrapolation != "nan")
        if output.shape != (int(np.prod(points.shape[:-1])),) + self.output_shape or np.any(np.isinf(output)):
            raise ValueError("interpolator returned invalid values or shape")
        return output.reshape(shape)


@dataclass(frozen=True)
class PolynomialProblem(InterpolationProblem):
    degree: int = 3

    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.degree, bool) or not isinstance(self.degree, (int, np.integer)) or self.degree < 0 or self.degree >= np.size(self.x):
            raise ValueError("degree must be an integer in [0, number of samples)")
        if np.ndim(self.y) != 1:
            raise ValueError("polynomial approximation accepts scalar sample values")

    def validate_result(self, result):
        if not isinstance(result, PolynomialResult):
            raise TypeError("tool must return PolynomialResult")
        super().validate_result(result.interpolant)
        if result.coefficients.shape != (self.degree+1,) or not 0 <= result.rank <= self.degree+1:
            raise ValueError("invalid polynomial coefficients or rank")


@dataclass(frozen=True)
class PolynomialResult:
    """Coefficients are in increasing ordinary powers of x, regardless of fit basis."""
    interpolant: Interpolant
    coefficients: Any
    rank: int
    diagnostics: Diagnostics

    def __post_init__(self):
        if not isinstance(self.interpolant, Interpolant) or not isinstance(self.diagnostics, Diagnostics):
            raise TypeError("expected Interpolant and Diagnostics")
        object.__setattr__(self, "coefficients", v.array(self.coefficients, "coefficients", ndim=1, real=True))
        v.integer(self.rank, "rank")

    def __call__(self, x):
        return self.interpolant(x)
