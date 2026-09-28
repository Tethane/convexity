"""Univariate interpolation with scalar or multichannel sample values."""

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .. import _validation as v


@dataclass(frozen=True)
class InterpolationProblem:
    """Strictly increasing x: (n,), y: (n, ...); n >= 2, sample axis zero."""

    x: ArrayLike
    y: ArrayLike
    extrapolate: bool = False

    def __post_init__(self) -> None:
        v.boolean(self.extrapolate, "extrapolate")
        x, y = v.array(self.x, "x", ndim=1, real=True), v.array(self.y, "y", real=True)
        if x.size < 2 or np.any(np.diff(x) <= 0):
            raise ValueError("x must contain at least two strictly increasing knots")
        if not y.ndim or y.shape[0] != x.size:
            raise ValueError("y must have shape (len(x), ...)")
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)

    def validate_result(self, result: Any) -> None:
        if not isinstance(result, Interpolant):
            raise TypeError("tool must return Interpolant")
        x = np.asarray(self.x)
        if (result.domain != (x[0], x[-1]) or
                result.output_shape != np.shape(self.y)[1:] or
                result.extrapolate != self.extrapolate):
            raise ValueError("interpolant domain, output shape, or extrapolation policy mismatch")


@dataclass(frozen=True, init=False)
class Interpolant:
    """Checked callable. Query shape S returns shape S + output_shape."""

    _evaluate: Callable[[NDArray[Any]], ArrayLike] = field(repr=False)
    domain: tuple[float, float]
    output_shape: tuple[int, ...] = ()
    extrapolate: bool = False

    def __init__(self, evaluate: Callable[[NDArray[Any]], ArrayLike],
                 domain: tuple[float, float], output_shape: tuple[int, ...] = (),
                 extrapolate: bool = False):
        v.boolean(extrapolate, "extrapolate")
        v.function(evaluate, "evaluate")
        endpoints = v.array(domain, "domain", ndim=1, real=True)
        if endpoints.shape != (2,) or endpoints[0] >= endpoints[1]:
            raise ValueError("domain must have two increasing finite endpoints")
        object.__setattr__(self, "domain", (float(endpoints[0]), float(endpoints[1])))
        for size in output_shape:
            v.integer(size, "output_shape dimension")
        object.__setattr__(self, "output_shape", tuple(output_shape))
        object.__setattr__(self, "extrapolate", extrapolate)
        object.__setattr__(self, "_evaluate", evaluate)

    def __call__(self, x: ArrayLike) -> NDArray[Any]:
        points = v.array(x, "query", real=True)
        if not self.extrapolate and np.any((points < self.domain[0]) | (points > self.domain[1])):
            raise ValueError("query lies outside the interpolation domain")
        result = v.array(self._evaluate(points), "interpolated values", real=True)
        if result.shape != points.shape + self.output_shape:
            raise ValueError("interpolator returned an incompatible shape")
        return result
