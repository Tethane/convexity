from dataclasses import dataclass
from typing import Any, Callable
import numpy as np
from .. import _validation as v
from .base import check_array_result


@dataclass(frozen=True)
class BoxIntegral:
    """Integral of f(x: (d,)) over rectangular, possibly infinite bounds."""
    function: Callable
    bounds: Any

    def __post_init__(self):
        v.function(self.function)
        bounds = v.array(self.bounds, "bounds", ndim=2, real=True, finite=False)
        if bounds.shape[1] != 2 or np.any(np.isnan(bounds)) or np.any(bounds[:, 0] >= bounds[:, 1]):
            raise ValueError("bounds must have shape (d,2) with increasing non-NaN endpoints")
        object.__setattr__(self, "bounds", bounds)

    def validate_result(self, result):
        check_array_result(result, (), real=True)


@dataclass(frozen=True)
class QuadratureRuleProblem:
    n: int
    family: str = "legendre"
    a: float = -1.
    b: float = 1.

    def __post_init__(self):
        v.integer(self.n, "n")
        v.choice(self.family, ("legendre", "hermite", "laguerre"), "family")
        a, b = v.scalar(self.a, "a"), v.scalar(self.b, "b")
        if a >= b:
            raise ValueError("a must be less than b")
        if self.family != "legendre" and (a, b) != (-1., 1.):
            raise ValueError("interval mapping is only supported for Legendre rules")

    def validate_result(self, result):
        if not isinstance(result, QuadratureRule) or result.nodes.shape != (self.n,) or result.weights.shape != (self.n,):
            raise ValueError("tool must return a quadrature rule with n nodes and weights")


@dataclass(frozen=True)
class QuadratureRule:
    """Nodes and positive weights. Weight function is explicit, never a distribution."""
    nodes: Any
    weights: Any
    weight_function: str = "1"

    def __post_init__(self):
        x, w = v.array(self.nodes, "nodes", ndim=1, real=True), v.array(self.weights, "weights", ndim=1, real=True)
        if x.shape != w.shape or np.any(w <= 0) or np.any(np.diff(x) <= 0):
            raise ValueError("quadrature nodes must increase and weights must be positive")
        object.__setattr__(self, "nodes", x)
        object.__setattr__(self, "weights", w)
