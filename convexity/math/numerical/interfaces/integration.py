from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike

from ..._typing import (
    FloatArray,
)
from .. import _validation as v
from .base import check_array_result


@dataclass(frozen=True)
class BoxIntegral:
    """Integral of f(x: (d,)) over rectangular, possibly infinite bounds."""

    function: Callable[[FloatArray], float]
    bounds: ArrayLike

    def __post_init__(self) -> None:
        v.function(self.function)
        bounds = v.array(self.bounds, "bounds", ndim=2, real=True, finite=False)
        if (
            bounds.shape[1] != 2
            or np.any(np.isnan(bounds))
            or np.any(bounds[:, 0] >= bounds[:, 1])
        ):
            raise ValueError(
                "bounds must have shape (d,2) with increasing non-NaN endpoints"
            )
        object.__setattr__(self, "bounds", bounds)

    def validate_result(self, result: object) -> None:
        check_array_result(result, (), real=True)


@dataclass(frozen=True)
class QuadratureRuleProblem:
    """Number of nodes, polynomial family and integration endpoints for a weighted Gaussian rule."""

    n: int
    family: str = "legendre"
    a: float = -1.0
    b: float = 1.0

    def __post_init__(self) -> None:
        v.integer(self.n, "n")
        v.choice(self.family, ("legendre", "hermite", "laguerre"), "family")
        a, b = v.scalar(self.a, "a"), v.scalar(self.b, "b")
        if a >= b:
            raise ValueError("a must be less than b")
        if self.family != "legendre" and (a, b) != (-1.0, 1.0):
            raise ValueError("interval mapping is only supported for Legendre rules")

    def validate_result(self, result: object) -> None:
        if (
            not isinstance(result, QuadratureRule)
            or result.nodes.shape != (self.n,)
            or result.weights.shape != (self.n,)
        ):
            raise ValueError(
                "tool must return a quadrature rule with n nodes and weights"
            )


@dataclass(frozen=True)
class QuadratureRule:
    """Nodes and positive weights. Weight function is explicit, never a distribution."""

    nodes: FloatArray
    weights: FloatArray
    weight_function: str = "1"

    def __post_init__(self) -> None:
        x, w = (
            v.array(self.nodes, "nodes", ndim=1, real=True),
            v.array(self.weights, "weights", ndim=1, real=True),
        )
        if x.shape != w.shape or np.any(w <= 0) or np.any(np.diff(x) <= 0):
            raise ValueError(
                "quadrature nodes must increase and weights must be positive"
            )
        object.__setattr__(self, "nodes", x)
        object.__setattr__(self, "weights", w)

    def __init__(
        self,
        nodes: ArrayLike,
        weights: ArrayLike,
        weight_function: str = "1",
    ) -> None:
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "weight_function", weight_function)
        self.__post_init__()
