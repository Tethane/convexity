from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from numpy.typing import ArrayLike

from ..._typing import FloatArray

if TYPE_CHECKING:
    from .. import ArrayResult, NumericalTool, QuadratureProblem


from dataclasses import dataclass

from .. import _validation as v
from .._backends import integration as backend
from ..interfaces import (
    ArrayResult,
    BoxIntegral,
    NumericalTool,
    QuadratureProblem,
    QuadratureRule,
    QuadratureRuleProblem,
)
from ..interfaces.base import run
from .calculus import AdaptiveQuadrature, quad


@dataclass(frozen=True)
class MultidimensionalQuadrature:
    """Integrate a scalar callback over a rectangular domain with numerical error controls."""

    atol: float = 1e-8
    rtol: float = 1e-8
    limit: int = 100

    def __post_init__(self) -> None:
        AdaptiveQuadrature(self.atol, self.rtol, self.limit)

    def __call__(self, problem: BoxIntegral):
        return backend.box_integral(
            problem, atol=self.atol, rtol=self.rtol, limit=self.limit
        )


@dataclass(frozen=True)
class GaussianQuadrature:
    """Integrate by a finite Gaussian quadrature rule; polynomial exactness depends on the chosen rule."""

    n: int = 16

    def __post_init__(self) -> None:
        v.integer(self.n, "n")

    def __call__(self, problem: QuadratureProblem):
        return backend.gaussian(problem, n=self.n)


@dataclass(frozen=True)
class GaussianRule:
    """Construct weighted Gaussian quadrature nodes, with the declared weight function and domain."""

    def __call__(self, problem: QuadratureRuleProblem):
        return backend.rule(problem)


def nquad(
    function: Callable[[FloatArray], float],
    bounds: ArrayLike | str,
    *,
    tool: NumericalTool[BoxIntegral, ArrayResult] | None = None,
    check_convergence: bool = True,
) -> ArrayResult:
    """Integrate a scalar function(x) over bounds (dimension,2).

    The callback receives one coordinate vector; bounds may be infinite.
    Return the integral with a numerical error estimate, not a probabilistic interval."""
    return run(
        BoxIntegral(function, bounds),
        tool if tool is not None else MultidimensionalQuadrature(),
        check_convergence=check_convergence,
    )


def gauss_quad(
    function: Callable[[float], float],
    a: float,
    b: float,
    *,
    n: int = 16,
    tool: NumericalTool[QuadratureProblem, ArrayResult] | None = None,
) -> ArrayResult:
    """Approximate the scalar integral from a to b using n Gaussian nodes.

    The fixed-order rule does not supply an adaptive error guarantee."""
    return run(
        QuadratureProblem(function, a, b),
        tool if tool is not None else GaussianQuadrature(n),
    )


def quadrature_rule(
    n: int,
    *,
    family: str = "legendre",
    a: float = -1.0,
    b: float = 1.0,
    tool: NumericalTool[QuadratureRuleProblem, QuadratureRule] | None = None,
) -> QuadratureRule:
    """Return n increasing nodes and positive weights.

    Legendre integrates unit weight on [a,b]; Hermite uses exp(-x**2) on the real
    line and Laguerre uses exp(-x) on [0,infinity). Weights are not probabilities."""
    return run(
        QuadratureRuleProblem(n, family, a, b),
        tool if tool is not None else GaussianRule(),
    )


def integrate_density(
    function: Callable[[float], float],
    density: Callable[[float], float],
    a: float,
    b: float,
    *,
    tool: NumericalTool[QuadratureProblem, ArrayResult] | None = None,
    check_convergence: bool = True,
) -> ArrayResult:
    """Integral f(x)*density(x) dx. Density is supplied; no normalization is assumed or fitted."""
    v.function(function)
    v.function(density, "density")

    def weighted(x):
        weight = v.scalar(density(x), "density output")
        if weight < 0:
            raise ValueError("density must be nonnegative at evaluated points")
        return v.scalar(function(x), "function output") * weight

    return quad(weighted, a, b, tool=tool, check_convergence=check_convergence)


__all__ = [
    "MultidimensionalQuadrature",
    "GaussianQuadrature",
    "GaussianRule",
    "nquad",
    "gauss_quad",
    "quadrature_rule",
    "integrate_density",
]
