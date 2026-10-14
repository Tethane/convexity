"""Explicit finite robust and stochastic convex formulations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ... import _decision as v
from ..._typing import FloatArray
from ...optimization import (
    OptimizationError,
    QuadraticProgram,
    SecondOrderCone,
    SolveResult,
)


@dataclass(frozen=True)
class Box:
    """Coordinatewise uncertainty set defined by a center and nonnegative coordinate radii in the same units as the uncertain vector."""

    center: FloatArray
    radius: FloatArray

    def __post_init__(self) -> None:
        c = v.array(self.center, "center", ndim=1)
        r = v.vector(self.radius, len(c), "radius")
        if np.any(r < 0):
            raise ValueError("radii must be nonnegative")
        object.__setattr__(self, "center", c)
        object.__setattr__(self, "radius", r)

    def support(self, direction: ArrayLike) -> float:
        d = v.array(direction, "direction", ndim=1)
        if d.shape != self.center.shape:
            raise ValueError("direction must align with center")
        return float(self.center @ d + self.radius @ abs(d))

    def __init__(
        self,
        center: ArrayLike,
        radius: ArrayLike,
    ) -> None:
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "radius", radius)
        self.__post_init__()


@dataclass(frozen=True)
class Ellipsoid:
    """center + shape @ u, ||u||_2 <= radius; shape need not be square."""

    center: FloatArray
    shape: FloatArray
    radius: float = 1.0

    def __post_init__(self) -> None:
        c, s = (
            v.array(self.center, "center", ndim=1),
            v.array(self.shape, "shape", ndim=2),
        )
        if s.shape[0] != len(c):
            raise ValueError("shape rows must align with center")
        v.scalar(self.radius, "radius", minimum=0)
        object.__setattr__(self, "center", c)
        object.__setattr__(self, "shape", s)

    def support(self, direction: ArrayLike) -> float:
        d = v.array(direction, "direction", ndim=1)
        if d.shape != self.center.shape:
            raise ValueError("direction must align with center")
        return float(self.center @ d + self.radius * np.linalg.norm(self.shape.T @ d))

    def __init__(
        self,
        center: ArrayLike,
        shape: ArrayLike,
        radius: float = 1.0,
    ) -> None:
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "radius", radius)
        self.__post_init__()


def _checked(result):
    if not result.success:
        raise OptimizationError(result)
    return result


def minimax(
    loss_matrix: ArrayLike,
    offsets: ArrayLike | None = None,
    *,
    lower_bounds: float = 0.0,
    upper_bounds: float = 1.0,
    budget: float = 1.0,
    solver: str | None = None,
) -> SolveResult:
    """Minimize max_s(A_s @ x + b_s), with sum(x)=budget."""
    from .._backends.cvxpy import minimax as run

    a = v.array(loss_matrix, "loss matrix", ndim=2)
    b = v.vector(0.0 if offsets is None else offsets, a.shape[0], "offsets")
    bounds = QuadraticProgram(
        np.zeros(a.shape[1]), lower_bounds=lower_bounds, upper_bounds=upper_bounds
    )
    return _checked(run(a, b, bounds, v.scalar(budget, "budget"), solver))


def mean_cvar(
    returns: ArrayLike,
    *,
    weights: ArrayLike | None = None,
    ambiguity: Ellipsoid | None = None,
    aversion: float = 1.0,
    level: float = 0.95,
    lower_bounds: float = 0.0,
    upper_bounds: float = 1.0,
    budget: float = 1.0,
    solver: str | None = None,
) -> SolveResult:
    """Minimize negative expected return plus aversion times empirical CVaR.

    An optional ellipsoid makes the mean term robust: its center must equal
    the scenario-weighted mean, and its support penalty is added to the loss.
    CVaR still uses the supplied scenarios; this does not make their tail
    distribution robust or account for losses absent from the sample.
    """
    from .._backends.cvxpy import mean_cvar as run

    r = v.array(returns, "returns", ndim=2)
    p = v.weights(weights, len(r))
    if ambiguity is not None:
        mean = p @ r
        if (
            not isinstance(ambiguity, Ellipsoid)
            or ambiguity.center.shape != mean.shape
            or not np.allclose(ambiguity.center, mean, rtol=1e-10, atol=1e-12)
        ):
            raise ValueError("ambiguity center must equal the scenario-weighted mean")

    a = v.scalar(aversion, "aversion", minimum=0)
    level = v.probability(level, "level", open_interval=True)
    bounds = QuadraticProgram(
        np.zeros(r.shape[1]), lower_bounds=lower_bounds, upper_bounds=upper_bounds
    )
    return _checked(
        run(r, p, a, level, bounds, v.scalar(budget, "budget"), solver, ambiguity)
    )


def robust_allocation(
    mean: ArrayLike,
    covariance: ArrayLike,
    ambiguity: Ellipsoid,
    *,
    aversion: float = 1.0,
    lower_bounds: float = 0.0,
    upper_bounds: float = 1.0,
    budget: float = 1.0,
    solver: str | None = None,
) -> SolveResult:
    """Minimize -mean@x + rho||shape.T@x|| + aversion/2*x.T@cov@x.

    Ambiguity center must equal the supplied mean (an explicit consistency check).
    """
    from .._backends.cvxpy import allocation

    m = v.array(mean, "mean", ndim=1)
    c = v.psd(covariance, "covariance")
    if (
        c.shape != (len(m), len(m))
        or not isinstance(ambiguity, Ellipsoid)
        or not np.array_equal(m, ambiguity.center)
    ):
        raise ValueError("covariance and ambiguity center must align with mean")
    a = v.scalar(aversion, "aversion", minimum=0)
    bounds = QuadraticProgram(m, lower_bounds=lower_bounds, upper_bounds=upper_bounds)
    return _checked(
        allocation(m, c, ambiguity, a, bounds, v.scalar(budget, "budget"), solver)
    )


def gaussian_chance_constraint(
    mean: ArrayLike, covariance: ArrayLike, bound: float, *, probability: float = 0.95
) -> SecondOrderCone:
    """Return SOC for P[a.T@x <= bound]>=probability, Gaussian a; p>=.5."""
    from .._backends.scipy import normal_quantile

    m, c = v.array(mean, "mean", ndim=1), v.psd(covariance, "covariance")
    p = v.probability(probability, open_interval=True)
    if p < 0.5 or c.shape != (len(m), len(m)):
        raise ValueError(
            "convex Gaussian chance constraint requires p>=.5 and aligned covariance"
        )
    from ..._backends.gaussian import square_root

    root = square_root(c).T
    return SecondOrderCone(
        normal_quantile(p) * root, np.zeros(len(m)), -m, v.scalar(bound, "bound")
    )


def kl_worst_case(
    losses: ArrayLike,
    radius: float,
    *,
    weights: ArrayLike | None = None,
    solver: str | None = None,
) -> SolveResult:
    """Maximize q@loss over KL(q||p)<=radius on supplied finite support.

    Returns SolveResult with solution=q and objective=worst expected loss.
    Zero nominal mass excludes that outcome from the ambiguity set.
    """
    from .._backends.cvxpy import kl_worst_case as run

    x = v.array(losses, "losses", ndim=1)
    p = v.weights(weights, len(x))
    radius = v.scalar(radius, "radius", minimum=0)
    return _checked(run(x, p, radius, solver))


def chance_probability(
    losses: ArrayLike, bound: float = 0.0, *, weights: ArrayLike | None = None
) -> float:
    """Return normalized scenario mass for losses <= bound; this is a finite-law probability."""
    x = v.array(losses, "losses", ndim=1)
    return float(v.weights(weights, len(x)) @ (x <= v.scalar(bound, "bound")))
