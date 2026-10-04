"""Execution costs, fills and assignment on supplied mathematical inputs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy.typing import ArrayLike

from ..._typing import FloatArray, IntArray

if TYPE_CHECKING:
    from .. import LogisticFit


if TYPE_CHECKING:
    from .. import LiquidationResult


from dataclasses import dataclass

import numpy as np

from ... import _decision as v


@dataclass(frozen=True)
class QuadraticCost:
    """Cash cost spread@abs(trade) + trade.T@impact@trade, in trade units."""

    spread: FloatArray
    impact: FloatArray

    def __post_init__(self) -> None:
        c = v.psd(self.impact, "impact")
        s = v.vector(self.spread, len(c), "spread")
        if np.any(s < 0):
            raise ValueError("spread must be nonnegative")
        object.__setattr__(self, "spread", s)
        object.__setattr__(self, "impact", c)

    def __call__(self, trade: ArrayLike) -> float:
        a = v.array(trade, "trade", ndim=1)
        if a.shape != self.spread.shape:
            raise ValueError("trade must align with costs")
        return float(self.spread @ abs(a) + a @ self.impact @ a)

    def __init__(
        self,
        spread: ArrayLike,
        impact: ArrayLike,
    ) -> None:
        object.__setattr__(self, "spread", spread)
        object.__setattr__(self, "impact", impact)
        self.__post_init__()


@dataclass(frozen=True)
class BinomialFills:
    """Conditionally independent counts per leg, with a supplied probability."""

    probability: FloatArray

    def __post_init__(self) -> None:
        p = v.array(self.probability, "probability")
        if p.ndim > 1 or np.any(p < 0) or np.any(p > 1):
            raise ValueError("probability must be a scalar/vector in [0,1]")
        object.__setattr__(self, "probability", p)

    def sample(self, quantities: ArrayLike, *, draws: int, rng: np.random.Generator):
        n = np.asarray(quantities)
        n = v.indices(n, n.shape, "quantities")
        if n.ndim != 1 or np.any(n < 0):
            raise ValueError("quantities must be a nonnegative integer vector")
        p = v.vector(self.probability, len(n), "probability")
        result = v.generator(rng).binomial(
            n, p, size=(v.integer(draws, "draws", 1), len(n))
        )
        return v.indices(result, result.shape, "fills")

    def __init__(
        self,
        probability: ArrayLike,
    ) -> None:
        object.__setattr__(self, "probability", probability)
        self.__post_init__()


def assignment(
    short_quantities: ArrayLike,
    probabilities: ArrayLike,
    *,
    draws: int,
    rng: np.random.Generator,
) -> IntArray:
    """Positive counts of short contracts; no claim of independence across time."""
    return BinomialFills(probabilities).sample(short_quantities, draws=draws, rng=rng)


def fit_assignment(
    features: ArrayLike,
    assigned: ArrayLike,
    trials: ArrayLike,
    *,
    lower_bounds: ArrayLike = -20.0,
    upper_bounds: ArrayLike = 20.0,
) -> LogisticFit:
    """Bounded binomial logistic MLE. Include an intercept column explicitly."""
    from .._backends.statistical import fit

    x = v.array(features, "features", ndim=2)
    y, n = v.array(assigned, "assigned", ndim=1), v.array(trials, "trials", ndim=1)
    if (
        y.shape != (len(x),)
        or n.shape != y.shape
        or np.any(n < 0)
        or not np.any(n > 0)
        or np.any(y < 0)
        or np.any(y > n)
        or np.any(y != np.floor(y))
        or np.any(n != np.floor(n))
    ):
        raise ValueError("assigned/trials must be aligned valid binomial counts")
    lo, hi = (
        v.vector(lower_bounds, x.shape[1], "lower bounds"),
        v.vector(upper_bounds, x.shape[1], "upper bounds"),
    )
    if np.any(lo >= hi):
        raise ValueError("finite coefficient bounds must be strictly ordered")
    return fit(x, y, n, lo, hi)


def liquidate(
    initial: ArrayLike,
    intervals: ArrayLike,
    *,
    spread: ArrayLike = 0.0,
    impact: ArrayLike,
    covariance: ArrayLike,
    aversion: float = 0.0,
    maximum_rate: ArrayLike = np.inf,
    monotone: bool = True,
    solver: str | None = None,
) -> LiquidationResult:
    """Signed inventory, trades reduce it; terminal inventory exactly zero.

    Cost = sum(spread*abs(trade) + trade'impact trade/dt).
    Risk preference = aversion*sum(dt*end_inventory'covariance end_inventory).
    """
    from .._backends.liquidation import solve

    q, dt = v.array(initial, "initial", ndim=1), v.array(intervals, "intervals", ndim=1)
    c, sigma = v.psd(impact, "impact"), v.psd(covariance, "covariance")
    s = v.vector(spread, len(q), "spread")
    rate = v.vector(maximum_rate, len(q), "maximum rate", finite=False)
    if (
        c.shape != (len(q), len(q))
        or sigma.shape != c.shape
        or np.any(dt <= 0)
        or np.any(s < 0)
        or np.any(rate < 0)
    ):
        raise ValueError("invalid dimensions, intervals, spread or rate limits")
    return solve(
        q,
        dt,
        s,
        c,
        sigma,
        v.scalar(aversion, "aversion", minimum=0),
        rate,
        v.boolean(monotone, "monotone"),
        solver,
    )
