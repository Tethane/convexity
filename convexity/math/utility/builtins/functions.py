"""Utility is applied to wealth; penalties are preferences, not cash debits."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy.typing import ArrayLike

from ..._typing import FloatArray

if TYPE_CHECKING:
    from .. import Utility


from dataclasses import dataclass

import numpy as np

from ... import _decision as v
from ...risk import expected_shortfall, variance
from .._backends.functions import exponential_ce, smooth_inverse


@dataclass(frozen=True)
class LinearUtility:
    """Affine utility of supplied wealth; coordinate units and any normalization are caller choices."""

    def __call__(self, wealth: ArrayLike) -> FloatArray:
        return v.array(wealth, "wealth")

    def inverse(self, utility: ArrayLike) -> FloatArray:
        return v.array(utility, "utility")


@dataclass(frozen=True)
class LogUtility:
    """Logarithmic wealth utility on its positive domain."""

    def __call__(self, wealth: ArrayLike) -> FloatArray:
        x = v.array(wealth, "wealth")
        if np.any(x <= 0):
            raise ValueError("log utility requires strictly positive wealth")
        return v.array(np.log(x), "utility")

    def inverse(self, utility: ArrayLike) -> FloatArray:
        return v.array(np.exp(v.array(utility, "utility")), "wealth")


@dataclass(frozen=True)
class ExponentialUtility:
    """Exponential utility with constant absolute risk aversion in reciprocal wealth units."""

    aversion: float = 1.0

    def __post_init__(self) -> None:
        v.scalar(self.aversion, "aversion", minimum=0, strict=True)

    def __call__(self, wealth: ArrayLike) -> FloatArray:
        return v.array(-np.exp(-self.aversion * v.array(wealth, "wealth")), "utility")

    def inverse(self, utility: ArrayLike) -> FloatArray:
        y = v.array(utility, "utility")
        if np.any(y >= 0):
            raise ValueError("exponential utility must be negative")
        return v.array(-np.log(-y) / self.aversion, "wealth")


@dataclass(frozen=True)
class PowerUtility:
    """CRRA: (w**(1-gamma)-1)/(1-gamma), log at gamma=1; w>0."""

    aversion: float = 2.0

    def __post_init__(self) -> None:
        v.scalar(self.aversion, "aversion", minimum=0)

    def __call__(self, wealth: ArrayLike) -> FloatArray:
        x = v.array(wealth, "wealth")
        if np.any(x <= 0):
            raise ValueError("power utility requires strictly positive wealth")
        g = self.aversion
        return v.array(
            np.log(x) if g == 1 else np.expm1((1 - g) * np.log(x)) / (1 - g), "utility"
        )

    def inverse(self, utility: ArrayLike):
        y = v.array(utility, "utility")
        if self.aversion == 1:
            return v.array(np.exp(y), "wealth")
        if np.any(1 + (1 - self.aversion) * y <= 0):
            raise ValueError("utility outside power utility range")
        return v.array(
            np.exp(np.log1p((1 - self.aversion) * y) / (1 - self.aversion)), "wealth"
        )


@dataclass(frozen=True)
class SmoothUtility:
    """Options sheet O5: w - aversion*sqrt(scale**2+w**2)."""

    aversion: float = 0.5
    scale: float = 1.0

    def __post_init__(self) -> None:
        v.probability(self.aversion, "aversion", open_interval=True)
        v.scalar(self.scale, "scale", minimum=0, strict=True)

    def __call__(self, wealth: ArrayLike) -> FloatArray:
        x = v.array(wealth, "wealth")
        return v.array(x - self.aversion * np.hypot(self.scale, x), "utility")

    def inverse(self, utility: ArrayLike) -> FloatArray:
        return v.array(
            smooth_inverse(v.array(utility, "utility"), self.scale, self.aversion),
            "wealth",
        )


def expected_utility(
    wealth: ArrayLike, utility: Utility, *, weights: ArrayLike | None = None
) -> float:
    """Average utility(wealth) under normalized nonnegative scenario masses.

    Utility domain errors remain explicit; weights describe a finite probability law."""
    x = v.array(wealth, "wealth", ndim=1)
    w = v.weights(weights, len(x))
    y = v.array(utility(x[w > 0]), "utility", ndim=1)
    if y.shape != x[w > 0].shape:
        raise ValueError("utility must preserve shape")
    return float(w[w > 0] @ y)


def certainty_equivalent(
    wealth: ArrayLike, utility: Utility, *, weights: ArrayLike | None = None
) -> float:
    """Return inverse_utility(E[utility(wealth)]) in wealth units.

    The supplied Utility must implement an inverse on the attainable utility domain."""
    x = v.array(wealth, "wealth", ndim=1)
    w = v.weights(weights, len(x))
    if isinstance(utility, ExponentialUtility):
        return exponential_ce(x, w, utility.aversion)
    return v.scalar(
        utility.inverse(expected_utility(x, utility, weights=w)), "certainty equivalent"
    )


def mean_variance(
    gains: ArrayLike, aversion: float = 1.0, *, weights: ArrayLike | None = None
) -> float:
    """Return E[gain] - aversion*Var(gain)/2 under scenario masses.

    Aversion has inverse gain units; this is a preference criterion, not expected utility."""
    x = v.array(gains, "gains", ndim=1)
    a = v.scalar(aversion, "aversion", minimum=0)
    return float(
        v.weights(weights, len(x)) @ x - 0.5 * a * variance(x, weights=weights)
    )


def mean_cvar(
    gains: ArrayLike,
    aversion: float = 1.0,
    level: float = 0.95,
    *,
    weights: ArrayLike | None = None,
) -> float:
    """Return E[gain] - aversion*CVaR(-gain) at the supplied confidence level.

    Gains are positive for favorable outcomes; aversion is dimensionless."""
    x = v.array(gains, "gains", ndim=1)
    a = v.scalar(aversion, "aversion", minimum=0)
    return float(
        v.weights(weights, len(x)) @ x
        - a * expected_shortfall(-x, level, weights=weights)
    )


def preference_penalty(
    trade: ArrayLike,
    *,
    turnover: float = 0.0,
    switching: float = 0.0,
    scale: ArrayLike = 1.0,
) -> float:
    """Return turnover and fixed switching penalties for a trade vector.

    Scale broadcasts to trade coordinates and must be positive; coefficients
    set the penalty units and must be nonnegative."""
    a = v.array(trade, "trade", ndim=1)
    d = v.vector(scale, len(a), "scale")
    if np.any(d <= 0):
        raise ValueError("trade scales must be positive")
    t, s = (
        v.scalar(turnover, "turnover", minimum=0),
        v.scalar(switching, "switching", minimum=0),
    )
    return float(t * np.sum(d * abs(a)) + s * np.any(a != 0))
