"""Distributional risk, separate from whole-path risk."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from ..._typing import FloatArray

if TYPE_CHECKING:
    from ...simulation import PathScenarios

from ... import _decision as v
from .._backends.empirical import quantile, tail_weights
from ..interfaces import LossScenarios, RiskContributions


def _law(losses, weights):
    if isinstance(losses, LossScenarios):
        if weights is not None:
            raise ValueError("weights already belong to LossScenarios")
        return losses
    return LossScenarios(losses, weights)


def variance(
    losses: ArrayLike | LossScenarios, *, weights: ArrayLike | None = None
) -> float:
    """Return the population variance of scalar loss scenarios with normalized masses.

    Weights are probabilities, not frequency degrees-of-freedom corrections."""
    law = _law(losses, weights)
    x, w = law.losses, law.weights
    return float(w @ (x - w @ x) ** 2)


def downside_moment(
    returns: ArrayLike,
    *,
    target: float = 0.0,
    order: float = 2.0,
    weights: ArrayLike | None = None,
) -> float:
    """E[max(target-return,0)**order], without conditioning on shortfall."""
    law = _law(returns, weights)
    p = v.scalar(order, "order", minimum=0, strict=True)
    return float(
        law.weights @ np.maximum(v.scalar(target, "target") - law.losses, 0.0) ** p
    )


def value_at_risk(
    losses: ArrayLike | LossScenarios,
    level: float = 0.95,
    *,
    weights: ArrayLike | None = None,
) -> float:
    """Return the lower inverse-CDF loss quantile at level in (0,1).

    Losses are positive for adverse outcomes; finite scenario weights are normalized."""
    law = _law(losses, weights)
    return quantile(law.losses, law.weights, v.probability(level, "level"))


def expected_shortfall(
    losses: ArrayLike | LossScenarios,
    level: float = 0.95,
    *,
    weights: ArrayLike | None = None,
) -> float:
    """Mean of the worst 1-level mass, splitting the atom at VaR exactly."""
    law = _law(losses, weights)
    level = v.probability(level, "level")
    if level == 1:
        raise ValueError("ES requires level < 1")
    return float(tail_weights(law.losses, law.weights, level) @ law.losses)


def volatility_contributions(
    holdings: ArrayLike, covariance: ArrayLike
) -> RiskContributions:
    """Return marginal and component portfolio volatility from holdings and covariance.

    Covariance uses squared per-period return units; components sum to volatility."""
    h = v.array(holdings, "holdings", ndim=1)
    c = v.psd(covariance, "covariance")
    if c.shape != (len(h), len(h)):
        raise ValueError("covariance must align with holdings")
    vol = float(np.sqrt(max(0.0, h @ c @ h)))
    if vol == 0:
        raise ValueError(
            "volatility gradient is undefined at zero portfolio volatility"
        )
    marginal = c @ h / vol
    return RiskContributions(vol, marginal, h * marginal)


def es_contributions(
    holdings: ArrayLike,
    unit_losses: ArrayLike | LossScenarios,
    level: float = 0.95,
    *,
    weights: ArrayLike | None = None,
) -> RiskContributions:
    """Euler ES subgradient; tied tail mass is distributed proportionally."""
    h, x = (
        v.array(holdings, "holdings", ndim=1),
        v.array(unit_losses, "unit losses", ndim=2),
    )
    if x.shape[1] != len(h):
        raise ValueError("unit losses must be (scenario,instrument)")
    law = LossScenarios(x @ h, weights)
    level = v.probability(level, "level")
    if level == 1:
        raise ValueError("ES requires level < 1")
    marginal = tail_weights(law.losses, law.weights, level) @ x
    return RiskContributions(float(h @ marginal), marginal, h * marginal)


def drawdown(wealth: ArrayLike, *, relative: bool = True) -> FloatArray:
    """Running peak minus wealth on the final time axis. Include initial wealth."""
    x = v.array(wealth, "wealth")
    if x.ndim not in (1, 2):
        raise ValueError("wealth must be (time,) or (scenario,time)")
    v.boolean(relative, "relative")
    peak = np.maximum.accumulate(x, axis=-1)
    if relative and np.any(peak <= 0):
        raise ValueError("relative drawdown requires positive running peaks")
    return v.array((peak - x) / peak if relative else peak - x, "drawdown")


def maximum_drawdown(wealth: ArrayLike, *, relative: bool = True) -> FloatArray:
    """Return the maximum peak-to-trough drawdown along the last wealth axis.

    relative=True reports a fraction of running peak; False reports wealth units."""
    return v.array(
        np.max(drawdown(wealth, relative=relative), axis=-1), "maximum drawdown"
    )


def breach_probability(
    paths: ArrayLike | PathScenarios,
    barrier: ArrayLike,
    *,
    direction: str = "below",
    weights: ArrayLike | None = None,
) -> float:
    """Inclusive breach on supplied grid only; no between-grid interpolation."""
    from ...simulation import PathScenarios

    if isinstance(paths, PathScenarios):
        if weights is not None:
            raise ValueError("weights already belong to PathScenarios")
        weights, paths = paths.weights, paths.values
    x = v.array(paths, "paths", ndim=2)
    b = v.vector(barrier, x.shape[1], "barrier")
    v.choice(direction, ("below", "above"), "direction")
    hit = np.any(x <= b if direction == "below" else x >= b, axis=1)
    return float(v.weights(weights, len(x)) @ hit)


def stress(
    values: ArrayLike,
    *,
    shift: ArrayLike = 0.0,
    scale: ArrayLike = 1.0,
    transform: Callable[[FloatArray], ArrayLike] | None = None,
) -> FloatArray:
    """Affine stress or supplied shape-preserving map; no probability inferred."""
    x = v.array(values, "values")
    y = (
        transform(x)
        if transform is not None
        else x * v.array(scale, "scale") + v.array(shift, "shift")
    )
    y = v.array(y, "stressed values")
    if y.shape != x.shape:
        raise ValueError("stress transformation must preserve shape")
    return y


def scenario_margin(
    losses: ArrayLike | LossScenarios,
    *,
    floor: float = 0.0,
    addon: float = 0.0,
    multiplier: float = 1.0,
) -> float:
    """max(floor, multiplier*max(0,worst supplied loss)) + addon."""
    x = v.array(losses, "stress losses", ndim=1)
    floor, addon, multiplier = [
        v.scalar(z, n, minimum=0)
        for z, n in [(floor, "floor"), (addon, "addon"), (multiplier, "multiplier")]
    ]
    return max(floor, multiplier * max(0.0, float(x.max()))) + addon
