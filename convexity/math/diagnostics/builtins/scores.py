"""Proper scores of finite laws, not sampling-unbiased U-statistic estimates."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from ..._typing import FloatArray

if TYPE_CHECKING:
    from ...simulation import PathScenarios

from ... import _decision as v
from .._backends import scoring


def log_score(log_density: ArrayLike, *, weights: ArrayLike | None = None) -> float:
    """Return the negative weighted mean of supplied log probabilities/densities.

    Smaller is better; values must refer to the same observation coordinates and measure."""
    x = v.array(log_density, "log densities", ndim=1, finite=False)
    w = v.weights(weights, len(x))
    return -float(w[w > 0] @ x[w > 0])


def quantile_score(
    actual: ArrayLike,
    prediction: ArrayLike,
    level: float,
    *,
    weights: ArrayLike | None = None,
) -> float:
    """Return mean pinball loss at a probability level strictly between zero and one.

    Prediction and actual must align; the score uses actual-outcome units."""
    a, p = v.array(actual, "actual", ndim=1), v.array(prediction, "prediction", ndim=1)
    if a.shape != p.shape:
        raise ValueError("actual and predictions must align")
    tau = v.probability(level, "level", open_interval=True)
    error = a - p
    return float(
        v.weights(weights, len(a)) @ np.maximum(tau * error, (tau - 1) * error)
    )


def crps(
    samples: ArrayLike | PathScenarios,
    actual: float,
    *,
    weights: ArrayLike | None = None,
) -> float:
    """Scalar empirical-law CRPS, O(n log n), weighted atoms allowed."""
    x = v.array(samples, "samples", ndim=1)
    return scoring.crps(x, v.scalar(actual, "actual"), v.weights(weights, len(x)))


def energy_score(
    samples: ArrayLike | PathScenarios,
    actual: ArrayLike,
    *,
    weights: ArrayLike | None = None,
    beta: float = 1.0,
) -> float:
    """Joint-vector/path energy score; flatten event coordinates explicitly."""
    from ...simulation import PathScenarios

    if isinstance(samples, PathScenarios):
        if weights is not None:
            raise ValueError("weights already belong to PathScenarios")
        weights = samples.weights
        a = v.array(actual, "actual")
        if a.shape != samples.values.shape[1:]:
            raise ValueError("actual path must match scenario event shape")
        samples, actual = samples.values.reshape(len(samples.values), -1), a.ravel()
    x, y = v.array(samples, "samples", ndim=2), v.array(actual, "actual", ndim=1)
    b = v.scalar(beta, "beta", minimum=0, strict=True)
    if x.shape[1:] != y.shape or b >= 2:
        raise ValueError("event dimensions must align and 0<beta<2")
    return scoring.energy(x, y, v.weights(weights, len(x)), b)


def variogram_score(
    samples: ArrayLike | PathScenarios,
    actual: ArrayLike,
    *,
    weights: ArrayLike | None = None,
    order: float = 0.5,
    pair_weights: ArrayLike | None = None,
) -> float:
    """Score a joint vector scenario law against one observed vector.

    Pair weights and the positive order weight coordinate differences; marginal
    forecasts alone cannot determine this dependence-sensitive score."""
    x, y = v.array(samples, "samples", ndim=2), v.array(actual, "actual", ndim=1)
    p = v.scalar(order, "order", minimum=0, strict=True)
    if x.shape[1:] != y.shape or p >= 2:
        raise ValueError("event dimensions must align and 0<order<2")
    pairs = (
        np.ones((len(y), len(y)))
        if pair_weights is None
        else v.array(pair_weights, "pair weights", ndim=2)
    )
    if (
        pairs.shape != (len(y), len(y))
        or np.any(pairs < 0)
        or not np.allclose(pairs, pairs.T)
    ):
        raise ValueError("pair weights must be nonnegative symmetric event matrix")
    empirical = np.einsum(
        "s,sij->ij", v.weights(weights, len(x)), abs(x[:, :, None] - x[:, None, :]) ** p
    )
    error = (abs(y[:, None] - y[None, :]) ** p - empirical) ** 2
    return float(np.sum(np.triu(pairs * error, 1)))


def pit(
    cdf: ArrayLike,
    *,
    lower_cdf: ArrayLike | None = None,
    rng: np.random.Generator | None = None,
) -> FloatArray:
    """Continuous F(y), or randomized F(y-)+U*(F(y)-F(y-)) for atoms."""
    hi = v.array(cdf, "cdf")
    if np.any(hi < 0) or np.any(hi > 1):
        raise ValueError("CDF values must lie in [0,1]")
    if lower_cdf is None:
        return hi
    lo = v.array(lower_cdf, "lower cdf")
    if lo.shape != hi.shape or np.any(lo < 0) or np.any(lo > hi):
        raise ValueError("lower CDF must align and satisfy 0<=F(y-)<=F(y)")
    return v.array(lo + v.generator(rng).uniform(size=hi.shape) * (hi - lo), "PIT")


def interval_coverage(
    actual: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike,
    *,
    weights: ArrayLike | None = None,
) -> float:
    """Return weighted empirical inclusion frequency for aligned closed intervals.

    Marginal coverage is not simultaneous coverage and does not define a joint law."""
    a, lo, hi = [
        v.array(x, name, ndim=1)
        for x, name in [(actual, "actual"), (lower, "lower"), (upper, "upper")]
    ]
    if a.shape != lo.shape or a.shape != hi.shape or np.any(lo > hi):
        raise ValueError("ordered intervals must align with observations")
    return float(v.weights(weights, len(a)) @ ((a >= lo) & (a <= hi)))
