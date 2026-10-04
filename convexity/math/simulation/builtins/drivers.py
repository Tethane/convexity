"""Correlated Gaussian vectors and independent-coordinate Brownian drivers."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from ..._typing import FloatArray
from .. import _validation as v
from .._backends.random import normals
from ..interfaces import BrownianIncrements


def _design(draws, method, antithetic, rng):
    draws = v.integer(draws, "draws", 1)
    v.choice(method, ("mc", "sobol", "halton"), "method")
    v.boolean(antithetic, "antithetic")
    v.generator(rng)
    if antithetic and (method != "mc" or draws % 2):
        raise ValueError("antithetic sampling requires mc and an even number of draws")
    if method == "sobol" and draws & (draws - 1):
        raise ValueError("Sobol draws must be a power of two")
    return draws // 2 if antithetic else draws


def gaussian_innovations(
    draws: int,
    covariance: ArrayLike,
    *,
    rng: np.random.Generator,
    method: str = "mc",
    antithetic: bool = False,
) -> FloatArray:
    """N(0,covariance) rows; singular PSD covariance is supported.

    Sobol uses one full-dimensional point per row. Antithetic MC returns base
    rows then their negatives. A covariance, not a correlation, is supplied.
    """
    count = _design(draws, method, antithetic, rng)
    covariance = v.array(covariance, "covariance", ndim=2)
    covariance = v.covariance(covariance, (len(covariance),))
    from ..._backends.gaussian import square_root

    loading = square_root(covariance)
    values = normals(count, len(covariance), method, rng) @ loading.T
    if antithetic:
        values = np.concatenate((values, -values))
    return v.array(values, "innovations")


def brownian_increments(
    times: ArrayLike,
    *,
    draws: int = 1,
    dimension: int = 1,
    rng: np.random.Generator,
    method: str = "mc",
    antithetic: bool = False,
) -> BrownianIncrements:
    """Independent-coordinate Wiener increments on an irregular time grid.

    For QMC each complete path is one point of dimension intervals*dimension,
    with time-major coordinates. Time has the same units as model coefficients.
    """
    times = v.times(times)
    dimension = v.integer(dimension, "dimension", 1)
    count = _design(draws, method, antithetic, rng)
    values = normals(count, (len(times) - 1) * dimension, method, rng)
    values = (
        values.reshape(count, len(times) - 1, dimension)
        * np.sqrt(np.diff(times))[None, :, None]
    )
    if antithetic:
        values = np.concatenate((values, -values))
    sampling = "antithetic" if antithetic else ("iid" if method == "mc" else method)
    return BrownianIncrements(times, values, sampling)
