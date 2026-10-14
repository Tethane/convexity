"""Structural contracts: a scalar value is not a sufficient rough state."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike

from ..._typing import FloatArray
from ...processes import ProcessPaths
from .. import _validation as v


class MemoryKernel(Protocol):
    """Locally square-integrable convolution kernel, with checked integrals.

    covariance(t,s,until) = integral_0^until K(t-u) K(s-u) du.
    The default cutoff is min(t,s). Singular K(0) is allowed.
    """

    def __call__(self, lag: ArrayLike) -> FloatArray: ...
    def integral(self, lower: float, upper: float) -> float: ...
    def covariance(
        self, t: float, s: float, *, until: float | None = None
    ) -> float: ...


class GaussianDriver(Protocol):
    """Gaussian driving-noise extension with declared covariance and reproducible sampling behavior."""

    def covariance(self, times: ArrayLike) -> FloatArray: ...
    def sample(
        self, times: ArrayLike, *, draws: int = 1, rng: np.random.Generator
    ) -> ProcessPaths: ...


def sample_driver(
    driver: GaussianDriver,
    times: ArrayLike,
    *,
    draws: int = 1,
    rng: np.random.Generator,
) -> ProcessPaths:
    """Checked structural dispatch for a user-supplied scalar Gaussian driver."""
    grid = v.times(times)
    draws = v.integer(draws, "draws", 1)
    v.generator(rng)
    result = driver.sample(grid, draws=draws, rng=rng)
    if not isinstance(result, ProcessPaths):
        raise TypeError("driver.sample must return ProcessPaths")
    if result.values.shape != (draws, len(grid)) or not (result.times == grid).all():
        raise ValueError("driver paths must preserve grid and (draw,time) shape")
    return result
