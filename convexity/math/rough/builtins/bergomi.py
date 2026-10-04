"""Rough Bergomi and its exponential-kernel lifts, with adapted log Euler."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ..._typing import (
    FloatArray,
)
from ...processes import ProcessPaths
from .. import _validation as v
from ..interfaces.base import MemoryKernel
from ..interfaces.results import VolterraPaths
from .drivers import VolterraGaussian
from .kernels import FractionalKernel
from .variance import LognormalVolterraVariance


@dataclass(frozen=True)
class BergomiPaths:
    """Spot/variance paths and retained joint volatility/asset innovations."""

    paths: ProcessPaths
    driver: VolterraPaths
    asset_increments: FloatArray

    def __post_init__(self) -> None:
        if self.paths.values.shape != self.driver.values.shape + (2,):
            raise ValueError("paths must have shape (draw,time,2)")
        if not np.array_equal(self.paths.times, self.driver.times):
            raise ValueError("path and driver times must agree")
        if np.any(self.paths.values <= 0) or self.paths.exact:
            raise ValueError("Bergomi paths must be positive and labeled approximate")
        object.__setattr__(
            self,
            "asset_increments",
            v.shaped(
                self.asset_increments,
                self.driver.brownian_increments.shape,
                "asset_increments",
            ),
        )

    @property
    def times(self):
        return self.paths.times

    @property
    def spot(self):
        return self.paths.values[:, :, 0]

    @property
    def variance(self):
        return self.paths.values[:, :, 1]

    def __init__(
        self,
        paths: ProcessPaths,
        driver: VolterraPaths,
        asset_increments: ArrayLike,
    ) -> None:
        object.__setattr__(self, "paths", paths)
        object.__setattr__(self, "driver", driver)
        object.__setattr__(self, "asset_increments", asset_increments)
        self.__post_init__()


@dataclass(frozen=True)
class VolterraBergomi:
    """dS/S=mu dt+sqrt(v)dB, B=rho W+sqrt(1-rho²)W_perp.

    kernel fixes v's Gaussian driver. No pricing measure is selected implicitly.
    Exact grid variance sampling does not make the spot discretization exact.
    """

    kernel: MemoryKernel
    eta: float = 1.0
    rho: float = -0.7
    xi0: ArrayLike | Callable[[FloatArray], ArrayLike] = 0.04
    mu: float = 0.0

    def __post_init__(self) -> None:
        variance = LognormalVolterraVariance(self.kernel, self.xi0, self.eta)
        object.__setattr__(self, "eta", variance.eta)
        rho = v.scalar(self.rho, "rho")
        if abs(rho) > 1:
            raise ValueError("rho must lie in [-1,1]")
        object.__setattr__(self, "rho", rho)
        object.__setattr__(self, "mu", v.scalar(self.mu, "mu"))

    @property
    def variance_model(self) -> LognormalVolterraVariance:
        return LognormalVolterraVariance(self.kernel, self.xi0, self.eta)

    def sample(
        self,
        initial: float,
        times: ArrayLike,
        *,
        draws: int = 1,
        rng: np.random.Generator,
        method: str = "dense",
    ) -> BergomiPaths:
        initial = v.scalar(initial, "initial", minimum=0, strict=True)
        times = v.times(times)
        draws = v.integer(draws, "draws", 1)
        v.generator(rng)
        driver = VolterraGaussian(self.kernel).sample_joint(
            times, draws=draws, rng=rng, method=method
        )
        dt = np.diff(times)
        variance = self.variance_model.evaluate(times, driver.values)
        independent = (
            rng.standard_normal((draws, len(dt))) * np.sqrt(dt)
            if abs(self.rho) < 1
            else np.zeros((draws, len(dt)))
        )
        asset = (
            self.rho * driver.brownian_increments
            + np.sqrt(1 - self.rho**2) * independent
        )
        # Left endpoint variance is adapted to the joint filtration. Using
        # right endpoint variance would introduce look-ahead and drift bias.
        log_returns = (self.mu - 0.5 * variance[:, :-1]) * dt + np.sqrt(
            variance[:, :-1]
        ) * asset
        log_spot = np.log(initial) + np.column_stack(
            (np.zeros(draws), np.cumsum(log_returns, axis=1))
        )
        with np.errstate(over="ignore", under="ignore"):
            spot = np.exp(log_spot)
        spot[:, 0] = initial
        paths = ProcessPaths(
            times,
            np.stack((spot, variance), axis=-1),
            driver.method + "+left-log-euler",
            driver.backend,
            False,
        )
        return BergomiPaths(paths, driver, asset)


class RoughBergomi(VolterraBergomi):
    """Normalized fractional kernel, 0 < H <= .5, including Brownian limit."""

    def __init__(
        self,
        hurst: float = 0.1,
        eta: float = 1.0,
        rho: float = -0.7,
        xi0: ArrayLike | Callable[[FloatArray], ArrayLike] = 0.04,
        mu: float = 0.0,
    ) -> None:
        super().__init__(
            FractionalKernel(v.hurst(hurst, rough=True)), eta, rho, xi0, mu
        )

    @property
    def hurst(self):
        assert isinstance(self.kernel, FractionalKernel)
        return self.kernel.hurst
