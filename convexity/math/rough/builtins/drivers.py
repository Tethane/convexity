"""Scalar Gaussian histories, including their joint driving Brownian motion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ..._typing import FloatArray
from ...processes import ProcessPaths
from .. import _validation as v
from .._backends import gaussian
from ..interfaces.base import MemoryKernel
from ..interfaces.results import VolterraPaths
from .kernels import ExponentialKernel, _decay_integral


@dataclass(frozen=True)
class FractionalBrownianMotion:
    """Standard fBm; stationary increments, Var(B_H(t))=t**(2H).

    This is distinct from the Riemann--Liouville fractional Volterra process.
    """

    hurst: float = 0.1

    def __post_init__(self) -> None:
        object.__setattr__(self, "hurst", v.hurst(self.hurst))

    def covariance(self, times: ArrayLike) -> FloatArray:
        times = v.times(times)
        p = 2 * self.hurst
        cov = 0.5 * (
            times[:, None] ** p
            + times[None, :] ** p
            - abs(times[:, None] - times[None, :]) ** p
        )
        return v.covariance(cov, (len(times),))

    def sample(
        self, times: ArrayLike, *, draws: int = 1, rng: np.random.Generator
    ) -> ProcessPaths:
        times = v.times(times)
        draws = v.integer(draws, "draws", 1)
        v.generator(rng)
        values = gaussian.sample(self.covariance(times), draws, rng)
        return ProcessPaths(times, values, "dense-gaussian", "numpy/scipy", True)


@dataclass(frozen=True)
class VolterraGaussian:
    """X(t)=integral_0^t K(t-s)dW(s), with its entire grid history."""

    kernel: MemoryKernel

    def __post_init__(self) -> None:
        if not all(
            callable(getattr(self.kernel, name, None))
            for name in ("__call__", "integral", "covariance")
        ):
            raise TypeError("kernel must implement MemoryKernel")

    def covariance(self, times: ArrayLike) -> FloatArray:
        times = v.times(times)
        cov = np.empty((len(times), len(times)))
        for i, t in enumerate(times):
            for j in range(i + 1):
                cov[i, j] = cov[j, i] = self.kernel.covariance(t, times[j])
        return v.covariance(cov, (len(times),))

    def brownian_covariance(self, times: ArrayLike) -> FloatArray:
        """Cov(X(t_i), W(t_{j+1})-W(t_j)), shape (time,interval)."""
        times = v.times(times)
        cross = np.zeros((len(times), len(times) - 1))
        for i, t in enumerate(times):
            for j in range(i):
                cross[i, j] = self.kernel.integral(t - times[j + 1], t - times[j])
        return v.array(cross, "cross covariance", empty=True)

    def sample(self, times, *, draws: int = 1, rng: np.random.Generator):
        return self.sample_joint(times, draws=draws, rng=rng).paths()

    def sample_joint(
        self,
        times: ArrayLike,
        *,
        draws: int = 1,
        rng: np.random.Generator,
        method: str = "dense",
    ):
        times = v.times(times)
        draws = v.integer(draws, "draws", 1)
        v.generator(rng)
        if method == "factors":
            if not isinstance(self.kernel, ExponentialKernel):
                raise ValueError("factor sampling requires an ExponentialKernel")
            return _sample_factors(self.kernel, times, draws, rng)
        if method != "dense":
            raise ValueError("method must be 'dense' or 'factors'")
        n = len(times)
        cross = self.brownian_covariance(times)
        cov = np.block(
            [[self.covariance(times), cross], [cross.T, np.diag(np.diff(times))]]
        )
        values = gaussian.sample(cov, draws, rng)
        return VolterraPaths(
            times, values[:, :n], values[:, n:], "dense-gaussian", True
        )


def _sample_factors(kernel, times, draws, rng):
    m = len(kernel.rates)
    factors = np.zeros((draws, len(times), m))
    increments = np.zeros((draws, len(times) - 1))
    # Every factor is driven by the same W; independent OU draws are incorrect.
    roots = {}
    for j, dt in enumerate(np.diff(times)):
        if dt not in roots:
            cov = np.empty((m + 1, m + 1))
            cov[:m, :m] = _decay_integral(
                kernel.rates[:, None] + kernel.rates[None, :], dt
            )
            cov[:m, m] = cov[m, :m] = _decay_integral(kernel.rates, dt)
            cov[m, m] = dt
            roots[dt] = gaussian.root(cov)
        innovations = rng.standard_normal((draws, m + 1)) @ roots[dt].T
        factors[:, j + 1] = (
            factors[:, j] * np.exp(-kernel.rates * dt) + innovations[:, :m]
        )
        increments[:, j] = innovations[:, m]
    return VolterraPaths(
        times, factors @ kernel.weights, increments, "exact-ou-factors", True, factors
    )
