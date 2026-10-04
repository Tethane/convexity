"""Lognormal Volterra variance and full/coarse-filtration forward states."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ..._typing import (
    FloatArray,
)
from .. import _validation as v
from .._backends import gaussian
from ..interfaces.base import MemoryKernel
from .drivers import VolterraGaussian
from .kernels import ExponentialKernel


def curve_values(curve, times):
    raw = curve(times) if callable(curve) else curve
    raw = v.array(raw, "initial forward variance")
    values = np.full(np.shape(times), float(raw)) if raw.ndim == 0 else raw
    values = v.shaped(values, np.shape(times), "initial forward variance")
    if np.any(values <= 0):
        raise ValueError("initial forward variance must be positive")
    return values


@dataclass(frozen=True)
class ForwardVarianceCurve:
    """Positive, log-linearly interpolated initial curve; no extrapolation."""

    times: FloatArray
    values: FloatArray

    def __post_init__(self) -> None:
        times = v.times(self.times)
        values = v.shaped(self.values, times.shape, "values")
        if np.any(values <= 0):
            raise ValueError("forward variances must be positive")
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "values", values)

    def __call__(self, times) -> FloatArray:
        times = v.nonnegative(times, "times")
        if np.any(times > self.times[-1]):
            raise ValueError("forward variance curve does not extrapolate")
        return v.array(
            np.exp(np.interp(times, self.times, np.log(self.values))),
            "forward variance",
        )

    def __init__(
        self,
        times: ArrayLike,
        values: ArrayLike,
    ) -> None:
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "values", values)
        self.__post_init__()


@dataclass(frozen=True)
class ForwardVarianceState:
    """xi_t(T) on a maturity grid, shape (draw,maturity), with filtration label.

    A finite maturity snapshot alone does not permit exact rough continuation.
    Retain the underlying Brownian history or all OU factors for future updates.
    """

    time: float
    maturities: FloatArray
    values: FloatArray
    method: str

    def __post_init__(self) -> None:
        time = v.scalar(self.time, "time", minimum=0)
        maturities = v.ordered_times(self.maturities)
        values = v.array(self.values, "values", ndim=2)
        if maturities[0] < time:
            raise ValueError("maturities must be >= observation time")
        if (
            values.shape[0] < 1
            or values.shape[1] != len(maturities)
            or np.any(values <= 0)
        ):
            raise ValueError(
                "forward variance values must be positive with shape (draw,maturity)"
            )
        v.label(self.method, "method")
        for name, value in (
            ("time", time),
            ("maturities", maturities),
            ("values", values),
        ):
            object.__setattr__(self, name, value)

    def __init__(
        self,
        time: float,
        maturities: ArrayLike,
        values: ArrayLike,
        method: str,
    ) -> None:
        object.__setattr__(self, "time", time)
        object.__setattr__(self, "maturities", maturities)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "method", method)
        self.__post_init__()


@dataclass(frozen=True)
class LognormalVolterraVariance:
    """v(t)=xi0(t)*exp(eta*X(t)-eta**2*Var(X(t))/2)."""

    kernel: MemoryKernel
    xi0: ArrayLike | Callable[[FloatArray], ArrayLike] = 0.04
    eta: float = 1.0

    def __post_init__(self) -> None:
        VolterraGaussian(self.kernel)
        object.__setattr__(self, "eta", v.scalar(self.eta, "eta", minimum=0))
        curve_values(self.xi0, np.array([0.0]))

    def evaluate(self, times: ArrayLike, driver_values: ArrayLike):
        times = v.times(times)
        values = v.array(driver_values, "driver_values", ndim=2)
        if values.shape[1] != len(times) or np.any(values[:, 0] != 0):
            raise ValueError(
                "driver_values must have shape (draw,time) and start at zero"
            )
        variance = np.array([self.kernel.covariance(t, t) for t in times])
        return self._values(times, values, variance)

    def _values(self, maturities, mean, explained_variance):
        # Log coordinates avoid multiplying a tiny initial curve by exp(large).
        with np.errstate(over="ignore", under="ignore"):
            values = np.exp(
                np.log(curve_values(self.xi0, maturities))
                + self.eta * mean
                - 0.5 * self.eta**2 * explained_variance
            )
        values = v.array(values, "variance values")
        if np.any(values <= 0):
            raise ValueError("variance values underflowed; rescale model parameters")
        return values

    def _maturities(self, time, maturities):
        time = v.scalar(time, "time", minimum=0)
        maturities = v.ordered_times(maturities)
        if maturities[0] < time:
            raise ValueError("maturities must be >= observation time")
        return time, maturities

    def sample_forward(
        self, time: float, maturities, *, draws: int = 1, rng: np.random.Generator
    ) -> ForwardVarianceState:
        """Unconditional draws of the full Brownian-filtration state at time t.

        Samples integral_0^t K(T-s)dW(s) jointly across T. This is a new joint
        draw, not an update of previously sampled spot/variance paths.
        """
        time, maturities = self._maturities(time, maturities)
        draws = v.integer(draws, "draws", 1)
        v.generator(rng)
        covariance = np.array(
            [
                [self.kernel.covariance(t, s, until=time) for s in maturities]
                for t in maturities
            ]
        )
        past = gaussian.sample(covariance, draws, rng)
        values = self._values(maturities, past, np.diag(covariance))
        return ForwardVarianceState(
            time, maturities, values, "full-brownian-filtration"
        )

    def forward_from_brownian(
        self, times: ArrayLike, increments: ArrayLike, maturities
    ) -> ForwardVarianceState:
        """E[v(T)|observed Brownian increments], not a full-history plug-in.

        The exponent compensator uses the variance explained by those
        increments. Unobserved Brownian bridges are integrated out exactly.
        """
        times = v.times(times)
        time, maturities = self._maturities(times[-1], maturities)
        increments = v.array(increments, "increments", empty=True)
        if increments.ndim == 1:
            increments = increments[None, :]
        if (
            increments.ndim != 2
            or increments.shape[0] < 1
            or increments.shape[1] != len(times) - 1
        ):
            raise ValueError(
                "increments must have shape (interval,) or (draw,interval)"
            )
        dt = np.diff(times)
        loadings = np.empty((len(maturities), len(dt)))
        for i, maturity in enumerate(maturities):
            for j in range(len(dt)):
                loadings[i, j] = (
                    self.kernel.integral(maturity - times[j + 1], maturity - times[j])
                    / dt[j]
                )
        past = increments @ loadings.T
        explained = (loadings**2) @ dt
        return ForwardVarianceState(
            time,
            maturities,
            self._values(maturities, past, explained),
            "observed-brownian-increments",
        )

    def forward_from_factors(
        self, time: float, factors: ArrayLike, maturities
    ) -> ForwardVarianceState:
        """Exact full-filtration forward curve of the supplied exponential lift."""
        if not isinstance(self.kernel, ExponentialKernel):
            raise ValueError("factor state requires an ExponentialKernel")
        time, maturities = self._maturities(time, maturities)
        factors = v.array(factors, "factors")
        if factors.ndim == 1:
            factors = factors[None, :]
        if factors.ndim != 2 or factors.shape[1] != len(self.kernel.rates):
            raise ValueError("factors must have shape (factor,) or (draw,factor)")
        if time == 0 and np.any(factors != 0):
            raise ValueError("factors at time zero must be zero")
        loading = (
            np.exp(-(maturities[:, None] - time) * self.kernel.rates)
            * self.kernel.weights
        )
        past = factors @ loading.T
        explained = np.array(
            [self.kernel.covariance(t, t, until=time) for t in maturities]
        )
        return ForwardVarianceState(
            time,
            maturities,
            self._values(maturities, past, explained),
            "full-factor-state",
        )
