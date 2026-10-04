"""Convolution memory kernels; singular endpoints are integrated, not sampled."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import quad

from ..._typing import FloatArray
from .. import _validation as v


def _power_difference(lower, upper, exponent):
    if upper == lower:
        return 0.0
    if lower == 0:
        return upper**exponent
    return upper**exponent * -np.expm1(exponent * np.log1p((lower - upper) / upper))


def _decay_integral(rate, duration):
    rate = np.asarray(rate)
    out = np.full(rate.shape, duration, dtype=float)
    np.divide(-np.expm1(-rate * duration), rate, out=out, where=rate != 0)
    return out


@dataclass(frozen=True)
class FractionalKernel:
    """K(u)=sqrt(2H) u**(H-1/2); its Volterra variance is t**(2H)."""

    hurst: float = 0.1

    def __post_init__(self) -> None:
        object.__setattr__(self, "hurst", v.hurst(self.hurst))

    def __call__(self, lag) -> FloatArray:
        lag = v.nonnegative(lag, "lag")
        with np.errstate(divide="ignore"):
            return v.array(
                np.sqrt(2 * self.hurst) * lag ** (self.hurst - 0.5),
                "kernel",
                finite=False,
            )

    def integral(self, lower, upper) -> float:
        lower, upper = v.interval(lower, upper)
        p = self.hurst + 0.5
        return float(np.sqrt(2 * self.hurst) * _power_difference(lower, upper, p) / p)

    def covariance(self, t, s, *, until=None):
        t, s, end = v.overlap(t, s, until)
        if end == 0:
            return 0.0
        short, gap = min(t, s), abs(t - s)
        low = short - end
        if gap == 0:
            return float(_power_difference(low, short, 2 * self.hurst))
        a = self.hurst - 0.5
        ratio = gap / short
        if low == 0:
            value = quad(
                lambda x: (x + ratio) ** a,
                0,
                1,
                weight="alg",
                wvar=(a, 0),
                epsabs=0,
                epsrel=2e-11,
            )[0]
        else:
            value = quad(
                lambda x: x**a * (x + ratio) ** a,
                low / short,
                1,
                epsabs=0,
                epsrel=2e-11,
            )[0]
        return float(2 * self.hurst * short ** (2 * self.hurst) * value)


@dataclass(frozen=True)
class ExponentialKernel:
    """K(u)=sum_i weights[i] exp(-rates[i]*u), with nonnegative rates.

    Signed weights and repeated/zero rates are supported. All factors share W.
    """

    weights: FloatArray
    rates: FloatArray

    def __post_init__(self) -> None:
        weights = v.array(self.weights, "weights", ndim=1)
        rates = v.shaped(self.rates, weights.shape, "rates")
        if np.any(rates < 0):
            raise ValueError("rates must be nonnegative")
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "rates", rates)

    def __call__(self, lag) -> FloatArray:
        lag = v.nonnegative(lag, "lag")
        return v.array(np.exp(-lag[..., None] * self.rates) @ self.weights, "kernel")

    def integral(self, lower, upper) -> float:
        lower, upper = v.interval(lower, upper)
        return float(
            (self.weights * np.exp(-self.rates * lower))
            @ _decay_integral(self.rates, upper - lower)
        )

    def covariance(self, t, s, *, until=None) -> float:
        t, s, end = v.overlap(t, s, until)
        q = _decay_integral(self.rates[:, None] + self.rates[None, :], end)
        return float(
            (self.weights * np.exp(-self.rates * (t - end)))
            @ q
            @ (self.weights * np.exp(-self.rates * (s - end)))
        )

    def factor_process(self):
        """Family-13 exact OU state for continuation from retained factors."""
        from ...processes import LinearDiffusion

        return LinearDiffusion(-np.diag(self.rates), np.ones((len(self.rates), 1)))

    def __init__(
        self,
        weights: ArrayLike,
        rates: ArrayLike,
    ) -> None:
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "rates", rates)
        self.__post_init__()


@dataclass(frozen=True)
class CallableKernel:
    """User kernel with optional exact integral/covariance callbacks.

    Scalar callback values must be finite away from zero. The user guarantees
    local square integrability; SciPy quadrature warnings remain visible.
    """

    function: Callable[[float], float]
    integral_function: Callable[[float, float], float] | None = None
    covariance_function: Callable[[float, float, float], float] | None = None

    def __post_init__(self) -> None:
        for name in ("function", "integral_function", "covariance_function"):
            value = getattr(self, name)
            if not callable(value) and (name == "function" or value is not None):
                raise TypeError(f"{name} must be callable")

    def __call__(self, lag) -> FloatArray:
        lag = v.nonnegative(lag, "lag")
        return v.array(
            np.array(
                [v.scalar(self.function(float(x)), "kernel") for x in lag.flat]
            ).reshape(lag.shape),
            "kernel",
        )

    def integral(self, lower, upper):
        lower, upper = v.interval(lower, upper)
        if lower == upper:
            return 0.0
        value = (
            self.integral_function(lower, upper)
            if self.integral_function is not None
            else quad(lambda u: float(self(u)), lower, upper, epsabs=0, epsrel=2e-10)[0]
        )
        return v.scalar(value, "kernel integral")

    def covariance(self, t, s, *, until=None):
        t, s, end = v.overlap(t, s, until)
        if end == 0:
            return 0.0
        value = (
            self.covariance_function(t, s, end)
            if self.covariance_function is not None
            else quad(
                lambda u: float(self(t - u) * self(s - u)),
                0,
                end,
                epsabs=0,
                epsrel=2e-10,
            )[0]
        )
        return v.scalar(value, "kernel covariance")
