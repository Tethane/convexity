"""Implied inversion, SVI/SSVI/SABR and QuantLib volatility structures."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from numpy.typing import ArrayLike

from ..._typing import DateLike, FloatArray

if TYPE_CHECKING:
    from ..._finance_types import (
        NativeDate,
        NativeVolatility,
    )
    from ...curves import YieldCurve


from dataclasses import dataclass

import numpy as np

from ..._finance import array, integer, positive, scalar
from .._backends import quantlib as _quantlib
from .._backends import scipy as _scipy
from ..interfaces import ArbitrageReport


def implied_volatility(
    price: float,
    forward: float,
    strike: float,
    maturity: float,
    *,
    discount: float = 1.0,
    kind: str = "call",
    model: str = "black76",
    displacement: float = 0.0,
    tolerance: float = 1e-12,
) -> float:
    """Invert a discounted forward-option price for its annualized volatility.

    Maturity is years. Black-76 volatility is 1/sqrt(year); normal volatility is
    price/sqrt(year). Return a scalar; unattainable prices or invalid domains fail."""
    return _scipy.implied_volatility(
        price,
        forward,
        strike,
        maturity,
        discount=discount,
        kind=kind,
        model=model,
        displacement=displacement,
        tolerance=tolerance,
    )


@dataclass(frozen=True)
class SVI:
    """Raw SVI w(k)=a+b[rho(k-m)+sqrt((k-m)^2+sigma^2)]."""

    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def __post_init__(self) -> None:
        scalar(self.a, "a")
        scalar(self.m, "m")
        scalar(self.b, "b", minimum=0)
        positive(self.sigma, "sigma")
        if abs(scalar(self.rho, "rho")) >= 1:
            raise ValueError("|rho| must be < 1")
        if self.a + self.b * self.sigma * np.sqrt(1 - self.rho**2) <= 0:
            raise ValueError("SVI must have strictly positive minimum variance")

    def __call__(self, k):
        x = array(k, "log moneyness") - self.m
        return self.a + self.b * (self.rho * x + np.sqrt(x * x + self.sigma**2))

    def derivatives(self, k):
        x = array(k, "log moneyness") - self.m
        root = np.sqrt(x * x + self.sigma**2)
        return (self.b * (self.rho + x / root), self.b * self.sigma**2 / root**3)

    def butterfly(self, k):
        k = array(k, "log moneyness")
        w = self(k)
        wp, wpp = self.derivatives(k)
        return (1 - k * wp / (2 * w)) ** 2 - wp**2 / 4 * (1 / w + 0.25) + wpp / 2

    @classmethod
    def fit(
        cls,
        k,
        variance,
        *,
        initial: ArrayLike = (0.02, 0.1, -0.2, 0.0, 0.2),
        constraint_grid=None,
    ):
        """Constrain positivity, Lee wing slopes, and g(k)>=0 on a declared grid."""
        return _scipy.fit_svi(
            cls, k, variance, initial=initial, constraint_grid=constraint_grid
        )


@dataclass(frozen=True)
class SSVI:
    """Power-law SSVI phi(theta)=eta/theta**gamma, on a bounded theta domain.

    validate_domain checks sufficient butterfly bounds and calendar monotonicity
    for 0<=gamma<=1/2, |rho|<1 and nondecreasing ATM total variance.
    """

    rho: float
    eta: float
    gamma: float = 0.5

    def __post_init__(self) -> None:
        if (
            abs(scalar(self.rho, "rho")) >= 1
            or not 0 <= scalar(self.gamma, "gamma") <= 0.5
        ):
            raise ValueError("|rho|<1 and 0<=gamma<=1/2 required")
        positive(self.eta, "eta")

    def __call__(self, theta, k):
        theta = positive(theta, "ATM variance")
        z = self.eta / theta**self.gamma * array(k, "log moneyness")
        return (
            theta
            / 2
            * (1 + self.rho * z + np.sqrt((z + self.rho) ** 2 + 1 - self.rho**2))
        )

    def validate_domain(self, theta_max):
        theta = positive(theta_max, "maximum ATM variance")
        if (
            self.eta * theta ** (1 - self.gamma) * (1 + abs(self.rho)) >= 4
            or self.eta**2 * theta ** (1 - 2 * self.gamma) * (1 + abs(self.rho)) > 4
        ):
            raise ValueError(
                "SSVI sufficient butterfly constraints fail on this domain"
            )
        return True


@dataclass(frozen=True)
class SmileSurface:
    """Linear interpolation of SVI total variance between supplied expiries.

    Extrapolation is rejected. Calendar and butterfly reports inspect the supplied
    grid, including intermediate maturities; slice constraints alone do not
    certify arbitrary interpolated surfaces.
    """

    times: FloatArray
    smiles: tuple[SVI, ...]

    def __post_init__(self) -> None:
        t = array(self.times, "times", ndim=1)
        if (
            len(t) < 2
            or t[0] <= 0
            or np.any(np.diff(t) <= 0)
            or (len(self.smiles) != len(t))
        ):
            raise ValueError("positive ordered expiries must match smiles")
        object.__setattr__(self, "times", t)
        object.__setattr__(self, "smiles", tuple(self.smiles))

    def total_variance(self, time: float, log_moneyness):
        t = scalar(time, "time")
        if not self.times[0] <= t <= self.times[-1]:
            raise ValueError("surface time extrapolation is disabled")
        i = min(np.searchsorted(self.times, t, side="right") - 1, len(self.times) - 2)
        alpha = (t - self.times[i]) / (self.times[i + 1] - self.times[i])
        return (1 - alpha) * self.smiles[i](log_moneyness) + alpha * self.smiles[i + 1](
            log_moneyness
        )

    def volatility(self, time: float, log_moneyness):
        return np.sqrt(
            self.total_variance(time, log_moneyness) / positive(time, "time")
        )

    def diagnose(self, log_moneyness, *, time_points: int = 21) -> ArbitrageReport:
        k = array(log_moneyness, "log moneyness", ndim=1)
        time_points = integer(time_points, "time_points", 2)
        times = np.unique(
            np.r_[self.times, np.linspace(self.times[0], self.times[-1], time_points)]
        )
        ws = []
        gs = []
        for t in times:
            i = min(
                np.searchsorted(self.times, t, side="right") - 1, len(self.times) - 2
            )
            a = (t - self.times[i]) / (self.times[i + 1] - self.times[i])
            w = self.total_variance(t, k)
            p0, pp0 = self.smiles[i].derivatives(k)
            p1, pp1 = self.smiles[i + 1].derivatives(k)
            wp = (1 - a) * p0 + a * p1
            wpp = (1 - a) * pp0 + a * pp1
            gs.append(
                (1 - k * wp / (2 * w)) ** 2 - wp**2 / 4 * (1 / w + 0.25) + wpp / 2
            )
            ws.append(w)
        return ArbitrageReport(gs, np.diff(ws, axis=0), float(np.min(ws)))

    def __init__(
        self,
        times: ArrayLike,
        smiles: tuple[SVI, ...],
    ) -> None:
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "smiles", smiles)
        self.__post_init__()


def sabr_volatility(
    strike: float,
    forward: float,
    maturity: float,
    alpha: float,
    beta: float,
    nu: float,
    rho: float,
    *,
    model: str = "black76",
    shift: float = 0.0,
) -> float:
    """Evaluate the SABR implied-volatility approximation at a strike.

    Maturity is years, beta in [0,1], rho in (-1,1); shift is in forward-price units.
    model chooses lognormal Black-76 or normal volatility units, not an exact law."""
    return _quantlib.sabr_volatility(
        strike, forward, maturity, alpha, beta, nu, rho, model=model, shift=shift
    )


def black_variance_surface(
    reference_date: DateLike | NativeDate,
    dates: Sequence[DateLike | NativeDate],
    strikes: ArrayLike,
    volatilities: ArrayLike,
    *,
    convention: str = "actual365",
    extrapolate: bool = False,
) -> NativeVolatility:
    """QuantLib surface; vols shape (strike,expiry), bilinear total variance."""
    return _quantlib.black_variance_surface(
        reference_date,
        dates,
        strikes,
        volatilities,
        convention=convention,
        extrapolate=extrapolate,
    )


def dupire_local_volatility(
    surface: NativeVolatility,
    discount: YieldCurve,
    dividend: YieldCurve,
    spot: float,
    time: float,
    strike: float,
) -> float:
    """Compute local volatility from a compatible implied-volatility surface and curves.

    Time is years, strike/spot are prices. Numerical surface derivatives require
    smoothness and arbitrage consistency; a negative local variance is an error."""
    return _quantlib.dupire_local_volatility(
        surface, discount, dividend, spot, time, strike
    )


def risk_neutral_density(
    call_price: Callable[[float], float],
    strikes: ArrayLike,
    discount: float,
    *,
    step: float = 0.01,
) -> FloatArray:
    """Breeden–Litzenberger finite difference C_KK/D; negative values are exposed."""
    k = array(strikes, "strikes")
    h = positive(step, "step")
    return array(
        [
            (
                call_price(float(x + h))
                - 2 * call_price(float(x))
                + call_price(float(x - h))
            )
            / h**2
            / positive(discount, "discount")
            for x in k.reshape(-1)
        ],
        "density",
    ).reshape(k.shape)
