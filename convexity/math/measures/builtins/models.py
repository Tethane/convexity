"""Model-specific diffusion changes and stable scenario reweighting."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy.typing import ArrayLike

from ..._typing import FloatArray

if TYPE_CHECKING:
    from ...simulation import PathScenarios


from dataclasses import dataclass

import numpy as np

from ..._finance import array, positive, scalar
from ...curves.interfaces.base import DiscountCurve
from ...processes import CIR, GeometricBrownianMotion, Heston, OrnsteinUhlenbeck
from .._backends import scipy as _scipy
from ..interfaces import MeasureModel


@dataclass(frozen=True)
class MoneyMarketNumeraire:
    """Deterministic money-market numeraire exp(rate * time) for a constant continuously compounded rate."""

    rate: float

    def __post_init__(self) -> None:
        scalar(self.rate, "rate")

    def value(self, time: float):
        t = array(time, "time")
        if np.any(t < 0):
            raise ValueError("time must be nonnegative")
        return np.exp(self.rate * t)


@dataclass(frozen=True)
class CurveNumeraire:
    """Deterministic bank account implied by a discount curve."""

    curve: DiscountCurve

    def value(self, time: float):
        return 1 / self.curve.discount(time)


@dataclass(frozen=True)
class DiffusionPair:
    """Compatible named P/Q diffusions; dW_Q = dW_P + lambda dt.

    Compatibility is checked at construction for supported model families, and
    the drift-span condition is checked at every market_price_of_risk call.
    This does not by itself establish Novikov/integrability conditions.
    """

    physical: MeasureModel[GeometricBrownianMotion | CIR | OrnsteinUhlenbeck | Heston]
    pricing: MeasureModel[GeometricBrownianMotion | CIR | OrnsteinUhlenbeck | Heston]

    def __post_init__(self) -> None:
        p, q = (self.physical, self.pricing)
        if p.measure != "P" or q.measure != "Q":
            raise ValueError("pair requires P followed by Q")
        if type(p.model) is not type(q.model):
            raise ValueError("paired models must have the same diffusion family")
        if isinstance(p.model, (GeometricBrownianMotion, CIR, OrnsteinUhlenbeck)):
            names = ("sigma",)
        elif isinstance(p.model, Heston):
            names = ("sigma", "rho")
        else:
            raise TypeError("supported paired diffusions are GBM, OU, CIR and Heston")
        if any((getattr(p.model, n) != getattr(q.model, n) for n in names)):
            raise ValueError("measure changes cannot change diffusion coefficients")

    def market_price_of_risk(self, time: float, state):
        return _scipy.market_price_of_risk(self, time, state)


def gbm_pair(mu: float, rate: float, dividend: float, sigma: float) -> DiffusionPair:
    """Construct explicitly supplied physical and pricing GBM laws with a money-market numeraire.

    mu, rate, dividend are annual drifts; sigma is annual volatility. Q drift is
    rate-dividend. This constructor records assumptions, it does not estimate a change of measure."""
    return DiffusionPair(
        MeasureModel(GeometricBrownianMotion(mu, sigma), "P"),
        MeasureModel(
            GeometricBrownianMotion(rate - dividend, sigma),
            "Q",
            MoneyMarketNumeraire(rate),
        ),
    )


def girsanov_log_density(
    brownian_increments: ArrayLike, risk: ArrayLike, dt: ArrayLike
) -> FloatArray:
    """Euler stochastic exponential log(dQ/dP), (draw,step,noise).

    risk broadcasts to increments; it must be predictable (evaluated BEFORE
    each increment). Constant risk is exact; state-dependent risk is discretized.
    """
    dw = array(brownian_increments, "increments", ndim=3)
    lam = np.broadcast_to(array(risk, "risk"), dw.shape)
    times = np.broadcast_to(array(dt, "dt"), (dw.shape[1],))
    if np.any(times <= 0):
        raise ValueError("time increments must be positive")
    return array(
        -np.sum(lam * dw + 0.5 * lam**2 * times[None, :, None], axis=(1, 2)),
        "log density",
    )


def reweight(scenarios: PathScenarios, log_likelihood: ArrayLike) -> PathScenarios:
    """Self-normalized importance law; no IID error estimate is asserted."""
    return _scipy.reweight(scenarios, log_likelihood)


def change_numeraire_weights(
    old_numeraire_terminal: ArrayLike,
    new_numeraire_terminal: ArrayLike,
    old_initial: float,
    new_initial: float,
) -> FloatArray:
    """dQ_new/dQ_old = (N_new(T)/N_old(T))/(N_new(0)/N_old(0))."""
    old, new = (
        array(old_numeraire_terminal, "old numeraire"),
        array(new_numeraire_terminal, "new numeraire"),
    )
    if np.any(old <= 0) or np.any(new <= 0):
        raise ValueError("numeraires must be positive")
    return (
        new
        / old
        * positive(old_initial, "old initial")
        / positive(new_initial, "new initial")
    )
