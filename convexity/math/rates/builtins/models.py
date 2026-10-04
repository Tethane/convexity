"""QuantLib affine/Gaussian models and empirical curve forecasting composition."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from numpy.typing import ArrayLike

from ..._typing import DateLike

if TYPE_CHECKING:
    from ..._finance_types import (
        NativeDate,
        NativeGSR,
        NativeInflationCurve,
        NativeInflationIndex,
        NativeMarketModel,
    )
    from ...curves import YieldCurve


if TYPE_CHECKING:
    from .. import RatePaths


from dataclasses import dataclass

import numpy as np

from ..._finance import array, choice, positive, scalar
from ...forecasting import Forecaster, TimeSeries
from .._backends import quantlib as _quantlib


@dataclass(frozen=True)
class ShortRateModel:
    """Short-rate dynamics and parameters; time and rate units must agree with the supplied curve convention."""

    kind: str
    parameters: Mapping[str, float]
    curve: YieldCurve | None = None
    measure: str = "Q"

    def __post_init__(self) -> None:
        from types import MappingProxyType

        choice(self.kind, ("vasicek", "cir", "hull_white", "g2"), "rate model")
        choice(self.measure, ("P", "Q"), "measure")
        p = dict(self.parameters)
        required = {
            "vasicek": {"r0", "kappa", "theta", "sigma"},
            "cir": {"r0", "kappa", "theta", "sigma"},
            "hull_white": {"kappa", "sigma"},
            "g2": {"a", "sigma", "b", "eta", "rho"},
        }[self.kind]
        if set(p) != required:
            raise ValueError(f"{self.kind} parameters must be {sorted(required)}")
        for name, value in p.items():
            p[name] = scalar(value, name)
            if name in ("kappa", "sigma", "a", "b", "eta"):
                positive(value, name)
        if self.kind == "cir":
            positive(p["r0"], "r0")
            positive(p["theta"], "theta")
        if self.kind == "g2" and abs(p["rho"]) >= 1:
            raise ValueError("|rho| must be <1")
        if self.kind in ("hull_white", "g2") and (
            self.curve is None or self.measure != "Q"
        ):
            raise ValueError("curve-fitted Hull–White/G2 models require Q and a curve")
        object.__setattr__(self, "parameters", MappingProxyType(p))

    def to_quantlib(self):
        return _quantlib.rate_model(self)

    def discount_bond(self, time: float, maturity: float, state):
        if self.measure != "Q":
            raise ValueError("P dynamics do not define bond prices")
        t = scalar(time, "time", minimum=0)
        end = scalar(maturity, "maturity", minimum=0)
        if end < t:
            raise ValueError("maturity must be >= time")
        native = self.to_quantlib()
        if self.kind == "g2":
            x = array(state, "state", ndim=1)
            if x.shape != (2,):
                raise ValueError("G2 state is (x,y)")
            return native.discountBond(t, end, x.tolist())
        value = scalar(state, "short rate")
        if self.kind == "cir" and value < 0:
            raise ValueError("CIR state must be nonnegative")
        return native.discountBond(t, end, value)

    def process(self):
        """Existing exact transition interface for time-homogeneous P or Q laws."""
        return _quantlib.rate_process(self)


def gaussian_hjm(
    curve: YieldCurve,
    step_dates: Sequence[DateLike | NativeDate],
    volatilities: ArrayLike,
    reversions: ArrayLike,
    *,
    terminal_horizon: float = 60.0,
) -> NativeGSR:
    """QuantLib GSR: one-factor Gaussian HJM with piecewise volatility/reversion.

    Native GSR state uses its terminal forward measure; do not label simulated
    native states as bank-account-measure short rates.
    """
    return _quantlib.gaussian_hjm(
        curve, step_dates, volatilities, reversions, terminal_horizon=terminal_horizon
    )


def market_model_paths(
    model: NativeMarketModel, *, draws: int, seed: int = 42
) -> RatePaths:
    """QuantLib predictor-corrector forward-rate paths under terminal measure.

    model is a native QuantLib MarketModel (e.g. AbcdVol); output axes are
    (draw,evolution step,forward rate). Numeraire weights are returned explicitly.
    """
    return _quantlib.market_model_paths(model, draws=draws, seed=seed)


@dataclass(frozen=True)
class EmpiricalCurveForecaster:
    """Compose an existing multivariate forecaster with tenors; P only."""

    forecaster: Forecaster
    tenors: ArrayLike
    measure = "P"

    def __post_init__(self) -> None:
        t = array(self.tenors, "tenors", ndim=1)
        if t[0] <= 0 or np.any(np.diff(t) <= 0):
            raise ValueError("tenors must be positive and increasing")
        object.__setattr__(self, "tenors", t)

    def fit(self, series: TimeSeries | ArrayLike, *, covariates=None):
        from ...forecasting import TimeSeries, fit

        series = series if isinstance(series, TimeSeries) else TimeSeries(series)
        if series.values.shape[1] != np.size(self.tenors):
            raise ValueError("curve columns must match tenors")
        return fit(series, self.forecaster, covariates=covariates)


def inflation_curve(
    reference_date: DateLike | NativeDate,
    dates: Sequence[DateLike | NativeDate],
    rates: ArrayLike,
    *,
    frequency: str | int | None = None,
    convention: str = "actual365",
    year_on_year: bool = False,
) -> NativeInflationCurve:
    """Construct a native zero or year-on-year inflation curve from dated annual rates.

    Reference date, observation frequency and day-count convention define its clock."""
    return _quantlib.inflation_curve(
        reference_date,
        dates,
        rates,
        frequency=frequency,
        convention=convention,
        year_on_year=year_on_year,
    )


def inflation_index[FixingDateT: DateLike | NativeDate](
    curve: NativeInflationCurve,
    *,
    year_on_year: bool = False,
    fixings: Mapping[FixingDateT, float] | None = None,
) -> NativeInflationIndex:
    """Inflation index; supplied histories are applied only during Convexity valuation."""
    return _quantlib.inflation_index(curve, year_on_year=year_on_year, fixings=fixings)
