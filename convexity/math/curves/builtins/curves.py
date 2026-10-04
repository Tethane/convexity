"""QuantLib yield curves, bootstraps and deterministic cost of carry."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING

from numpy.typing import ArrayLike

from ..._typing import DateLike, FloatArray

if TYPE_CHECKING:
    from ..._finance_types import (
        NativeDate,
        NativeRateHelper,
    )
    from ...contracts import CashFlow
    from ...curves import YieldCurve


if TYPE_CHECKING:
    from .. import CurveFitResult


if TYPE_CHECKING:
    from .. import YieldCurve


import numpy as np

if TYPE_CHECKING:
    from ..._finance_types import NativeYieldCurve

from dataclasses import dataclass, field

from ..._finance import positive, scalar
from .._backends import quantlib as _quantlib
from .._backends import scipy as _scipy


@dataclass(frozen=True)
class YieldCurve:
    """Own a QuantLib term structure. Extrapolation is opt-in at construction."""

    structure: NativeYieldCurve
    _constant_rate: float | None = field(
        default=None, init=False, repr=False, compare=False
    )

    calibration_errors: FloatArray | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        return _quantlib.validate_curve(self)

    @property
    def reference_date(self):
        return _quantlib.reference_date(self)

    @property
    def handle(self):
        return _quantlib.handle(self)

    def time(self, maturity: float | DateLike | NativeDate):
        return _quantlib.time(self, maturity)

    def discount(self, maturity: float | DateLike | NativeDate):
        return _quantlib.discount(self, maturity)

    def zero_rate(self, maturity: float | DateLike | NativeDate):
        return _quantlib.zero_rate(self, maturity)

    def forward_rate(self, start, end):
        return _quantlib.forward_rate(self, start, end)

    def shifted(self, spread):
        return _quantlib.shifted(self, spread)

    def rolled(self, reference_date):
        """Conditional discount curve D(new,T)/D(old,new), fixed original curve."""
        return _quantlib.rolled(self, reference_date)


def flat_curve(
    reference_date: DateLike | NativeDate, rate: float, *, convention: str = "actual365"
) -> YieldCurve:
    """Create a discount curve from a calendar reference date and continuous annual rate.

    Times are year fractions under convention; discount(0)=1. Negative rates are allowed."""
    return _quantlib.flat_curve(reference_date, rate, convention=convention)


def discount_curve(
    dates: Sequence[DateLike | NativeDate],
    discounts: ArrayLike,
    *,
    convention: str = "actual365",
    extrapolate: bool = False,
) -> YieldCurve:
    """Interpolate discount factors at increasing dates.

    Discount factors are positive dimensionless present-value multipliers.
    Queries beyond the final node require extrapolate=True."""
    return _quantlib.discount_curve(
        dates, discounts, convention=convention, extrapolate=extrapolate
    )


def zero_curve(
    dates: Sequence[DateLike | NativeDate],
    rates: ArrayLike,
    *,
    convention: str = "actual365",
    extrapolate: bool = False,
) -> YieldCurve:
    """Interpolate continuous annual zero rates at increasing dates.

    Rates are decimal units and times follow convention; extrapolation is explicit."""
    return _quantlib.zero_curve(
        dates, rates, convention=convention, extrapolate=extrapolate
    )


def bootstrap(
    reference_date: DateLike | NativeDate,
    helpers: Sequence[NativeRateHelper],
    *,
    convention: str = "actual365",
    interpolation: str = "log_discount",
    extrapolate: bool = False,
) -> YieldCurve:
    """Native rate helpers retain deposit/swap/OIS/futures/bond conventions."""
    return _quantlib.bootstrap(
        reference_date,
        helpers,
        convention=convention,
        interpolation=interpolation,
        extrapolate=extrapolate,
    )


def forward_price(
    spot: float,
    maturity: float | DateLike | NativeDate,
    financing: YieldCurve,
    *,
    dividend: YieldCurve | None = None,
    borrowing: YieldCurve | None = None,
    cash_dividends: Iterable[CashFlow] = (),
    storage_rate: float = 0.0,
    convenience_yield: float = 0.0,
) -> float:
    """Deterministic carry: (S-PV(dividends))*D_div/(D_fin*D_borrow)*exp((u-y)T).

    Borrowing is an additional positive carrying cost. Continuous yield and cash
    dividends are mutually exclusive to prevent double counting.
    """
    spot = positive(spot, "spot")
    t = scalar(financing.time(maturity), "maturity", minimum=0)
    cash_dividends = tuple(cash_dividends)
    if dividend is not None and cash_dividends:
        raise ValueError("choose continuous yield or discrete dividends")
    for curve in (dividend, borrowing):
        if curve is not None and curve.reference_date != financing.reference_date:
            raise ValueError("carry curves must share a reference date")
    prepaid = spot
    for cf in cash_dividends:
        if financing.reference_date < cf.date and financing.time(cf.date) <= t:
            prepaid -= cf.amount * financing.discount(cf.date)
    if prepaid <= 0:
        raise ValueError("prepaid forward must be positive")
    return (
        prepaid
        * (1 if dividend is None else dividend.discount(maturity))
        / financing.discount(maturity)
        / (1 if borrowing is None else borrowing.discount(maturity))
        * np.exp(
            (scalar(storage_rate, "storage") - scalar(convenience_yield, "convenience"))
            * t
        )
    )


def futures_basis(futures: float, forward: float) -> float:
    """Return futures minus forward in their shared price units; no funding adjustment is inferred."""
    return scalar(futures, "futures") - scalar(forward, "forward")


def fit_discount_curve(
    reference_date: DateLike | NativeDate,
    cashflow_times: ArrayLike,
    cashflows: ArrayLike,
    prices: ArrayLike,
    initial: ArrayLike,
    *,
    model: Callable[[FloatArray, FloatArray], ArrayLike],
) -> CurveFitResult:
    """Fit model(times, parameters); times are years from reference_date.

    Return CurveFitResult with owned parameters and local fit diagnostics.
    Failed optimization raises RuntimeError; discounts must be positive.
    """
    return _scipy.fit_discount_curve(
        reference_date, cashflow_times, cashflows, prices, initial, model=model
    )


def deposit_quote(
    rate: float,
    tenor: str,
    *,
    settlement_days: int = 0,
    calendar: str = "none",
    adjustment: str = "unadjusted",
    convention: str = "actual365",
) -> NativeRateHelper:
    """Deposit quote for bootstrap; rate is decimal, tenor e.g. '3M'."""
    return _quantlib.deposit_quote(
        rate,
        tenor,
        settlement_days=settlement_days,
        calendar=calendar,
        adjustment=adjustment,
        convention=convention,
    )


def swap_quote(
    rate: float,
    tenor: str,
    *,
    settlement_days: int = 0,
    calendar: str = "none",
    adjustment: str = "unadjusted",
    fixed_frequency: str = "annual",
    fixed_convention: str = "30_360",
    floating_tenor: str = "6M",
    floating_convention: str = "actual360",
    currency: str = "USD",
    discount: YieldCurve | None = None,
) -> NativeRateHelper:
    """Par fixed/IBOR swap quote; optional external discount curve for dual-curve fitting."""
    return _quantlib.swap_quote(
        rate,
        tenor,
        settlement_days=settlement_days,
        calendar=calendar,
        adjustment=adjustment,
        fixed_frequency=fixed_frequency,
        fixed_convention=fixed_convention,
        floating_tenor=floating_tenor,
        floating_convention=floating_convention,
        currency=currency,
        discount=discount,
    )


def ois_quote(
    rate: float,
    tenor: str,
    *,
    settlement_days: int = 0,
    calendar: str = "weekends",
    currency: str = "USD",
    convention: str = "actual360",
    payment_frequency: str = "annual",
    payment_adjustment: str = "following",
    discount: YieldCurve | None = None,
) -> NativeRateHelper:
    """Par overnight-indexed swap quote, with compounded overnight coupons."""
    return _quantlib.ois_quote(
        rate,
        tenor,
        settlement_days=settlement_days,
        calendar=calendar,
        currency=currency,
        convention=convention,
        payment_frequency=payment_frequency,
        payment_adjustment=payment_adjustment,
        discount=discount,
    )
