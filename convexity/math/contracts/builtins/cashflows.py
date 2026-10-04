"""QuantLib schedules, day counts, interest conventions and simple payoffs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, cast

from numpy.typing import ArrayLike

from ..._typing import DateLike, FloatArray

if TYPE_CHECKING:
    from ..._finance_types import (
        NativeCalendar,
        NativeDate,
        NativeDayCounter,
        NativeSchedule,
    )
    from ...contracts import CashFlow
    from ...curves import YieldCurve


if TYPE_CHECKING:
    from .. import CashFlow


import numpy as np

from ..._finance import array, choice, positive, scalar
from .._backends import quantlib as _quantlib


def payoff(
    spot: ArrayLike, strike: float, *, kind: str = "call", cash: float | None = None
) -> float | FloatArray:
    """Evaluate vanilla terminal payoff elementwise in spot units.

    kind selects call/put; supplying cash selects a cash-or-nothing digital payoff.
    The output is undiscounted and does not specify a probability measure."""
    s = array(spot, "spot")
    k = scalar(strike, "strike")
    intrinsic = (s - k) * (1 if choice(kind, ("call", "put"), "kind") == "call" else -1)
    return cast(
        float | FloatArray,
        (
            np.maximum(intrinsic, 0)
            if cash is None
            else scalar(cash, "cash") * (intrinsic > 0)
        ),
    )


def schedule(
    start: DateLike | NativeDate,
    end: DateLike | NativeDate,
    *,
    months: int = 6,
    calendar: str | NativeCalendar | None = None,
    convention: str | int | None = None,
    backwards: bool = True,
    end_of_month: bool = False,
    first_date: DateLike | NativeDate | None = None,
    next_to_last_date: DateLike | NativeDate | None = None,
) -> NativeSchedule:
    """Construct a native QuantLib schedule between calendar dates.

    months is the regular tenor; calendar/convention specify business-day adjustment.
    Stub dates and end-of-month handling are explicit. Use schedule_dates for Python dates."""
    return _quantlib.schedule(
        start,
        end,
        months=months,
        calendar=calendar,
        convention=convention,
        backwards=backwards,
        end_of_month=end_of_month,
        first_date=first_date,
        next_to_last_date=next_to_last_date,
    )


def year_fraction(
    start: DateLike | NativeDate,
    end: DateLike | NativeDate,
    convention: str | NativeDayCounter = "actual365",
) -> float:
    """Compute elapsed year fraction between dates under the selected day-count convention."""
    return _quantlib.year_fraction(start, end, convention)


def interest_factor(
    rate: float, time: float, *, compounding: str = "continuous", frequency: int = 2
) -> float:
    """Return accumulation over time years for a decimal annual rate.

    Compounding is continuous, simple or periodic; frequency is periods per year."""
    return _quantlib.interest_factor(
        rate, time, compounding=compounding, frequency=frequency
    )


def fixed_cashflows(
    dates: NativeSchedule,
    coupon: float,
    *,
    notional: float = 100.0,
    convention: str = "actual365",
    redemption: float = 100.0,
    currency: str = "USD",
) -> tuple[CashFlow, ...]:
    """Construct coupon and redemption CashFlows from ordered schedule dates.

    Coupon is an annual decimal rate; each accrual uses the selected day count.
    Notional and redemption are in the supplied currency."""
    return _quantlib.fixed_cashflows(
        dates,
        coupon,
        notional=notional,
        convention=convention,
        redemption=redemption,
        currency=currency,
    )


def present_value(
    cashflows: Sequence[CashFlow],
    curve: YieldCurve,
    *,
    currency: str = "USD",
    fx: Mapping[str, float] | None = None,
    include_reference: bool = False,
) -> float:
    """FX maps source currency to valuation-currency units per source unit."""
    total = 0.0
    for cf in cashflows:
        if cf.date < curve.reference_date or (
            cf.date == curve.reference_date and (not include_reference)
        ):
            continue
        conversion = (
            1.0 if cf.currency == currency else positive((fx or {})[cf.currency], "FX")
        )
        total += cf.amount * conversion * curve.discount(cf.date)
    return float(total)


def split_holdings(quantity: float, price: float, ratio: float) -> tuple[float, float]:
    """Apply a positive split ratio, multiplying quantity and dividing price.

    Return adjusted holdings and price; total marked value is preserved."""
    ratio = positive(ratio, "split ratio")
    return (
        scalar(quantity, "quantity") * ratio,
        scalar(price, "price", minimum=0) / ratio,
    )


def dividend_adjustment(
    quantity: float, price: float, dividend: float
) -> tuple[float, float, float]:
    """Frictionless ex-dividend transformation: (shares, ex-price, cash)."""
    q, p, d = (
        scalar(quantity, "quantity"),
        scalar(price, "price", minimum=0),
        scalar(dividend, "dividend", minimum=0),
    )
    if d > p:
        raise ValueError("dividend exceeds price")
    return (q, p - d, q * d)
