"""QuantLib bonds and cash-flow analytics, preserving native conventions."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from numpy.typing import ArrayLike

from ..._typing import DateLike, FloatArray

if TYPE_CHECKING:
    from ..._finance_types import (
        NativeBond,
        NativeCalendar,
        NativeDate,
        NativeDayCounter,
        NativePricingEngine,
        NativeSchedule,
    )
    from ...contracts import CashFlow
    from ...curves import YieldCurve


if TYPE_CHECKING:
    from .. import CarryResult


if TYPE_CHECKING:
    from .. import BondResult


if TYPE_CHECKING:
    from ..._finance_types import NativeInflationIndex


if TYPE_CHECKING:
    from ..._finance_types import NativeIborIndex


from .._backends import quantlib as _quantlib


def zero_coupon(
    maturity: DateLike | NativeDate,
    *,
    face: float = 100.0,
    settlement_days: int = 0,
    calendar: str | NativeCalendar | None = None,
    issue_date: DateLike | NativeDate | None = None,
) -> NativeBond:
    """Construct a native zero-coupon bond with calendar maturity.

    Face is redemption cash; settlement_days counts calendar-adjusted settlement lag."""
    return _quantlib.zero_coupon(
        maturity,
        face=face,
        settlement_days=settlement_days,
        calendar=calendar,
        issue_date=issue_date,
    )


def fixed_bond(
    schedule: NativeSchedule,
    coupon: float,
    *,
    face: float = 100.0,
    settlement_days: int = 0,
    convention: str | NativeDayCounter = "actual365",
    redemption: float = 100.0,
) -> NativeBond:
    """Construct a native fixed-coupon bond from a schedule and annual decimal coupon.

    Face and redemption set cash amounts; convention controls coupon accrual."""
    return _quantlib.fixed_bond(
        schedule,
        coupon,
        face=face,
        settlement_days=settlement_days,
        convention=convention,
        redemption=redemption,
    )


def floating_bond(
    schedule: NativeSchedule,
    index: NativeIborIndex,
    *,
    face: float = 100.0,
    spread: float = 0.0,
    settlement_days: int = 0,
    convention: str | NativeDayCounter = "actual360",
    fixing_days: int | None = None,
) -> NativeBond:
    """Index carries forecast curve and historical fixings; missing fixings fail."""
    return _quantlib.floating_bond(
        schedule,
        index,
        face=face,
        spread=spread,
        settlement_days=settlement_days,
        convention=convention,
        fixing_days=fixing_days,
    )


def amortizing_bond(
    schedule: NativeSchedule,
    notionals: ArrayLike,
    coupon: float,
    *,
    settlement_days: int = 0,
    convention: str | NativeDayCounter = "actual365",
) -> NativeBond:
    """Construct a native amortizing bond with notionals aligned to schedule periods.

    Coupon is an annual decimal rate; principal changes generate repayment cashflows."""
    return _quantlib.amortizing_bond(
        schedule,
        notionals,
        coupon,
        settlement_days=settlement_days,
        convention=convention,
    )


def inflation_bond(
    schedule: NativeSchedule,
    index: NativeInflationIndex,
    base_cpi: float,
    coupon: float,
    *,
    observation_lag_months: int = 3,
    face: float = 100.0,
    settlement_days: int = 0,
    convention: str | NativeDayCounter = "actual365",
    growth_only: bool = False,
    interpolation: str | int | None = None,
) -> NativeBond:
    """Construct a native inflation-linked bond with explicit CPI index and base CPI.

    Coupon is a real annual rate. Observation lag, interpolation and growth-only
    redemption specify indexation; future CPI requires an explicit inflation curve."""
    return _quantlib.inflation_bond(
        schedule,
        index,
        base_cpi,
        coupon,
        observation_lag_months=observation_lag_months,
        face=face,
        settlement_days=settlement_days,
        convention=convention,
        growth_only=growth_only,
        interpolation=interpolation,
    )


def callable_bond(
    schedule: NativeSchedule,
    coupon: float,
    callabilities: Sequence[tuple[DateLike | NativeDate, float, str]],
    *,
    face: float = 100.0,
    settlement_days: int = 0,
    convention: str | NativeDayCounter = "actual365",
) -> NativeBond:
    """callabilities: (date, clean price per 100, 'call' or 'put')."""
    return _quantlib.callable_bond(
        schedule,
        coupon,
        callabilities,
        face=face,
        settlement_days=settlement_days,
        convention=convention,
    )


def cashflows(
    bond: NativeBond, valuation_date: DateLike | NativeDate, *, currency: str = "USD"
) -> tuple[CashFlow, ...]:
    """Extract dated owned CashFlow records from a native bond at a valuation date.

    The evaluation date is scoped to the call; amounts use the supplied currency label."""
    return _quantlib.cashflows(bond, valuation_date, currency=currency)


def yield_from_price(
    bond: NativeBond,
    clean_price: float,
    settlement_date: DateLike | NativeDate,
    *,
    convention: str | NativeDayCounter = "actual365",
    frequency: str | int | None = None,
    compounding: str | int | None = None,
    valuation_date: DateLike | NativeDate | None = None,
) -> float:
    """Invert a bond clean price per 100 face into annual decimal yield.

    Settlement date, day count, compounding and frequency determine the yield convention."""
    return _quantlib.yield_from_price(
        bond,
        clean_price,
        settlement_date,
        convention=convention,
        frequency=frequency,
        compounding=compounding,
        valuation_date=valuation_date,
    )


def price_from_yield(
    bond: NativeBond,
    yield_rate: float,
    settlement_date: DateLike | NativeDate,
    *,
    convention: str | NativeDayCounter = "actual365",
    frequency: str | int | None = None,
    compounding: str | int | None = None,
    valuation_date: DateLike | NativeDate | None = None,
) -> float:
    """Return clean bond price per 100 face for an annual decimal yield.

    Settlement date, day count, compounding and frequency must match yield_from_price."""
    return _quantlib.price_from_yield(
        bond,
        yield_rate,
        settlement_date,
        convention=convention,
        frequency=frequency,
        compounding=compounding,
        valuation_date=valuation_date,
    )


def bond_analytics(
    bond: NativeBond,
    curve: YieldCurve,
    *,
    settlement_date: DateLike | NativeDate | None = None,
    convention: str | NativeDayCounter = "actual365",
    frequency: str | int | None = None,
    engine: NativePricingEngine | None = None,
    compute_yield: bool = True,
) -> BondResult:
    """Yield analytics describe promised cash flows, not option-adjusted risk.

    With a callable engine, DV01/duration from this function remain yield-based;
    use curve_sensitivities for effective callable-bond sensitivities.
    """
    return _quantlib.bond_analytics(
        bond,
        curve,
        settlement_date=settlement_date,
        convention=convention,
        frequency=frequency,
        engine=engine,
        compute_yield=compute_yield,
    )


def key_rate_duration(
    bond: NativeBond,
    curve: YieldCurve,
    key_dates: Sequence[DateLike | NativeDate],
    *,
    bump: float = 0.0001,
    engine_factory: Callable[[YieldCurve], NativePricingEngine] | None = None,
) -> FloatArray:
    """Triangular linearly interpolated zero-spread bumps; flat endpoint wings."""
    return _quantlib.key_rate_duration(
        bond, curve, key_dates, bump=bump, engine_factory=engine_factory
    )


def carry_roll_down(
    bond: NativeBond,
    curve: YieldCurve,
    horizon: DateLike | NativeDate,
    *,
    evolution: str = "constant_tenor",
) -> CarryResult:
    """Cash-unit decomposition with coupons reinvested at implied forwards.

    carry = accrued financing of initial NPV. roll_down = terminal wealth minus
    initial NPV minus carry. Uses valuation-date cash units, not clean quotes.
    constant_tenor holds today's zero curve fixed as a function of remaining
    tenor; implied_forward rolls today's dated curve and gives zero excess
    return by construction. Neither convention is a forecast of future rates.
    """
    return _quantlib.carry_roll_down(bond, curve, horizon, evolution=evolution)


def callable_bond_engine(
    curve: YieldCurve,
    *,
    mean_reversion: float = 0.1,
    volatility: float = 0.01,
    steps: int = 300,
) -> NativePricingEngine:
    """Hull–White exercise tree, usable directly or as a curve-risk factory."""
    return _quantlib.callable_bond_engine(
        curve, mean_reversion=mean_reversion, volatility=volatility, steps=steps
    )
