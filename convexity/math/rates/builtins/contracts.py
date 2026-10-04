"""Financial contracts from ordinary dates, numbers and index conventions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from numpy.typing import ArrayLike

from ..._typing import DateLike

if TYPE_CHECKING:
    from ..._finance_types import (
        NativeCapFloor,
        NativeDate,
        NativeGSR,
        NativeIborIndex,
        NativeInflationIndex,
        NativeInflationSwap,
        NativeMarketModel,
        NativeSchedule,
        NativeSwap,
        NativeSwaption,
    )
    from ...curves import YieldCurve


if TYPE_CHECKING:
    from .. import SwapResult


from .._backends import contracts as _backend


def rate_index[FixingDateT: DateLike | NativeDate](
    curve: YieldCurve,
    *,
    tenor: str = "6M",
    currency: str = "USD",
    fixing_days: int = 0,
    calendar: str = "none",
    adjustment: str = "unadjusted",
    convention: str = "actual360",
    fixings: Mapping[FixingDateT, float] | None = None,
) -> NativeIborIndex:
    """Forecast index with optional {date: rate} history, scoped during valuation."""
    return _backend.rate_index(
        curve,
        tenor=tenor,
        currency=currency,
        fixing_days=fixing_days,
        calendar=calendar,
        adjustment=adjustment,
        convention=convention,
        fixings=fixings,
    )


def cap_floor(
    schedule: NativeSchedule,
    index: NativeIborIndex,
    strike: float,
    *,
    kind: str = "cap",
    notional: float = 100.0,
    spread: float = 0.0,
) -> NativeCapFloor:
    """European cap/floor on the supplied index schedule; rates are decimal."""
    return _backend.cap_floor(
        schedule, index, strike, kind=kind, notional=notional, spread=spread
    )


def interest_rate_swap(
    fixed_schedule: NativeSchedule,
    floating_schedule: NativeSchedule,
    index: NativeIborIndex,
    fixed_rate: float,
    *,
    notional: float = 100.0,
    payer: bool = True,
    spread: float = 0.0,
    fixed_convention: str = "30_360",
    floating_convention: str = "actual360",
) -> NativeSwap:
    """Fixed/IBOR swap; payer=True pays fixed and receives index plus spread.

    Schedules come from contracts.schedule. Rates are decimal; notional and
    swap_price leg values are in the index currency. Forecasting uses the
    index curve; discounting is supplied separately to swap_price.
    """
    return _backend.interest_rate_swap(
        fixed_schedule,
        floating_schedule,
        index,
        fixed_rate,
        notional=notional,
        payer=payer,
        spread=spread,
        fixed_convention=fixed_convention,
        floating_convention=floating_convention,
    )


def swaption(
    swap: NativeSwap,
    exercise_dates: Sequence[DateLike | NativeDate],
    *,
    style: str = "european",
) -> NativeSwaption:
    """Physically settled swaption with European or Bermudan exercise.

    The first exercise is on/before swap start; later Bermudan exercise dates
    must precede maturity. Exercise dates are an ordered sequence of dates.
    """
    return _backend.swaption(swap, exercise_dates, style=style)


def swap_price(swap: NativeSwap, curve: YieldCurve) -> SwapResult:
    """Return SwapResult with signed leg NPVs, decimal fair_rate and currency."""
    return _backend.swap_price(swap, curve)


def inflation_swap(
    start: DateLike | NativeDate,
    end: DateLike | NativeDate,
    index: NativeInflationIndex,
    fixed_rate: float,
    *,
    notional: float = 100.0,
    payer: bool = True,
    calendar: str = "none",
    adjustment: str = "unadjusted",
    convention: str = "actual365",
    observation_lag_months: int = 3,
    interpolation: str = "flat",
) -> NativeInflationSwap:
    """Zero-coupon CPI swap; payer=True pays fixed growth and receives inflation.

    Dates accept ISO strings or Python dates. fixed_rate is an annual decimal
    growth rate, and notional is in index currency. Supply required historical
    CPI observations through inflation_index(..., fixings={date: level}).
    """
    return _backend.inflation_swap(
        start,
        end,
        index,
        fixed_rate,
        notional=notional,
        payer=payer,
        calendar=calendar,
        adjustment=adjustment,
        convention=convention,
        observation_lag_months=observation_lag_months,
        interpolation=interpolation,
    )


def forward_market_model(
    rate_times: ArrayLike,
    forwards: ArrayLike,
    *,
    volatility: float = 0.2,
    factors: int | None = None,
    displacement: float = 0.0,
    long_run_correlation: float = 0.5,
    correlation_decay: float = 0.2,
) -> NativeMarketModel:
    """Lognormal forward-rate model with constant vol and exponential correlation.

    rate_times are N+1 reset/payment times, forwards have N entries. Paths use
    the terminal bond measure. Displacements allow negative unshifted forwards.
    """
    return _backend.forward_market_model(
        rate_times,
        forwards,
        volatility=volatility,
        factors=factors,
        displacement=displacement,
        long_run_correlation=long_run_correlation,
        correlation_decay=correlation_decay,
    )


def gaussian_bond_price(
    model: NativeGSR, maturity: float, *, time: float = 0.0, state: float = 0.0
) -> float:
    """GSR conditional zero-bond price; state is the standardized Gaussian factor."""
    return _backend.gaussian_bond_price(model, maturity, time=time, state=state)
