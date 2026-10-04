"""QuantLib contract construction and model composition; no public vendor syntax."""

import numpy as np
import QuantLib as ql
from ..._finance import (
    scalar,
    positive,
    integer,
    choice,
    array,
    qdate,
    day_count,
    evaluation_date,
)
from ..._validation import boolean
from ..._backends.quantlib import (
    calendar as cal,
    adjustment as adj,
    period,
    currency as ccy,
    attach_fixings,
    set_fixings,
    fixing_context,
)
from ..interfaces.swaps import SwapResult


def rate_index(
    curve,
    *,
    tenor="6M",
    currency="USD",
    fixing_days=0,
    calendar="none",
    adjustment="unadjusted",
    convention="actual360",
    fixings=None,
):
    index = ql.IborIndex(
        "Convexity-" + currency,
        period(tenor),
        integer(fixing_days, "fixing days"),
        ccy(currency),
        cal(calendar),
        adj(adjustment),
        False,
        day_count(convention),
        curve.handle,
    )
    return set_fixings(index, fixings)


def cap_floor(schedule, index, strike, *, kind="cap", notional=100.0, spread=0.0):
    choice(kind, ("cap", "floor"), "kind")
    leg = ql.IborLeg(
        [positive(notional, "notional")],
        schedule,
        index,
        spreads=[scalar(spread, "spread")],
    )
    result = (ql.Cap if kind == "cap" else ql.Floor)(leg, [scalar(strike, "strike")])
    return attach_fixings(result, index)


def interest_rate_swap(
    fixed_schedule,
    floating_schedule,
    index,
    fixed_rate,
    *,
    notional=100.0,
    payer=True,
    spread=0.0,
    fixed_convention="30_360",
    floating_convention="actual360",
):
    result = ql.VanillaSwap(
        ql.VanillaSwap.Payer if boolean(payer, "payer") else ql.VanillaSwap.Receiver,
        positive(notional, "notional"),
        fixed_schedule,
        scalar(fixed_rate, "fixed rate"),
        day_count(fixed_convention),
        floating_schedule,
        index,
        scalar(spread, "spread"),
        day_count(floating_convention),
    )
    return attach_fixings(result, index)


def swaption(swap, exercise_dates, *, style="european"):
    from ...contracts import Exercise

    choice(style, ("european", "bermudan"), "swaption exercise")
    exercise = Exercise(style, tuple(exercise_dates))
    if exercise.dates[0] > swap.startDate():
        raise ValueError("first swaption exercise cannot follow swap start")
    # Bermudan native engine handles cancellation of coupons before later exercise.
    if exercise.dates[-1] >= swap.maturityDate():
        raise ValueError("swaption exercise must precede swap maturity")
    result = ql.Swaption(swap, exercise.to_quantlib())
    return attach_fixings(result, *getattr(swap, "_convexity_indices", ()))


def swap_price(swap, curve):
    with evaluation_date(curve.reference_date), fixing_context(swap):
        if swap.maturityDate() <= curve.reference_date:
            raise ValueError("swap must mature after valuation")
        swap.setPricingEngine(ql.DiscountingSwapEngine(curve.handle))
        return SwapResult(
            swap.NPV(),
            swap.fairRate(),
            swap.fixedLegNPV(),
            swap.floatingLegNPV(),
            getattr(swap, "_convexity_currency", "USD"),
        )


def inflation_swap(
    start,
    end,
    index,
    fixed_rate,
    *,
    notional=100.0,
    payer=True,
    calendar="none",
    adjustment="unadjusted",
    convention="actual365",
    observation_lag_months=3,
    interpolation="flat",
):
    start, end = qdate(start), qdate(end)
    if start >= end:
        raise ValueError("swap start must precede maturity")
    methods = {"flat": ql.CPI.Flat, "linear": ql.CPI.Linear}
    interp = methods[choice(interpolation, tuple(methods), "CPI interpolation")]
    result = ql.ZeroCouponInflationSwap(
        ql.Swap.Payer if boolean(payer, "payer") else ql.Swap.Receiver,
        positive(notional, "notional"),
        start,
        end,
        cal(calendar),
        adj(adjustment),
        day_count(convention),
        scalar(fixed_rate, "fixed rate"),
        index,
        ql.Period(integer(observation_lag_months, "observation lag"), ql.Months),
        interp,
    )
    return attach_fixings(result, index)


def forward_market_model(
    rate_times,
    forwards,
    *,
    volatility=0.2,
    factors=None,
    displacement=0.0,
    long_run_correlation=0.5,
    correlation_decay=0.2,
):
    t, f = array(rate_times, "rate times", ndim=1), array(forwards, "forwards", ndim=1)
    n = len(f)
    if len(t) != n + 1 or t[0] <= 0 or np.any(np.diff(t) <= 0):
        raise ValueError("N forwards require N+1 positive increasing rate times")
    shift = np.broadcast_to(array(displacement, "displacement"), f.shape)
    if np.any(f + shift <= 0):
        raise ValueError("shifted forwards must be positive")
    vol = np.broadcast_to(array(volatility, "volatility"), f.shape)
    if np.any(vol < 0):
        raise ValueError("volatility must be nonnegative")
    factors = n if factors is None else integer(factors, "factors", 1)
    if factors > n:
        raise ValueError("factor count cannot exceed forward count")
    corr = scalar(long_run_correlation, "long-run correlation", minimum=0)
    if corr > 1:
        raise ValueError("long-run correlation must be <=1")
    correlation = ql.ExponentialForwardCorrelation(
        t.tolist(), corr, scalar(correlation_decay, "correlation decay", minimum=0)
    )
    return ql.AbcdVol(
        0.0,
        0.0,
        1.0,
        1.0,
        vol.tolist(),
        correlation,
        ql.EvolutionDescription(t.tolist()),
        factors,
        f.tolist(),
        shift.tolist(),
    )


def gaussian_bond_price(model, maturity, *, time=0.0, state=0.0):
    t = scalar(time, "time", minimum=0)
    end = scalar(maturity, "maturity", minimum=0)
    if end < t:
        raise ValueError("maturity must be on or after time")
    return model.zerobond(end, t, scalar(state, "state"))
