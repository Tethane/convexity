"""Private quantlib execution for fixed_income."""

from ..interfaces import CarryResult
import numpy as np
import QuantLib as ql
from ..._finance import (
    scalar,
    positive,
    integer,
    array,
    qdate,
    day_count,
    evaluation_date,
)
from ...contracts import CashFlow
from ..interfaces import BondResult
from ..._backends.quantlib import (
    calendar as calendar_value,
    frequency as frequency_value,
    compounding as compounding_value,
    attach_fixings,
    fixing_context,
)


def zero_coupon(
    maturity, *, face=100.0, settlement_days=0, calendar=None, issue_date=None
):
    return ql.ZeroCouponBond(
        integer(settlement_days, "settlement days"),
        calendar_value(calendar),
        positive(face, "face"),
        qdate(maturity),
        ql.Unadjusted,
        100.0,
        qdate(issue_date) if issue_date else ql.Date(),
    )


def fixed_bond(
    schedule,
    coupon,
    *,
    face=100.0,
    settlement_days=0,
    convention: str | ql.DayCounter = "actual365",
    redemption=100.0,
):
    return ql.FixedRateBond(
        integer(settlement_days, "settlement days"),
        positive(face, "face"),
        schedule,
        [scalar(coupon, "coupon")],
        day_count(convention),
        ql.Unadjusted,
        scalar(redemption, "redemption", minimum=0),
        schedule[0],
    )


def floating_bond(
    schedule,
    index,
    *,
    face=100.0,
    spread=0.0,
    settlement_days=0,
    convention: str | ql.DayCounter = "actual360",
    fixing_days=None,
):
    """Index carries forecast curve and historical fixings; missing fixings fail."""
    bond = ql.FloatingRateBond(
        integer(settlement_days, "settlement days"),
        positive(face, "face"),
        schedule,
        index,
        day_count(convention),
        ql.Unadjusted,
        index.fixingDays()
        if fixing_days is None
        else integer(fixing_days, "fixing days"),
        [1.0],
        [scalar(spread, "spread")],
        [],
        [],
        False,
        100.0,
        schedule[0],
    )
    ql.setCouponPricer(bond.cashflows(), ql.BlackIborCouponPricer())
    return attach_fixings(bond, index)


def amortizing_bond(
    schedule,
    notionals,
    coupon,
    *,
    settlement_days=0,
    convention: str | ql.DayCounter = "actual365",
):
    n = array(notionals, "notionals", ndim=1)
    if len(n) != len(schedule) - 1 or np.any(n <= 0) or np.any(np.diff(n) > 0):
        raise ValueError(
            "positive nonincreasing notionals required for each coupon period"
        )
    return ql.AmortizingFixedRateBond(
        integer(settlement_days, "settlement days"),
        n.tolist(),
        schedule,
        [scalar(coupon, "coupon")],
        day_count(convention),
        ql.Unadjusted,
        schedule[0],
    )


def inflation_bond(
    schedule,
    index,
    base_cpi,
    coupon,
    *,
    observation_lag_months=3,
    face=100.0,
    settlement_days=0,
    convention: str | ql.DayCounter = "actual365",
    growth_only=False,
    interpolation=None,
):
    if interpolation is None:
        interpolation = ql.CPI.Flat
    bond = ql.CPIBond(
        integer(settlement_days, "settlement days"),
        positive(face, "face"),
        growth_only,
        positive(base_cpi, "base CPI"),
        ql.Period(integer(observation_lag_months, "lag"), ql.Months),
        index,
        interpolation,
        schedule,
        [scalar(coupon, "coupon")],
        day_count(convention),
        ql.Unadjusted,
        schedule[0],
    )

    return attach_fixings(bond, index)


def callable_bond(
    schedule,
    coupon,
    callabilities,
    *,
    face=100.0,
    settlement_days=0,
    convention: str | ql.DayCounter = "actual365",
):
    """callabilities: (date, clean price per 100, 'call' or 'put')."""
    calls = []
    for date, price, kind in callabilities:
        if kind not in ("call", "put"):
            raise ValueError("callability kind must be call or put")
        d = qdate(date)
        if d <= schedule[0] or d > schedule[-1]:
            raise ValueError("callability dates must be within bond life")
        calls.append(
            ql.Callability(
                ql.BondPrice(positive(price, "exercise price"), ql.BondPrice.Clean),
                ql.Callability.Call if kind == "call" else ql.Callability.Put,
                d,
            )
        )
    return ql.CallableFixedRateBond(
        integer(settlement_days, "settlement days"),
        positive(face, "face"),
        schedule,
        [scalar(coupon, "coupon")],
        day_count(convention),
        ql.Unadjusted,
        100.0,
        schedule[0],
        calls,
    )


def cashflows(bond, valuation_date, *, currency="USD"):
    with evaluation_date(valuation_date), fixing_context(bond):
        return tuple(
            (CashFlow(cf.date(), cf.amount(), currency) for cf in bond.cashflows())
        )


def yield_from_price(
    bond,
    clean_price,
    settlement_date,
    *,
    convention: str | ql.DayCounter = "actual365",
    frequency=None,
    compounding=None,
    valuation_date=None,
):
    from ..._finance import evaluation_date as context

    compounding = compounding_value(compounding)
    frequency = frequency_value(frequency)
    with (
        context(valuation_date if valuation_date is not None else settlement_date),
        fixing_context(bond),
    ):
        return ql.BondFunctions.bondYield(
            bond,
            ql.BondPrice(positive(clean_price, "clean price"), ql.BondPrice.Clean),
            day_count(convention),
            compounding,
            frequency,
            qdate(settlement_date),
            1e-12,
            1000,
        )


def price_from_yield(
    bond,
    yield_rate,
    settlement_date,
    *,
    convention: str | ql.DayCounter = "actual365",
    frequency=None,
    compounding=None,
    valuation_date=None,
):
    from ..._finance import evaluation_date as context

    compounding = compounding_value(compounding)
    frequency = frequency_value(frequency)
    with (
        context(valuation_date if valuation_date is not None else settlement_date),
        fixing_context(bond),
    ):
        return ql.BondFunctions.cleanPrice(
            bond,
            scalar(yield_rate, "yield"),
            day_count(convention),
            compounding,
            frequency,
            qdate(settlement_date),
        )


def bond_analytics(
    bond,
    curve,
    *,
    settlement_date=None,
    convention: str | ql.DayCounter = "actual365",
    frequency=None,
    engine=None,
    compute_yield=True,
):
    """Yield analytics describe promised cash flows, not option-adjusted risk.

    With a callable engine, DV01/duration from this function remain yield-based;
    use curve_sensitivities for effective callable-bond sensitivities.
    """
    frequency = frequency_value(frequency)
    if isinstance(bond, ql.CallableFixedRateBond) and engine is None:
        raise ValueError("callable/putable bond valuation requires an exercise engine")
    with evaluation_date(curve.reference_date), fixing_context(bond):
        settlement = (
            bond.settlementDate() if settlement_date is None else qdate(settlement_date)
        )
        if settlement < curve.reference_date or settlement >= bond.maturityDate():
            raise ValueError(
                "settlement must be on/after valuation and before maturity"
            )
        if engine is None:
            clean = ql.BondFunctions.cleanPrice(bond, curve.structure, settlement)
            dirty = ql.BondFunctions.dirtyPrice(bond, curve.structure, settlement)
        else:
            if settlement != bond.settlementDate():
                raise ValueError(
                    "engine pricing uses the bond contractual settlement date"
                )
            bond.setPricingEngine(engine)
            clean = bond.cleanPrice()
            dirty = bond.dirtyPrice()
        accrued = bond.accruedAmount(settlement)
        if not compute_yield:
            return BondResult(clean, dirty, accrued, settlement)
        dc = day_count(convention)
        y = yield_from_price(
            bond,
            clean,
            settlement,
            convention=dc,
            frequency=frequency,
            valuation_date=curve.reference_date,
        )
        duration = ql.BondFunctions.duration(
            bond, y, dc, ql.Compounded, frequency, ql.Duration.Modified, settlement
        )
        convexity = ql.BondFunctions.convexity(
            bond, y, dc, ql.Compounded, frequency, settlement
        )
        dv01 = (
            price_from_yield(
                bond,
                y - 0.0001,
                settlement,
                convention=dc,
                frequency=frequency,
                valuation_date=curve.reference_date,
            )
            - price_from_yield(
                bond,
                y + 0.0001,
                settlement,
                convention=dc,
                frequency=frequency,
                valuation_date=curve.reference_date,
            )
        ) / 2
        return BondResult(
            clean, dirty, accrued, settlement, y, duration, convexity, dv01
        )


def key_rate_duration(bond, curve, key_dates, *, bump=0.0001, engine_factory=None):
    """Triangular linearly interpolated zero-spread bumps; flat endpoint wings."""
    from ...curves import YieldCurve

    if isinstance(bond, ql.CallableFixedRateBond) and engine_factory is None:
        raise ValueError("callable/putable curve risk requires an engine_factory")

    def dirty(c):
        if engine_factory is None:
            return ql.BondFunctions.dirtyPrice(bond, c.structure)
        return bond_analytics(
            bond, c, engine=engine_factory(c), compute_yield=False
        ).dirty

    dates = [qdate(d) for d in key_dates]
    h = positive(bump, "bump")
    if (
        len(dates) < 2
        or dates[0] < curve.reference_date
        or any((b <= a for a, b in zip(dates, dates[1:])))
    ):
        raise ValueError("ordered key dates on/after reference required")
    with evaluation_date(curve.reference_date), fixing_context(bond):
        base = dirty(curve)
        out = []
        for i in range(len(dates)):
            prices = []
            for sign in (-1, 1):
                quotes = [
                    ql.QuoteHandle(ql.SimpleQuote(sign * h if j == i else 0.0))
                    for j in range(len(dates))
                ]
                bumped = ql.SpreadedLinearZeroInterpolatedTermStructure(
                    curve.handle, quotes, dates
                )
                bumped.enableExtrapolation()
                prices.append(dirty(YieldCurve(bumped)))
            out.append((prices[0] - prices[1]) / (2 * h * base))
        return array(out, "key rate durations")


def carry_roll_down(bond, curve, horizon, *, evolution="constant_tenor"):
    """Cash-unit decomposition with coupons reinvested at implied forwards.

    carry = accrued financing of initial NPV. roll_down = terminal wealth minus
    initial NPV minus carry. Uses valuation-date cash units, not clean quotes.
    constant_tenor holds today's zero curve fixed as a function of remaining
    tenor; implied_forward rolls today's dated curve and gives zero excess
    return by construction. Neither convention is a forecast of future rates.
    """
    from ..._finance import choice

    if isinstance(bond, ql.CallableFixedRateBond):
        raise ValueError("callable bond carry requires an explicit exercise policy")
    choice(evolution, ("constant_tenor", "implied_forward"), "curve evolution")
    horizon = qdate(horizon)
    if horizon <= curve.reference_date or horizon >= bond.maturityDate():
        raise ValueError("horizon must be after valuation and before maturity")
    with evaluation_date(curve.reference_date), fixing_context(bond):
        flows = [
            (c.date(), c.amount())
            for c in bond.cashflows()
            if c.date() > curve.reference_date
        ]
        initial = sum((a * curve.discount(d) for d, a in flows))
        factor = curve.discount(horizon)
        coupons = sum(
            (a * curve.discount(d) / factor for d, a in flows if d <= horizon)
        )
        if evolution == "implied_forward":
            remaining = sum(
                (a * curve.discount(d) / factor for d, a in flows if d > horizon)
            )
        else:
            dc = curve.structure.dayCounter()
            remaining = sum(
                (
                    a * curve.discount(dc.yearFraction(horizon, d))
                    for d, a in flows
                    if d > horizon
                )
            )
        carry = initial * (1 / factor - 1)
        return CarryResult(
            initial_value=initial,
            horizon_value=remaining,
            reinvested_coupons=coupons,
            carry=carry,
            roll_down=remaining + coupons - initial - carry,
            total_return=(remaining + coupons) / initial - 1,
        )


def callable_bond_engine(curve, *, mean_reversion=0.1, volatility=0.01, steps=300):
    with evaluation_date(curve.reference_date):
        model = ql.HullWhite(
            curve.handle,
            positive(mean_reversion, "mean reversion"),
            positive(volatility, "volatility"),
        )
        return ql.TreeCallableFixedRateBondEngine(model, integer(steps, "steps", 2))
