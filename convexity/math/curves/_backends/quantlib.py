"""Private quantlib execution for curves."""

from numbers import Real
import numpy as np
import QuantLib as ql
from ..._finance import qdate, day_count, scalar, choice, evaluation_date, array


def validate_curve(self):
    if not isinstance(self.structure, ql.YieldTermStructure):
        raise TypeError("structure must be a QuantLib YieldTermStructure")


def reference_date(self):
    return self.structure.referenceDate()


def handle(self):
    return ql.YieldTermStructureHandle(self.structure)


def time(self, maturity):
    value = (
        maturity
        if isinstance(maturity, Real)
        else self.structure.timeFromReference(qdate(maturity))
    )
    return scalar(value, "time", minimum=0)


def discount(self, maturity):
    t = self.time(maturity)
    if t < 0:
        raise ValueError("maturity precedes curve reference")
    return float(self.structure.discount(t))


def zero_rate(self, maturity):
    return float(self.structure.zeroRate(self.time(maturity), ql.Continuous).rate())


def forward_rate(self, start, end):
    a, b = (self.time(start), self.time(end))
    if a < 0 or b <= a:
        raise ValueError("forward interval must satisfy 0 <= start < end")
    return float(self.structure.forwardRate(a, b, ql.Continuous).rate())


def shifted(self, spread):
    from ..builtins.curves import YieldCurve

    s = ql.QuoteHandle(ql.SimpleQuote(scalar(spread, "spread")))
    result = YieldCurve(ql.ZeroSpreadedTermStructure(self.handle, s))
    if self._constant_rate is not None:
        object.__setattr__(result, "_constant_rate", self._constant_rate + spread)
    return result


def rolled(self, reference_date):
    """Conditional discount curve D(new,T)/D(old,new), fixed original curve."""
    from ..builtins.curves import YieldCurve

    d = qdate(reference_date)
    if d < self.reference_date:
        raise ValueError("roll date precedes reference")
    result = YieldCurve(ql.ImpliedTermStructure(self.handle, d))
    object.__setattr__(result, "_constant_rate", self._constant_rate)
    return result


def flat_curve(reference_date, rate, *, convention="actual365"):
    from ..builtins.curves import YieldCurve

    result = YieldCurve(
        ql.FlatForward(
            qdate(reference_date), scalar(rate, "rate"), day_count(convention)
        )
    )
    object.__setattr__(result, "_constant_rate", scalar(rate, "rate"))
    return result


def discount_curve(dates, discounts, *, convention="actual365", extrapolate=False):
    from ..builtins.curves import YieldCurve

    ds, values = ([qdate(d) for d in dates], array(discounts, "discounts", ndim=1))
    if (
        len(ds) != len(values)
        or len(ds) < 2
        or any((b <= a for a, b in zip(ds, ds[1:])))
    ):
        raise ValueError("curve needs matching, strictly increasing dates")
    if values[0] != 1 or np.any(values <= 0):
        raise ValueError("discount factors must be positive and start at one")
    curve = ql.DiscountCurve(ds, values.tolist(), day_count(convention))
    curve.enableExtrapolation(extrapolate)
    return YieldCurve(curve)


def zero_curve(dates, rates, *, convention="actual365", extrapolate=False):
    from ..builtins.curves import YieldCurve

    ds, values = ([qdate(d) for d in dates], array(rates, "rates", ndim=1))
    if (
        len(ds) != len(values)
        or len(ds) < 2
        or any((b <= a for a, b in zip(ds, ds[1:])))
    ):
        raise ValueError("curve needs matching, strictly increasing dates")
    curve = ql.ZeroCurve(ds, values.tolist(), day_count(convention))
    curve.enableExtrapolation(extrapolate)
    return YieldCurve(curve)


def bootstrap(
    reference_date,
    helpers,
    *,
    convention="actual365",
    interpolation="log_discount",
    extrapolate=False,
):
    """Native rate helpers retain deposit/swap/OIS/futures/bond conventions."""
    from ..builtins.curves import YieldCurve

    constructors = {
        "log_discount": ql.PiecewiseLogLinearDiscount,
        "linear_zero": ql.PiecewiseLinearZero,
        "flat_forward": ql.PiecewiseFlatForward,
    }
    constructor = constructors[
        choice(interpolation, tuple(constructors), "interpolation")
    ]
    helpers = list(helpers)
    if not helpers:
        raise ValueError("at least one bootstrap helper required")
    with evaluation_date(reference_date):
        curve = constructor(qdate(reference_date), helpers, day_count(convention))
        curve.enableExtrapolation(extrapolate)
        curve.discount(curve.maxDate())
        errors = array(
            [h.impliedQuote() - h.quote().value() for h in helpers],
            "calibration errors",
        )
        curve.freeze()
    result = YieldCurve(curve)
    object.__setattr__(result, "calibration_errors", errors)
    return result


def deposit_quote(
    rate,
    tenor,
    *,
    settlement_days=0,
    calendar="none",
    adjustment="unadjusted",
    convention="actual365",
):
    from ..._backends.quantlib import calendar as cal, adjustment as adj, period
    from ..._finance import integer

    return ql.DepositRateHelper(
        scalar(rate, "rate"),
        period(tenor),
        integer(settlement_days, "settlement days"),
        cal(calendar),
        adj(adjustment),
        False,
        day_count(convention),
    )


def swap_quote(
    rate,
    tenor,
    *,
    settlement_days=0,
    calendar="none",
    adjustment="unadjusted",
    fixed_frequency="annual",
    fixed_convention="30_360",
    floating_tenor="6M",
    floating_convention="actual360",
    currency="USD",
    discount=None,
):
    from ..._backends.quantlib import (
        calendar as cal,
        adjustment as adj,
        period,
        frequency,
        currency as ccy,
    )
    from ..._finance import integer

    days = integer(settlement_days, "settlement days")
    index = ql.IborIndex(
        "ConvexityBootstrap",
        period(floating_tenor),
        days,
        ccy(currency),
        cal(calendar),
        adj(adjustment),
        False,
        day_count(floating_convention),
    )
    return ql.SwapRateHelper(
        scalar(rate, "rate"),
        period(tenor),
        cal(calendar),
        frequency(fixed_frequency),
        adj(adjustment),
        day_count(fixed_convention),
        index,
        ql.QuoteHandle(),
        ql.Period(0, ql.Days),
        ql.YieldTermStructureHandle() if discount is None else discount.handle,
        days,
    )


def ois_quote(
    rate,
    tenor,
    *,
    settlement_days=0,
    calendar="weekends",
    currency="USD",
    convention="actual360",
    payment_frequency="annual",
    payment_adjustment="following",
    discount=None,
):
    from ..._backends.quantlib import (
        calendar as cal,
        currency as ccy,
        period,
        frequency,
        adjustment,
    )
    from ..._finance import integer

    index = ql.OvernightIndex(
        "ConvexityOIS-" + currency,
        0,
        ccy(currency),
        cal(calendar),
        day_count(convention),
    )
    return ql.OISRateHelper(
        integer(settlement_days, "settlement days"),
        period(tenor),
        scalar(rate, "rate"),
        index,
        ql.YieldTermStructureHandle() if discount is None else discount.handle,
        paymentConvention=adjustment(payment_adjustment),
        paymentFrequency=frequency(payment_frequency),
        paymentCalendar=cal(calendar),
    )
