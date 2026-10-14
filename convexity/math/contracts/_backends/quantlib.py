"""Private quantlib execution for contracts."""

import QuantLib as ql
from ..._finance import qdate, choice, positive, scalar, option_type
from ..._finance import day_count, integer
from ..interfaces import CashFlow
from ..._backends.quantlib import calendar as calendar_value, adjustment
from ..._validation import boolean


def exercise(self):
    if self.style == "european":
        return ql.EuropeanExercise(self.dates[0])
    if self.style == "american":
        return ql.AmericanExercise(*self.dates)
    return ql.BermudanExercise(list(self.dates))


def vanilla_contract(self):
    return ql.VanillaOption(
        ql.PlainVanillaPayoff(option_type(self.kind), self.strike),
        self.exercise.to_quantlib(),
    )


def cashflow(self):
    return ql.SimpleCashFlow(self.amount, self.date)


def schedule(
    start,
    end,
    *,
    months=6,
    calendar=None,
    convention=None,
    backwards=True,
    end_of_month=False,
    first_date=None,
    next_to_last_date=None,
):
    convention = adjustment(convention)
    backwards = boolean(backwards, "backwards")
    end_of_month = boolean(end_of_month, "end_of_month")
    start, end = (qdate(start), qdate(end))
    if start >= end:
        raise ValueError("schedule start must precede end")
    return ql.Schedule(
        start,
        end,
        ql.Period(integer(months, "months", 1), ql.Months),
        calendar_value(calendar),
        convention,
        convention,
        ql.DateGeneration.Backward if backwards else ql.DateGeneration.Forward,
        end_of_month,
        qdate(first_date) if first_date else ql.Date(),
        qdate(next_to_last_date) if next_to_last_date else ql.Date(),
    )


def year_fraction(start, end, convention: str | ql.DayCounter = "actual365"):
    return day_count(convention).yearFraction(qdate(start), qdate(end))


def interest_factor(rate, time, *, compounding="continuous", frequency=2):
    conventions = {
        "continuous": ql.Continuous,
        "simple": ql.Simple,
        "periodic": ql.Compounded,
    }
    comp = conventions[choice(compounding, tuple(conventions), "compounding")]
    freq = integer(frequency, "frequency", 1)
    if freq not in (1, 2, 3, 4, 6, 12):
        raise ValueError("frequency must be 1, 2, 3, 4, 6 or 12")
    return ql.InterestRate(
        scalar(rate, "rate"), ql.Actual365Fixed(), comp, freq
    ).compoundFactor(scalar(time, "time", minimum=0))


def fixed_cashflows(
    dates,
    coupon,
    *,
    notional=100.0,
    convention="actual365",
    redemption=100.0,
    currency="USD",
):
    leg = ql.FixedRateLeg(
        dates,
        day_count(convention),
        [positive(notional, "notional")],
        [scalar(coupon, "coupon")],
        paymentAdjustment=ql.Unadjusted,
    )
    flows = [CashFlow(c.date(), c.amount(), currency) for c in leg]
    flows.append(
        CashFlow(
            dates[-1],
            notional * scalar(redemption, "redemption", minimum=0) / 100,
            currency,
        )
    )
    return tuple(flows)


def date_value(value):
    from datetime import date

    d = qdate(value)
    return date(d.year(), d.month(), d.dayOfMonth())
