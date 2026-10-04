"""Private quantlib execution for credit."""

from ..interfaces import CDSResult
import numpy as np
import QuantLib as ql
from ..._finance import (
    integer,
    qdate,
    day_count,
    scalar,
    positive,
    unit_interval,
    array,
    choice,
    evaluation_date,
)


def validate_curve(self):
    choice(self.measure, ("P", "Q"), "measure")
    if not isinstance(self.structure, ql.DefaultProbabilityTermStructure):
        raise TypeError("structure must be a QuantLib default probability curve")


def reference_date(self):
    return self.structure.referenceDate()


def handle(self):
    return ql.DefaultProbabilityTermStructureHandle(self.structure)


def survival(self, maturity):
    return self.structure.survivalProbability(self._arg(maturity))


def hazard(self, maturity):
    return self.structure.hazardRate(self._arg(maturity))


def density(self, maturity):
    return self.structure.defaultDensity(self._arg(maturity))


def flat_hazard(reference_date, hazard, *, measure="Q", convention="actual365"):
    from ..builtins.default import HazardCurve

    return HazardCurve(
        ql.FlatHazardRate(
            qdate(reference_date),
            ql.QuoteHandle(ql.SimpleQuote(scalar(hazard, "hazard", minimum=0))),
            day_count(convention),
        ),
        measure,
    )


def survival_curve(
    dates, probabilities, *, measure="Q", convention="actual365", extrapolate=False
):
    from ..builtins.default import HazardCurve

    ds = [qdate(d) for d in dates]
    p = array(probabilities, "survival", ndim=1)
    if (
        len(ds) != len(p)
        or len(ds) < 2
        or any((b <= a for a, b in zip(ds, ds[1:])))
        or (p[0] != 1)
        or np.any(p <= 0)
        or np.any(np.diff(p) > 0)
    ):
        raise ValueError(
            "ordered dates and positive nonincreasing survival starting at one required"
        )
    curve = ql.SurvivalProbabilityCurve(ds, p.tolist(), day_count(convention))
    curve.enableExtrapolation(extrapolate)
    return HazardCurve(curve, measure)


def bootstrap_cds(
    reference_date, helpers, *, convention="actual365", extrapolate=False
):
    from ..builtins.default import HazardCurve

    helpers = list(helpers)
    if not helpers:
        raise ValueError("CDS helpers required")
    with evaluation_date(reference_date):
        curve = ql.PiecewiseFlatHazardRate(
            qdate(reference_date), helpers, day_count(convention)
        )
        curve.survivalProbability(curve.maxDate())
        nodes = list(curve.nodes())
        snapshot = ql.HazardRateCurve(
            [d for d, h in nodes], [h for d, h in nodes], day_count(convention)
        )
        snapshot.enableExtrapolation(extrapolate)
        errors = array(
            [h.impliedQuote() - h.quote().value() for h in helpers],
            "calibration errors",
        )
    result = HazardCurve(snapshot, "Q", curve)
    object.__setattr__(result, "calibration_errors", errors)
    return result


def cds_price(
    schedule,
    spread,
    discount,
    default,
    *,
    notional=1.0,
    recovery=0.4,
    buyer=True,
    protection_start=None,
    trade_date=None,
    cash_settlement_days=3,
):
    if default.measure != "Q":
        raise ValueError("CDS valuation requires Q default probabilities")
    if discount.reference_date != default.reference_date:
        raise ValueError("credit and discount references must agree")
    from ..._validation import boolean

    boolean(buyer, "buyer")
    rec = unit_interval(recovery, "recovery")
    spread = scalar(spread, "spread", minimum=0)
    settlement_days = integer(cash_settlement_days, "cash settlement days")
    if len(schedule) < 2 or schedule[-1] <= discount.reference_date:
        raise ValueError("CDS schedule must end after valuation")
    with evaluation_date(discount.reference_date):

        def instrument(coupon):
            cds = ql.CreditDefaultSwap(
                ql.Protection.Buyer if buyer else ql.Protection.Seller,
                positive(notional, "notional"),
                coupon,
                schedule,
                ql.Unadjusted,
                ql.Actual360(),
                True,
                True,
                ql.Date() if protection_start is None else qdate(protection_start),
                ql.FaceValueClaim(),
                ql.Actual360(),
                True,
                ql.Date() if trade_date is None else qdate(trade_date),
                settlement_days,
            )
            cds.setPricingEngine(
                ql.MidPointCdsEngine(default.handle, rec, discount.handle)
            )
            return cds

        cds = instrument(spread)
        value = cds.NPV()
        rebate = cds.accrualRebateNPV()
        bumped = instrument(spread + 0.0001)
        spread_risk = (bumped.NPV() - value) / 0.0001
        return CDSResult(
            value,
            cds.fairSpread() if spread else bumped.fairSpread(),
            cds.couponLegNPV() + rebate,
            cds.defaultLegNPV(),
            spread_risk,
            rebate,
        )


def cds_quote(
    spread,
    tenor,
    discount,
    *,
    recovery=0.4,
    settlement_days=0,
    calendar="none",
    adjustment="unadjusted",
    frequency="quarterly",
    convention="actual360",
):
    from ..._backends.quantlib import (
        calendar as cal,
        adjustment as adj,
        period,
        frequency as freq,
    )

    with evaluation_date(discount.reference_date):
        return ql.SpreadCdsHelper(
            scalar(spread, "spread", minimum=0),
            period(tenor),
            integer(settlement_days, "settlement days"),
            cal(calendar),
            freq(frequency),
            adj(adjustment),
            ql.DateGeneration.Forward,
            day_count(convention),
            unit_interval(recovery, "recovery"),
            discount.handle,
        )
