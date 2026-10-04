"""Private quantlib_models execution for pricing."""

import QuantLib as ql
from ..._finance import positive, scalar


def validate_curve_clocks(self):
    if self.discount.structure.dayCounter() != self.dividend.structure.dayCounter():
        raise ValueError("discount and dividend day-count conventions must agree")


def validate_bsm(self):
    object.__setattr__(self, "spot", positive(self.spot, "spot"))
    if self.discount.reference_date != self.dividend.reference_date:
        raise ValueError("discount and dividend curves must share a reference date")
    validate_curve_clocks(self)
    if not isinstance(self.volatility, ql.BlackVolTermStructure):
        object.__setattr__(
            self, "volatility", scalar(self.volatility, "volatility", minimum=0)
        )
    elif self.volatility.referenceDate() != self.discount.reference_date:
        raise ValueError("volatility and discount reference dates must agree")
    elif self.volatility.dayCounter() != self.discount.structure.dayCounter():
        raise ValueError("volatility and discount day-count conventions must agree")


def bsm_process(self):
    vol = self.volatility
    if not isinstance(vol, ql.BlackVolTermStructure):
        vol = ql.BlackConstantVol(
            self.discount.reference_date,
            ql.NullCalendar(),
            float(vol),
            self.discount.structure.dayCounter(),
        )
    return ql.BlackScholesMertonProcess(
        ql.QuoteHandle(ql.SimpleQuote(self.spot)),
        self.dividend.handle,
        self.discount.handle,
        ql.BlackVolTermStructureHandle(vol),
    )


def heston_model(self):
    return ql.HestonModel(
        ql.HestonProcess(
            self.discount.handle,
            self.dividend.handle,
            ql.QuoteHandle(ql.SimpleQuote(self.spot)),
            self.variance,
            self.kappa,
            self.theta,
            self.sigma,
            self.rho,
        )
    )


def constant_volatility(model):
    return not isinstance(model.volatility, ql.BlackVolTermStructure) or isinstance(
        model.volatility, ql.BlackConstantVol
    )


def zero_volatility(model):
    vol = model.volatility
    if isinstance(vol, ql.BlackConstantVol):
        return vol.blackVol(0.0, model.spot) == 0.0
    return not isinstance(vol, ql.BlackVolTermStructure) and vol == 0.0


def require_constant_coefficients(model, engine):
    if (
        model.discount._constant_rate is None
        or model.dividend._constant_rate is None
        or (not constant_volatility(model))
    ):
        raise ValueError(
            f"{engine} requires flat_curve rates and constant volatility; use finite differences or Monte Carlo for term structures"
        )


def same_numeraire(left, right):
    if (
        left.reference_date != right.reference_date
        or left.structure.dayCounter() != right.structure.dayCounter()
    ):
        return False
    if left.structure is right.structure:
        return True
    return (
        left._constant_rate is not None
        and right._constant_rate is not None
        and (abs(left._constant_rate - right._constant_rate) < 1e-12)
    )
