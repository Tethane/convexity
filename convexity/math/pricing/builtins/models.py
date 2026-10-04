"""QuantLib pricing models with explicit discount/dividend curves."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...curves import YieldCurve

if TYPE_CHECKING:
    from ..._finance_types import NativeVolatility

from dataclasses import dataclass

from ..._finance import positive, scalar
from .._backends import quantlib_models as _quantlib_models


@dataclass(frozen=True)
class BlackScholesModel:
    """Risk-neutral lognormal model inputs for vanilla pricing, with discount/dividend curves and volatility."""

    spot: float
    discount: YieldCurve
    dividend: YieldCurve
    volatility: float | NativeVolatility
    measure = "Q"

    def __post_init__(self) -> None:
        return _quantlib_models.validate_bsm(self)

    def to_quantlib(self):
        return _quantlib_models.bsm_process(self)

    @classmethod
    def from_measure_pair(cls, pair, spot: float, reference_date):
        """Compose an explicitly supplied GBM Q model with QuantLib curves."""
        from ...curves import flat_curve
        from ...measures import DiffusionPair, MoneyMarketNumeraire
        from ...processes import GeometricBrownianMotion

        if not isinstance(pair, DiffusionPair) or not isinstance(
            pair.pricing.model, GeometricBrownianMotion
        ):
            raise TypeError("a GBM DiffusionPair is required")
        if not isinstance(pair.pricing.numeraire, MoneyMarketNumeraire):
            raise TypeError("constant-rate money-market numeraire required")
        q = pair.pricing.model
        r = pair.pricing.numeraire.rate
        return cls(
            spot,
            flat_curve(reference_date, r),
            flat_curve(reference_date, r - q.mu),
            q.sigma,
        )


@dataclass(frozen=True)
class HestonModel:
    """Risk-neutral stochastic-variance pricing specification with explicit variance-process parameters."""

    spot: float
    discount: YieldCurve
    dividend: YieldCurve
    variance: float
    kappa: float
    theta: float
    sigma: float
    rho: float
    measure = "Q"

    def __post_init__(self) -> None:
        for name in ("spot", "variance", "kappa", "theta", "sigma"):
            object.__setattr__(self, name, positive(getattr(self, name), name))
        object.__setattr__(self, "rho", scalar(self.rho, "rho"))
        if abs(self.rho) >= 1:
            raise ValueError("rho must lie strictly between -1 and 1")
        if self.discount.reference_date != self.dividend.reference_date:
            raise ValueError("discount and dividend reference dates must agree")
        _quantlib_models.validate_curve_clocks(self)

    def to_quantlib(self):
        return _quantlib_models.heston_model(self)
