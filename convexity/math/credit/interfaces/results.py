"""Credit valuation and estimation results with explicit units and measure."""

from __future__ import annotations

from dataclasses import dataclass

from ..._finance import ResultRecord, integer, positive, scalar, unit_interval


@dataclass(frozen=True)
class CDSResult(ResultRecord):
    """Credit-default-swap valuation under explicit premium, default and recovery assumptions."""

    value: float
    fair_spread: float
    premium_leg: float
    protection_leg: float
    spread_sensitivity: float
    accrual_rebate: float = 0.0

    def __post_init__(self) -> None:
        for name in self.keys():
            scalar(getattr(self, name), name)
        if abs(self.value - self.premium_leg - self.protection_leg) > 1e-8 * max(
            1.0, abs(self.value)
        ):
            raise ValueError("CDS value must equal net premium plus protection legs")


@dataclass(frozen=True)
class HazardEstimate(ResultRecord):
    """Estimated default intensity/survival information from the supplied observations."""

    hazard: float
    events: int
    exposure: float
    standard_error: float | None
    measure: str = "P"

    def __post_init__(self) -> None:
        scalar(self.hazard, "hazard", minimum=0)
        integer(self.events, "events")
        positive(self.exposure, "exposure")
        if self.standard_error is not None:
            scalar(self.standard_error, "standard error", minimum=0)
        if self.measure != "P":
            raise ValueError("statistical hazard estimates are P")


@dataclass(frozen=True)
class StructuralCreditResult(ResultRecord):
    """Structural default-model values and probability quantities under the declared model assumptions."""

    equity: float
    debt: float
    q_default_probability: float
    credit_spread: float
    p_default_probability: float | None = None

    def __post_init__(self) -> None:
        scalar(self.equity, "equity", minimum=0)
        positive(self.debt, "debt")
        unit_interval(self.q_default_probability, "Q probability")
        scalar(self.credit_spread, "spread")
        if self.p_default_probability is not None:
            unit_interval(self.p_default_probability, "P probability")
