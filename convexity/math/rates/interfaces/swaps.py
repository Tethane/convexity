"""Currency-unit swap valuation with decimal fair coupon."""

from __future__ import annotations

from dataclasses import dataclass

from ..._finance import scalar


@dataclass(frozen=True)
class SwapResult:
    """Swap valuation with leg/rate information under the specified curve and schedule conventions."""

    value: float
    fair_rate: float
    fixed_leg: float
    floating_leg: float
    currency: str = "USD"

    def __post_init__(self) -> None:
        for name in ("value", "fair_rate", "fixed_leg", "floating_leg"):
            scalar(getattr(self, name), name)
        if abs(self.value - self.fixed_leg - self.floating_leg) > 1e-9 * max(
            1.0, abs(self.value)
        ):
            raise ValueError("swap value must equal its signed leg values")
