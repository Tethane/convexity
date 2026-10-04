"""Bond results use prices per 100 current outstanding principal."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..._typing import (
    DateLike,
)

if TYPE_CHECKING:
    from ..._finance_types import NativeDate

from dataclasses import dataclass
from typing import Protocol

from ..._finance import ResultRecord, positive, qdate, scalar


@dataclass(frozen=True)
class BondResult:
    """Bond valuation and dated cash flows with the pricing conventions retained."""

    clean: float
    dirty: float
    accrued: float
    settlement_date: NativeDate
    yield_to_maturity: float | None = None
    duration: float | None = None
    convexity: float | None = None
    dv01: float | None = None

    def __post_init__(self) -> None:
        for name in (
            "clean",
            "dirty",
            "accrued",
            "yield_to_maturity",
            "duration",
            "convexity",
            "dv01",
        ):
            if getattr(self, name) is not None:
                scalar(getattr(self, name), name)
        object.__setattr__(self, "settlement_date", qdate(self.settlement_date))
        if abs(self.dirty - self.clean - self.accrued) > 1e-8:
            raise ValueError("dirty price must equal clean plus accrued")

    def __init__(
        self,
        clean: float,
        dirty: float,
        accrued: float,
        settlement_date: DateLike | NativeDate,
        yield_to_maturity: float | None = None,
        duration: float | None = None,
        convexity: float | None = None,
        dv01: float | None = None,
    ) -> None:
        object.__setattr__(self, "clean", clean)
        object.__setattr__(self, "dirty", dirty)
        object.__setattr__(self, "accrued", accrued)
        object.__setattr__(self, "settlement_date", settlement_date)
        object.__setattr__(self, "yield_to_maturity", yield_to_maturity)
        object.__setattr__(self, "duration", duration)
        object.__setattr__(self, "convexity", convexity)
        object.__setattr__(self, "dv01", dv01)
        self.__post_init__()


class BondPricer(Protocol):
    """Value a supplied bond under explicit date, curve and pricing conventions."""
    def __call__(self, bond, curve) -> BondResult: ...


@dataclass(frozen=True)
class CarryResult(ResultRecord):
    """Carry/roll decomposition under the supplied horizon and curve assumptions."""

    initial_value: float
    horizon_value: float
    reinvested_coupons: float
    carry: float
    roll_down: float
    total_return: float

    def __post_init__(self) -> None:
        for name in self.keys():
            scalar(getattr(self, name), name)
        positive(self.initial_value, "initial value")
        pnl = self.horizon_value + self.reinvested_coupons - self.initial_value
        if abs(pnl - self.carry - self.roll_down) > 1e-9 * max(1.0, abs(pnl)):
            raise ValueError("carry and roll-down must reconcile with terminal wealth")
