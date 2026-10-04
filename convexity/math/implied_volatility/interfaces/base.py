"""Annualized implied volatility and total-variance surface contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from numpy.typing import ArrayLike

from ..._finance import array, scalar
from ..._typing import BoolArray


class VarianceSurface(Protocol):
    """Implied-variance surface extension with explicit maturity and strike coordinates."""
    def total_variance(self, time: float, log_moneyness): ...


@dataclass(frozen=True)
class ArbitrageReport:
    """Sampled violations, not a proof of arbitrage freedom outside the grid."""

    butterfly: BoolArray
    calendar: BoolArray
    minimum_variance: float

    def __post_init__(self) -> None:
        for name in ("butterfly", "calendar"):
            object.__setattr__(self, name, array(getattr(self, name), name))
        scalar(self.minimum_variance, "minimum variance")

    @property
    def passed(self):
        return (
            self.minimum_variance > 0
            and (self.butterfly >= -1e-10).all()
            and (self.calendar >= -1e-10).all()
        )

    def __init__(
        self,
        butterfly: ArrayLike,
        calendar: ArrayLike,
        minimum_variance: float,
    ) -> None:
        object.__setattr__(self, "butterfly", butterfly)
        object.__setattr__(self, "calendar", calendar)
        object.__setattr__(self, "minimum_variance", minimum_variance)
        self.__post_init__()
