"""Rate models explicitly distinguish P dynamics from Q bond valuation."""

from typing import Protocol


class RateModel(Protocol):
    """Interest-rate model extension with explicitly interpreted state and time coordinates."""
    measure: str

    def discount_bond(self, time: float, maturity: float, state): ...
