"""Concrete specifications and convenience functions for wealth.

Ordinary workflows may import these names directly from convexity.math.wealth.
See docs/reference/wealth.md for public signatures and mathematical conventions.
"""
from .transforms import (
    trade,
    accrue,
    collateral_transfer,
    cash_flow,
    exercise_delivery,
    settle,
    futures_variation,
    financed_gain,
    liquidation_wealth,
    pnl_attribution,
)

__all__ = [
    "trade",
    "accrue",
    "collateral_transfer",
    "cash_flow",
    "exercise_delivery",
    "settle",
    "futures_variation",
    "financed_gain",
    "liquidation_wealth",
    "pnl_attribution",
]
