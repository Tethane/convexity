"""Concrete specifications and convenience functions for risk.

Ordinary workflows may import these names directly from convexity.math.risk.
See docs/math/reference/risk.md for public signatures and mathematical conventions.
"""
from .measures import (
    variance,
    downside_moment,
    value_at_risk,
    expected_shortfall,
    volatility_contributions,
    es_contributions,
    drawdown,
    maximum_drawdown,
    breach_probability,
    stress,
    scenario_margin,
)

__all__ = [
    "variance",
    "downside_moment",
    "value_at_risk",
    "expected_shortfall",
    "volatility_contributions",
    "es_contributions",
    "drawdown",
    "maximum_drawdown",
    "breach_probability",
    "stress",
    "scenario_margin",
]
