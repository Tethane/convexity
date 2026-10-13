"""Concrete specifications and convenience functions for rates.

Ordinary workflows may import these names directly from convexity.math.rates.
See docs/math/reference/rates.md for public signatures and mathematical conventions.
"""
from .models import (
    ShortRateModel,
    gaussian_hjm,
    market_model_paths,
    EmpiricalCurveForecaster,
    inflation_curve,
    inflation_index,
)
from .engines import rate_option_price, inflation_swap_price
from .inflation import InflationDynamics
from .contracts import (
    rate_index,
    cap_floor,
    interest_rate_swap,
    swaption,
    swap_price,
    inflation_swap,
    forward_market_model,
    gaussian_bond_price,
)
from .processes import HullWhiteProcess

__all__ = [
    "InflationDynamics",
    "ShortRateModel",
    "gaussian_hjm",
    "market_model_paths",
    "EmpiricalCurveForecaster",
    "inflation_curve",
    "inflation_index",
    "rate_option_price",
    "inflation_swap_price",
    "rate_index",
    "cap_floor",
    "interest_rate_swap",
    "swaption",
    "swap_price",
    "inflation_swap",
    "forward_market_model",
    "gaussian_bond_price",
    "HullWhiteProcess",
]
