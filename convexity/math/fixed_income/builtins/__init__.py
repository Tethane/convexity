"""Concrete specifications and convenience functions for fixed_income.

Ordinary workflows may import these names directly from convexity.math.fixed_income.
See docs/reference/fixed_income.md for public signatures and mathematical conventions.
"""
from .bonds import (
    zero_coupon,
    fixed_bond,
    floating_bond,
    amortizing_bond,
    inflation_bond,
    callable_bond,
    cashflows,
    yield_from_price,
    price_from_yield,
    bond_analytics,
    key_rate_duration,
    carry_roll_down,
)
from .bonds import callable_bond_engine

__all__ = [
    "zero_coupon",
    "fixed_bond",
    "floating_bond",
    "amortizing_bond",
    "inflation_bond",
    "callable_bond",
    "cashflows",
    "yield_from_price",
    "price_from_yield",
    "bond_analytics",
    "key_rate_duration",
    "carry_roll_down",
    "callable_bond_engine",
]
