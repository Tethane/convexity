"""Owned problem/result records and structural extension protocols for implied_volatility.

Ordinary workflows may import these names directly from convexity.math.implied_volatility.
See docs/reference/implied_volatility.md for public signatures and mathematical conventions.
"""
from .base import VarianceSurface, ArbitrageReport

__all__ = ["VarianceSurface", "ArbitrageReport"]
