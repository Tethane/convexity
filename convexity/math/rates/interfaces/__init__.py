"""Owned problem/result records and structural extension protocols for rates.

Ordinary workflows may import these names directly from convexity.math.rates.
"""

from .base import RateModel
from .results import RatePaths
from .swaps import SwapResult

__all__ = ["RatePaths", "RateModel", "SwapResult"]
