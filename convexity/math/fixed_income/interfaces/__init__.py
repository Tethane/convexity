"""Owned problem/result records and structural extension protocols for fixed_income.

Ordinary workflows may import these names directly from convexity.math.fixed_income.
"""

from .base import BondResult
from .base import CarryResult, BondPricer

__all__ = ["CarryResult", "BondPricer", "BondResult"]
