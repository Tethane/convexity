"""Owned problem/result records and structural extension protocols for contracts.

Ordinary workflows may import these names directly from convexity.math.contracts.
See docs/reference/contracts.md for public signatures and mathematical conventions.
"""
from .base import Payoff, Exercise, VanillaContract, CashFlow

__all__ = ["Payoff", "Exercise", "VanillaContract", "CashFlow"]
