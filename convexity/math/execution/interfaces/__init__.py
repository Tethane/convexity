"""Owned problem/result records and structural extension protocols for execution.

Ordinary workflows may import these names directly from convexity.math.execution.
See docs/reference/execution.md for public signatures and mathematical conventions.
"""
from .base import LiquidationResult, LogisticFit, CostModel, FillLaw

__all__ = ["LiquidationResult", "LogisticFit", "CostModel", "FillLaw"]
