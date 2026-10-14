"""Owned problem/result records and structural extension protocols for execution.

Ordinary workflows may import these names directly from convexity.math.execution.
"""

from .base import LiquidationResult, LogisticFit, CostModel, FillLaw

__all__ = ["LiquidationResult", "LogisticFit", "CostModel", "FillLaw"]
