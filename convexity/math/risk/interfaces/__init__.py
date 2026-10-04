"""Owned problem/result records and structural extension protocols for risk.

Ordinary workflows may import these names directly from convexity.math.risk.
See docs/reference/risk.md for public signatures and mathematical conventions.
"""
from .base import LossScenarios, RiskContributions, RiskMeasure

__all__ = ["LossScenarios", "RiskContributions", "RiskMeasure"]
