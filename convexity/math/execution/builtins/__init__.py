"""Concrete specifications and convenience functions for execution.

Ordinary workflows may import these names directly from convexity.math.execution.
"""

from .models import QuadraticCost, BinomialFills, assignment, fit_assignment, liquidate

__all__ = [
    "QuadraticCost",
    "BinomialFills",
    "assignment",
    "fit_assignment",
    "liquidate",
]
