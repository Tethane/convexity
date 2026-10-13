"""Owned problem/result records and structural extension protocols for stopping.

Ordinary workflows may import these names directly from convexity.math.stopping.
See docs/math/reference/stopping.md for public signatures and mathematical conventions.
"""
from .base import StoppingProblem, StoppingResult, StoppingSolver

from .results import PolicyEvaluation

__all__ = ["PolicyEvaluation", "StoppingProblem", "StoppingResult", "StoppingSolver"]
