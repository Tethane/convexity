"""Concrete specifications and convenience functions for stopping.

Ordinary workflows may import these names directly from convexity.math.stopping.
See docs/reference/stopping.md for public signatures and mathematical conventions.
"""
from .solvers import BackwardInduction, ExercisePolicy, least_squares_policy

__all__ = ["BackwardInduction", "ExercisePolicy", "least_squares_policy"]
