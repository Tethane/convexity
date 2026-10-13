"""Concrete specifications and convenience functions for optimization.

Ordinary workflows may import these names directly from convexity.math.optimization.
See docs/math/reference/optimization.md for public signatures and mathematical conventions.
"""
from .solvers import ConvexSolver, NonlinearSolver, solve, enumerate_minimum

__all__ = ["ConvexSolver", "NonlinearSolver", "solve", "enumerate_minimum"]
