"""Owned problem/result records and structural extension protocols for optimization.

Ordinary workflows may import these names directly from convexity.math.optimization.
See docs/math/reference/optimization.md for public signatures and mathematical conventions.
"""
from .problems import (
    SecondOrderCone,
    QuadraticProgram,
    NonlinearProblem,
    SolveResult,
    Solver,
    OptimizationError,
)

__all__ = [
    "SecondOrderCone",
    "QuadraticProgram",
    "NonlinearProblem",
    "SolveResult",
    "Solver",
    "OptimizationError",
]
