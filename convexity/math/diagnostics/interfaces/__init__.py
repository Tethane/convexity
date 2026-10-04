"""Owned problem/result records and structural extension protocols for diagnostics.

Ordinary workflows may import these names directly from convexity.math.diagnostics.
See docs/reference/diagnostics.md for public signatures and mathematical conventions.
"""
from .base import TestResult, BootstrapResult, Score

__all__ = ["TestResult", "BootstrapResult", "Score"]
