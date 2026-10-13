"""Concrete specifications and convenience functions for policies.

Ordinary workflows may import these names directly from convexity.math.policies.
See docs/math/reference/policies.md for public signatures and mathematical conventions.
"""
from .thresholds import Hysteresis, no_action

__all__ = ["Hysteresis", "no_action"]
