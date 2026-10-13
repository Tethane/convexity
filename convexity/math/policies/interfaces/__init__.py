"""Owned problem/result records and structural extension protocols for policies.

Ordinary workflows may import these names directly from convexity.math.policies.
See docs/math/reference/policies.md for public signatures and mathematical conventions.
"""
from .base import PolicyState, ActionDecision, Policy

__all__ = ["PolicyState", "ActionDecision", "Policy"]
