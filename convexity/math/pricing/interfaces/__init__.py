"""Owned problem/result records and structural extension protocols for pricing.

Ordinary workflows may import these names directly from convexity.math.pricing.
See docs/reference/pricing.md for public signatures and mathematical conventions.
"""
from .base import PricingProblem, PricingResult, PricingEngine, price

__all__ = ["PricingProblem", "PricingResult", "PricingEngine", "price"]
