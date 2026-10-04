"""Owned problem/result records and structural extension protocols for credit.

Ordinary workflows may import these names directly from convexity.math.credit.
See docs/reference/credit.md for public signatures and mathematical conventions.
"""
from .base import DefaultLaw
from .results import CDSResult, HazardEstimate, StructuralCreditResult

__all__ = ["CDSResult", "HazardEstimate", "StructuralCreditResult", "DefaultLaw"]
