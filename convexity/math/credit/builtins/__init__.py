"""Concrete specifications and convenience functions for credit.

Ordinary workflows may import these names directly from convexity.math.credit.
See docs/reference/credit.md for public signatures and mathematical conventions.
"""
from .default import (
    HazardCurve,
    flat_hazard,
    survival_curve,
    bootstrap_cds,
    cds_price,
    risky_cashflows,
    estimate_hazard,
    MigrationModel,
    merton_structural,
)
from .default import cds_quote

__all__ = [
    "HazardCurve",
    "flat_hazard",
    "survival_curve",
    "bootstrap_cds",
    "cds_price",
    "risky_cashflows",
    "estimate_hazard",
    "MigrationModel",
    "merton_structural",
    "cds_quote",
]
