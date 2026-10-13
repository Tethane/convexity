"""Concrete specifications and convenience functions for diagnostics.

Ordinary workflows may import these names directly from convexity.math.diagnostics.
See docs/math/reference/diagnostics.md for public signatures and mathematical conventions.
"""
from .scores import (
    log_score,
    quantile_score,
    crps,
    energy_score,
    variogram_score,
    pit,
    interval_coverage,
)
from .tests import (
    residual_test,
    stationarity_test,
    cointegration_test,
    hac_mean_covariance,
    compare_scores,
    adjust_pvalues,
    block_bootstrap,
)

__all__ = [
    "log_score",
    "quantile_score",
    "crps",
    "energy_score",
    "variogram_score",
    "pit",
    "interval_coverage",
    "residual_test",
    "stationarity_test",
    "cointegration_test",
    "hac_mean_covariance",
    "compare_scores",
    "adjust_pvalues",
    "block_bootstrap",
]
