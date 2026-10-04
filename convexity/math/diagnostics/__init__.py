"""Diagnostics mathematics on supplied arrays and models."""

from . import interfaces, builtins
from .interfaces import TestResult, BootstrapResult, Score
from .builtins import (
    log_score,
    quantile_score,
    crps,
    energy_score,
    variogram_score,
    pit,
    interval_coverage,
    residual_test,
    stationarity_test,
    cointegration_test,
    hac_mean_covariance,
    compare_scores,
    adjust_pvalues,
    block_bootstrap,
)

__all__ = [
    "interfaces",
    "builtins",
    "TestResult",
    "BootstrapResult",
    "Score",
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
