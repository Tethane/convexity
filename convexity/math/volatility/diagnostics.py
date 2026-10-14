"""Residual dependence and remaining conditional heteroskedasticity diagnostics."""

from __future__ import annotations
from typing import TypedDict
from typing import TYPE_CHECKING
from collections.abc import Callable, Sequence, Mapping
from numpy.typing import ArrayLike, DTypeLike
from .._typing import FloatArray, NumericArray, BoolArray, IntArray, DateLike

from inspect import signature

import numpy as np

from ..forecasting import _validation as v
from ..forecasting.diagnostics import residual_diagnostics


class VolatilityDiagnostics(TypedDict):
    """Per-target residual and squared-residual tests, followed by ARCH LM arrays."""

    residuals: dict[str, FloatArray]
    squared_residuals: dict[str, FloatArray]
    arch_lm_statistic: FloatArray
    arch_lm_pvalue: FloatArray


def volatility_diagnostics(
    standardized_residuals: ArrayLike, *, lags: int = 10, ddof: int = 0
) -> VolatilityDiagnostics:
    """Ljung-Box on z and z**2, plus Engle ARCH LM on supplied residuals.

    Pass finite post-initialization standardized residuals. Interior NaNs are
    rejected rather than silently collapsing observation gaps. ddof is the
    caller's fitted-parameter correction for the LM statistic.
    """
    from statsmodels.stats.diagnostic import het_arch

    z = v.matrix(standardized_residuals, "standardized_residuals")
    lags, ddof = v.integer(lags, "lags", 1), v.integer(ddof, "ddof")
    if len(z) - lags <= max(lags + 1, ddof):
        raise ValueError("insufficient residuals for the requested ARCH LM regression")
    # statsmodels 0.15 adds a result-object option and deprecates its implicit
    # tuple return. Opt in when supported while retaining 0.14 compatibility.
    options = (
        {"result_object": True}
        if "result_object" in signature(het_arch).parameters
        else {}
    )
    lm = [
        het_arch(z[:, k], nlags=lags, ddof=ddof, **options) for k in range(z.shape[1])
    ]
    return {
        "residuals": residual_diagnostics(z, lags=lags),
        "squared_residuals": residual_diagnostics(z**2, lags=lags),
        "arch_lm_statistic": v.array([x[0] for x in lm], "ARCH LM statistic"),
        "arch_lm_pvalue": v.array([x[1] for x in lm], "ARCH LM pvalue"),
    }
