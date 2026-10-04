"""Numerical diagnostics, including data suitable for calibration/ACF plots."""
from __future__ import annotations
from typing import TYPE_CHECKING
from collections.abc import Callable, Sequence, Mapping
from numpy.typing import ArrayLike, DTypeLike
from .._typing import FloatArray, NumericArray, BoolArray, IntArray, DateLike

from typing import TYPE_CHECKING
from collections.abc import Callable, Sequence, Mapping
from numpy.typing import ArrayLike, DTypeLike
from .._typing import FloatArray, NumericArray, BoolArray, IntArray, DateLike
if TYPE_CHECKING:
    from . import FittedForecaster, ForecastHorizon

import numpy as np

from . import _validation as v


def residual_diagnostics(residuals: ArrayLike, *, lags: int=10) -> dict[str, FloatArray]:
    """Per-target mean, RMSE, ACF and Ljung-Box; finite residual rows only.

    Leading state-space initialization NaNs can be removed explicitly by the
    caller. Interior missing observations are not compressed into adjacent time.
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.tsa.stattools import acf
    x = v.matrix(residuals, 'residuals')
    lags = v.integer(lags, 'lags', 1)
    if lags >= len(x):
        raise ValueError('lags must be less than the number of residual rows')
    if np.any(np.var(x, axis=0) == 0):
        raise ValueError('ACF and Ljung-Box are undefined for constant residuals')
    tables = [acorr_ljungbox(x[:, k], lags=[lags], return_df=True) for k in range(x.shape[1])]
    correlations = []
    for k in range(x.shape[1]):
        result = acf(x[:, k], nlags=lags, fft=False)
        # statsmodels supports both the legacy array and an AcfResult record.
        correlations.append(v.array(getattr(result, 'acf', result), 'acf', ndim=1))
    return {'mean': v.array(x.mean(axis=0), 'mean'), 'rmse': v.array(np.sqrt((x*x).mean(axis=0)), 'rmse'),
            'acf': v.array(np.stack(correlations, axis=1), 'acf'),
            'ljung_box_statistic': v.array([t['lb_stat'].iloc[0] for t in tables], 'statistic'),
            'ljung_box_pvalue': v.array([t['lb_pvalue'].iloc[0] for t in tables], 'pvalue')}


def covariate_sensitivity(fitted: FittedForecaster, horizon: int | ForecastHorizon | ArrayLike, covariates: ArrayLike, *, delta: float=1e-4) -> FloatArray:
    """Central differences of point forecasts per known-future input coordinate.

    Returns (forecast_horizon,targets,future_rows,covariate_columns). This is
    local numerical sensitivity, not causal attribution.
    """
    x = v.matrix(covariates, 'covariates')
    delta = v.scalar(delta, 'delta', minimum=0, strict=True)
    base = fitted.predict(horizon, covariates=x)
    out = np.empty(base.shape + x.shape)
    for row, col in np.ndindex(x.shape):
        plus, minus = x.copy(), x.copy()
        plus[row, col] += delta
        minus[row, col] -= delta
        out[:, :, row, col] = (fitted.predict(horizon, covariates=plus) - fitted.predict(horizon, covariates=minus)) / (2 * delta)
    return v.array(out, 'sensitivity')
