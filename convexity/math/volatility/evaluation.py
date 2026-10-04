"""Causal variance evaluation reusing forecasting's temporal backtest records."""
from __future__ import annotations
from typing import Literal, overload
from typing import TYPE_CHECKING
from collections.abc import Callable, Sequence, Mapping
from numpy.typing import ArrayLike, DTypeLike
from .._typing import FloatArray, NumericArray, BoolArray, IntArray, DateLike
if TYPE_CHECKING:
    from . import Forecaster, TimeSeries, VarianceBacktestResult
from typing import Unpack
from ..forecasting.evaluation import BacktestOptions
from ..forecasting import Forecaster

from typing import TYPE_CHECKING
from collections.abc import Callable, Sequence, Mapping
from numpy.typing import ArrayLike, DTypeLike
from .._typing import FloatArray, NumericArray, BoolArray, IntArray, DateLike

from dataclasses import dataclass

import numpy as np

from ..forecasting import BacktestResult, ForecastRecord, backtest
from ..forecasting import _validation as v
from ..forecasting.interfaces.data import as_series
from .interfaces import TimeSeries, VolatilityForecastResult


def qlike(actual: ArrayLike, predicted: ArrayLike, *, axis: int | tuple[int, ...] | None=None) -> np.float64 | FloatArray:
    """Gaussian quasi-likelihood loss log(predicted) + actual/predicted.

    actual is a nonnegative variance proxy; predicted must be positive. This
    version permits zero realized variance and omits forecast-independent terms.
    It is not constrained to be nonnegative. Lower scores are better.
    """
    actual, predicted = v.array(actual, 'actual'), v.array(predicted, 'predicted')
    if actual.shape != predicted.shape or np.any(actual < 0) or np.any(predicted <= 0):
        raise ValueError('equal shapes, nonnegative actual and positive predicted variances are required')
    return np.mean(np.log(predicted) + actual / predicted, axis=axis)


@dataclass(frozen=True)
class VarianceBacktestResult(BacktestResult):
    """Held-out variance forecasts with proxy/score semantics retained; a proxy is not observed latent variance."""
    @overload
    def score(self, metric: str = 'qlike', *, by_horizon: Literal[False] = False, **kwargs: object) -> float: ...

    @overload
    def score(self, metric: str = 'qlike', *, by_horizon: Literal[True], **kwargs: object) -> FloatArray: ...

    @overload
    def score(self, metric: str = 'qlike', *, by_horizon: bool, **kwargs: object) -> float | FloatArray: ...

    def score(self, metric='qlike', *, by_horizon=False, **kwargs):
        if metric == 'mase':
            raise ValueError('variance backtests retain model-input training units; use an explicit variance scaling instead of MASE')
        if metric != 'qlike':
            return super().score(metric, by_horizon=by_horizon, **kwargs)
        v.boolean(by_horizon, 'by_horizon')
        return qlike(np.stack([r.actual for r in self.records]),
                     np.stack([r.forecast.point for r in self.records]), axis=0 if by_horizon else None)


def variance_backtest(model: Forecaster, data: TimeSeries | ArrayLike, *, variance_proxy: TimeSeries | ArrayLike | None=None, **kwargs: Unpack[BacktestOptions]) -> VarianceBacktestResult:
    """Refit on data prefixes; score conditional variance against a supplied proxy.

    For return models supply a proxy aligned with every input row (for example
    independently measured realized variance or explicitly chosen squared
    demeaned returns). HAR uses its realized-variance targets by default.
    Proxies enter scoring only, never fitting. TimeSeries proxies must have
    exactly matching timestamps. Horizons/origins/options are those of backtest.
    """
    history = as_series(data)
    proxy = None
    if variance_proxy is not None:
        if isinstance(variance_proxy, TimeSeries):
            if not np.array_equal(variance_proxy.timestamps, history.timestamps):
                raise ValueError('variance proxy timestamps must match training data')
            variance_proxy = variance_proxy.values
        proxy = v.matrix(variance_proxy, 'variance_proxy')
        if proxy.shape != history.values.shape or np.any(proxy < 0):
            raise ValueError('variance proxy must be nonnegative and aligned with data')
    result = backtest(model, history, **kwargs)
    records = []
    for record in result.records:
        forecast = record.forecast
        if not isinstance(forecast, VolatilityForecastResult):
            raise TypeError('variance_backtest requires VolatilityForecastResult outputs')
        if proxy is None:
            if forecast.target_kind != 'realized_variance':
                raise ValueError('return variance evaluation requires an explicit variance_proxy')
            actual = record.actual
        else:
            assert forecast.horizon is not None
            actual = proxy[record.origin + forecast.horizon.steps - 1]
        records.append(ForecastRecord(record.origin, record.training_start, record.training,
                                      actual, forecast.variance_forecast()))
    return VarianceBacktestResult(tuple(records), result.model_identity)
