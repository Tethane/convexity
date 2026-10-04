"""Expanding/rolling origin evaluation, baseline comparison, and temporal search."""
from __future__ import annotations
from typing import TYPE_CHECKING
from collections.abc import Callable, Sequence, Mapping
from numpy.typing import ArrayLike, DTypeLike
from .._typing import FloatArray, NumericArray, BoolArray, IntArray, DateLike
if TYPE_CHECKING:
    from . import Forecaster, SelectionResult, TimeSeries

from typing import TYPE_CHECKING
from collections.abc import Callable, Sequence, Mapping
from numpy.typing import ArrayLike, DTypeLike
from .._typing import FloatArray, NumericArray, BoolArray, IntArray, DateLike
if TYPE_CHECKING:
    from . import Forecaster, SelectionResult, TimeSeries

from typing import TYPE_CHECKING
from collections.abc import Callable, Sequence, Mapping
from numpy.typing import ArrayLike, DTypeLike
from .._typing import FloatArray, NumericArray, BoolArray, IntArray, DateLike
if TYPE_CHECKING:
    from . import BacktestResult, Forecaster, TimeSeries

from typing import TYPE_CHECKING
from collections.abc import Callable, Sequence, Mapping
from numpy.typing import ArrayLike, DTypeLike
from .._typing import FloatArray, NumericArray, BoolArray, IntArray, DateLike
if TYPE_CHECKING:
    from . import BacktestResult, Forecaster, TimeSeries

from numpy.typing import ArrayLike
from .interfaces.base import Forecaster, FittedForecaster
from .interfaces.result import ForecastResult
from .._typing import FloatArray
from dataclasses import dataclass
from itertools import product
from typing import Literal, overload, TypedDict, Unpack
from collections.abc import Mapping, Sequence, Callable

import numpy as np

from . import _validation as v
from . import metrics
from .interfaces.base import fit
from .interfaces.data import ForecastHorizon, TimeSeries, as_series


class BacktestOptions(TypedDict, total=False):
    """Relative horizons and positional training windows for temporal validation."""
    horizon: int | ForecastHorizon | ArrayLike
    initial_window: int | None
    step: int
    window: int | None
    origins: ArrayLike | None
    seed: int


class SelectionOptions(BacktestOptions, total=False):
    """Backtest settings plus the selection criterion and final refit choice."""
    metric: str
    refit: bool


@dataclass(frozen=True)
class ForecastRecord:
    """One backtest origin, training window, held-out targets and forecast with aligned horizon coordinates."""
    origin: int
    training_start: int
    training: TimeSeries
    actual: FloatArray
    forecast: ForecastResult

    def __post_init__(self):
        object.__setattr__(self, 'actual', v.matrix(self.actual, 'actual'))


@dataclass(frozen=True)
class BacktestResult:
    """Chronological held-out forecast records with scalar or per-horizon scoring."""
    records: tuple[ForecastRecord, ...]
    model_identity: str

    @property
    def errors(self):
        return v.array(np.stack([r.forecast.point - r.actual for r in self.records]), 'errors')

    @property
    def origins(self):
        return tuple(r.origin for r in self.records)

    @overload
    def score(self, metric: str = 'mae', *, by_horizon: Literal[False] = False,
              seasonality: int = 1, probability: float = .9) -> float: ...

    @overload
    def score(self, metric: str = 'mae', *, by_horizon: Literal[True],
              seasonality: int = 1, probability: float = .9) -> FloatArray: ...

    @overload
    def score(self, metric: str = 'mae', *, by_horizon: bool,
              seasonality: int = 1, probability: float = .9) -> float | FloatArray: ...

    def score(self, metric='mae', *, by_horizon=False, seasonality=1, probability=.9) -> float | FloatArray:
        """Pool origins; by_horizon=True retains (horizon,target) score axes."""
        v.boolean(by_horizon, 'by_horizon')
        axis = 0 if by_horizon else None
        actual = np.stack([r.actual for r in self.records])
        point = np.stack([r.forecast.point for r in self.records])
        if metric in ('mae', 'rmse', 'smape'):
            return getattr(metrics, metric)(actual, point, axis=axis)
        if metric == 'mase':
            scores = [metrics.mase(r.actual, r.forecast.point, r.training.values,
                                   seasonality=seasonality, axis=()) for r in self.records]
        elif metric == 'pinball':
            scores = [metrics.pinball_loss(r.actual, r.forecast.quantile(probability), probability, axis=())
                      for r in self.records]
        elif metric == 'energy':
            if by_horizon:
                raise ValueError('energy is a joint path score, not horizon-specific')
            scores = [metrics.energy_score(r.forecast, r.actual) for r in self.records]
        elif metric in ('crps', 'log_score', 'interval_coverage', 'sharpness'):
            function = getattr(metrics, metric)
            if metric == 'sharpness':
                scores = [function(r.forecast, probability, axis=()) for r in self.records]
            elif metric == 'interval_coverage':
                scores = [function(r.forecast, r.actual, probability, axis=()) for r in self.records]
            else:
                scores = [function(r.forecast, r.actual, axis=()) for r in self.records]
        else:
            raise ValueError(f'unknown forecast metric {metric}')
        return np.mean(scores, axis=axis)

    def calibration(self, levels=(.1, .5, .9)):
        levels = v.levels(levels)
        values = [np.mean([np.mean(r.actual <= r.forecast.quantile(p)) for r in self.records]) for p in levels]
        return v.array(np.column_stack((levels, values)), 'calibration')

    def conditional_errors(self, labels):
        """Group origin-level errors by supplied regime labels, without refitting."""
        labels = np.asarray(labels)
        if labels.shape != (len(self.records),):
            raise ValueError('one regime label is required per forecast origin')
        return {label.item() if hasattr(label, 'item') else label:
                v.array(abs(self.errors[labels == label]).mean(axis=0), 'conditional MAE')
                for label in np.unique(labels)}


def backtest(model: Forecaster, data: TimeSeries | ArrayLike, *, horizon: int | ForecastHorizon | ArrayLike = 1, initial_window: int | None=None, step: int=1, window: int | None=None, origins: ArrayLike | None=None, seed: int=0) -> BacktestResult:
    """Refit at every origin; expose future rows only for known-future covariates.

    An origin is the exclusive end of its training slice. Direct/multi-horizon
    estimators construct all supervised labels inside that slice. Explicit
    origins are strictly increasing positions, never timestamps.
    """
    history = as_series(data)
    h = horizon if isinstance(horizon, ForecastHorizon) else ForecastHorizon(horizon)
    if h.timestamps is not None:
        raise ValueError('backtesting derives timestamps per origin; supply relative horizon steps only')
    v.integer(step, 'step', 1)
    v.seed(seed)
    if window is not None:
        v.integer(window, 'window', 2)
    if origins is None:
        initial = max(2, len(history) // 2) if initial_window is None else v.integer(initial_window, 'initial_window', 2)
        origins = np.arange(initial, len(history) - h.maximum + 1, step)
    else:
        if initial_window is not None:
            raise ValueError('supply origins or initial_window, not both')
        raw = np.asarray(origins)
        if raw.ndim != 1 or raw.dtype.kind not in 'iu' or np.any(np.diff(raw.astype(np.int64)) <= 0):
            raise ValueError('origins must be an increasing integer vector')
        origins = raw
    if not len(origins) or np.any(origins < 2) or np.any(origins + h.maximum > len(history)):
        raise ValueError('backtest origins must leave training history and the complete horizon')
    rng = np.random.default_rng(seed)
    records = []
    for origin in origins:
        origin = int(origin)
        start = 0 if window is None else max(0, origin - window)
        train = history.slice(start, origin)
        fitted = fit(train, model)
        cov = history.covariates
        future = None if cov is None or cov.known_future is None else cov.known_future[origin:origin+h.maximum]
        rows = origin + h.steps - 1
        forecast = fitted.forecast(ForecastHorizon(h.steps, history.timestamps[rows]), covariates=future, rng=rng)
        actual = v.matrix(history.values[rows], 'held-out targets')
        records.append(ForecastRecord(origin, start, train, actual, forecast))
    return BacktestResult(tuple(records), records[-1].forecast.model_identity)


def compare(models: Mapping[str, Forecaster], data: TimeSeries | ArrayLike, *, include_baselines: bool=True, seasonality: int=1, **kwargs: Unpack[BacktestOptions]) -> dict[str, BacktestResult]:
    """Return named backtests including last/drift/mean/seasonal references."""
    from .builtins.baselines import (NaiveForecaster, DriftForecaster, MovingMeanForecaster,
                                    SeasonalNaiveForecaster, SeasonalMeanForecaster)
    v.boolean(include_baselines, 'include_baselines')
    v.integer(seasonality, 'seasonality', 1)
    if not isinstance(models, Mapping) or not models:
        raise ValueError('models must be a nonempty name-to-forecaster mapping')
    output = {name: backtest(model, data, **kwargs) for name, model in models.items()}
    if include_baselines:
        history = as_series(data)
        plain = TimeSeries(history.values, history.timestamps, names=history.names)
        baseline_models = {'baseline:last': NaiveForecaster(), 'baseline:drift': DriftForecaster(),
                           'baseline:mean': MovingMeanForecaster(window=2)}
        if seasonality > 1:
            baseline_models.update({'baseline:seasonal_naive': SeasonalNaiveForecaster(period=seasonality),
                                    'baseline:seasonal_mean': SeasonalMeanForecaster(period=seasonality)})
        for name, model in baseline_models.items():
            if name in output:
                raise ValueError(f'model name {name} is reserved for baseline comparisons')
            output[name] = backtest(model, plain, **kwargs)
    return output


@dataclass(frozen=True)
class SelectionResult:
    """Model-selection scores, named backtests, selected specification and optional refit."""
    best_name: str
    best_model: Forecaster
    scores: dict[str, float]
    backtests: dict[str, BacktestResult]
    fitted: FittedForecaster | None

    def best_by_horizon(self, metric='mae'):
        names = tuple(self.backtests)
        scores = np.stack([self.backtests[name].score(metric, by_horizon=True) for name in names])
        return np.asarray(names)[np.argmin(scores, axis=0)]


def select_model(models: Mapping[str, Forecaster], data: TimeSeries | ArrayLike, *, metric: str='mae', refit: bool=True, **kwargs: Unpack[BacktestOptions]) -> SelectionResult:
    """Temporal validation selection. Scores are selection scores, not test estimates."""
    v.boolean(refit, 'refit')
    if metric in ('interval_coverage', 'sharpness'):
        raise ValueError('coverage/sharpness alone are not model-selection objectives')
    results = compare(models, data, include_baselines=False, **kwargs)
    scores = {name: float(result.score(metric)) for name, result in results.items()}
    if not all(np.isfinite(x) for x in scores.values()):
        raise ValueError('selection scores must be finite')
    best = min(scores, key=lambda name: scores[name])
    return SelectionResult(best, models[best], scores, results, fit(data, models[best]) if refit else None)


def grid_search(factory: Callable[..., Forecaster], parameter_grid: Mapping[str, Sequence[object]], data: TimeSeries | ArrayLike, **kwargs: Unpack[SelectionOptions]) -> SelectionResult:
    """Evaluate a finite Cartesian parameter grid using temporal backtests."""
    keys = tuple(parameter_grid)
    candidates = {}
    for values in product(*(parameter_grid[key] for key in keys)):
        params = dict(zip(keys, values))
        candidates[repr(params)] = factory(**params)
    if not candidates:
        raise ValueError('parameter grid must contain candidates')
    return select_model(candidates, data, **kwargs)
