"""Causal lag/rolling features with recursive, direct and multi-output fitting."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol

import numpy as np
from numpy.typing import ArrayLike

from ..._typing import FloatArray
from ..interfaces.data import TimeSeries

if TYPE_CHECKING:
    from ...learning import Learner
    from .. import Covariates, TimeSeries

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from .. import _validation as v
from ..interfaces.base import Diagnostics, FittedForecaster
from ..interfaces.data import prepare
from ..interfaces.result import ForecastResult


def lag_features(
    values: FloatArray,
    origin,
    lags: Sequence[int],
    rolling_windows=(),
    observed=None,
    static=None,
):
    """Features at an exclusive origin: every target/past row is < origin."""
    parts = [values[origin - np.asarray(lags)].ravel()]
    for window in rolling_windows:
        block = values[origin - window : origin]
        parts.extend((block.mean(axis=0), block.std(axis=0)))
    if observed is not None:
        parts.append(observed[origin - np.asarray(lags)].ravel())
    if static is not None:
        parts.append(static)
    return np.concatenate(parts)


class RegressionEstimator(Protocol):
    """Fit a supervised numeric predictor; fit and predict receive sample rows."""

    def fit(self, features: FloatArray, response: FloatArray, /) -> object: ...
    def predict(self, features: FloatArray, /) -> ArrayLike: ...


@dataclass(frozen=True)
class RegressionForecaster:
    """Fit a regression learner to explicit lag features; forecast strategy controls multi-step recursion."""

    estimator: RegressionEstimator | Learner | None = None
    lags: tuple[int, ...] = (1, 2, 3)
    rolling_windows: tuple[int, ...] = ()
    strategy: str = "recursive"
    max_horizon: int = 1
    missing: str = "raise"

    def __post_init__(self) -> None:
        for name in ("lags", "rolling_windows"):
            values = tuple(v.integer(x, name, 1) for x in getattr(self, name))
            if len(set(values)) != len(values) or (name == "lags" and not values):
                raise ValueError(
                    "lags must be nonempty and feature windows must be unique"
                )
            object.__setattr__(self, name, values)
        v.choice(self.strategy, ("recursive", "direct", "multioutput"), "strategy")
        v.choice(self.missing, ("raise", "ffill"), "missing")
        v.integer(self.max_horizon, "max_horizon", 1)
        object.__setattr__(self, "estimator", deepcopy(self.estimator))

    @property
    def context_length(self):
        return max(self.lags + self.rolling_windows)

    def fit(
        self, data: TimeSeries | ArrayLike, *, covariates: Covariates | None = None
    ) -> FittedForecaster:
        from sklearn.base import BaseEstimator, clone, is_classifier
        from sklearn.linear_model import Ridge

        history = prepare(data, covariates, missing=self.missing)
        cov = history.covariates
        known = None if cov is None else cov.known_future
        observed = None if cov is None else cov.observed_past
        static = None if cov is None else cov.static
        if self.strategy == "recursive" and observed is not None:
            raise ValueError(
                "recursive regression cannot consume unavailable future observed-past lags; use direct"
            )
        horizon = 1 if self.strategy == "recursive" else self.max_horizon
        if len(history) < self.context_length + horizon + 1:
            raise ValueError(
                "insufficient history for lag features and the training horizon"
            )
        models = []
        for lead in (
            [horizon] if self.strategy == "multioutput" else range(1, horizon + 1)
        ):
            x, y = [], []
            for origin in range(self.context_length, len(history) - lead + 1):
                row = lag_features(
                    history.values,
                    origin,
                    self.lags,
                    self.rolling_windows,
                    observed,
                    static,
                )
                if known is not None:
                    future = (
                        known[origin : origin + lead]
                        if self.strategy == "multioutput"
                        else known[origin + lead - 1]
                    )
                    row = np.concatenate((row, future.ravel()))
                x.append(row)
                target = (
                    history.values[origin : origin + lead]
                    if self.strategy == "multioutput"
                    else history.values[origin + lead - 1]
                )
                y.append(target.ravel())
            estimator = Ridge(alpha=1.0) if self.estimator is None else self.estimator
            if getattr(estimator, "_estimator_type", None) == "classifier" or (
                isinstance(estimator, BaseEstimator) and is_classifier(estimator)
            ):
                raise ValueError(
                    "regression forecasting requires a regressor, not a classifier"
                )
            try:
                model: Any = clone(estimator)
            except TypeError:
                model = deepcopy(estimator)
            target: Any = np.asarray(y)
            # Preserve true multi-output estimators. Users can supply sklearn's
            # MultiOutputRegressor for regressors accepting only scalar targets.
            fitted = model.fit(
                np.asarray(x), target[:, 0] if target.shape[1] == 1 else target
            )
            if hasattr(fitted, "capabilities") and "mean" not in fitted.capabilities:
                raise ValueError(
                    "Convexity regression forecasting requires a mean learner"
                )
            models.append(fitted)
        adapter = _RegressionAdapter(tuple(models), history, self)
        return FittedForecaster(
            adapter,
            history,
            self,
            Diagnostics("regression", message="Causal supervised windows"),
            {
                "lags": self.lags,
                "rolling_windows": self.rolling_windows,
                "strategy": self.strategy,
                "max_horizon": self.max_horizon,
            },
            "regression." + self.strategy,
        )


@dataclass(frozen=True)
class _RegressionAdapter:
    models: tuple
    history: TimeSeries
    spec: RegressionForecaster

    def forecast(self, steps, covariates, *, rng: np.random.Generator | None = None):
        history, spec = self.history, self.spec
        if spec.strategy != "recursive" and steps > spec.max_horizon:
            raise ValueError(
                "requested horizon exceeds the fitted direct/multi-output horizon"
            )
        if (
            spec.strategy == "multioutput"
            and covariates is not None
            and steps != spec.max_horizon
        ):
            raise ValueError(
                "multi-output regression with known covariates requires its full fitted horizon"
            )
        cov = history.covariates
        observed = None if cov is None else cov.observed_past
        static = None if cov is None else cov.static
        values = np.array(history.values)
        out = []
        for h in range(steps if spec.strategy != "multioutput" else 1):
            row = lag_features(
                values, len(values), spec.lags, spec.rolling_windows, observed, static
            )
            if covariates is not None:
                future = covariates if spec.strategy == "multioutput" else covariates[h]
                row = np.concatenate((row, future.ravel()))
            model = self.models[h] if spec.strategy == "direct" else self.models[0]
            prediction = v.array(
                model.predict(row[None, :]), "regression prediction"
            ).reshape(-1, history.dimension)
            expected = spec.max_horizon if spec.strategy == "multioutput" else 1
            if prediction.shape != (expected, history.dimension):
                raise ValueError(
                    "regressor returned the wrong number of target coordinates"
                )
            out.append(prediction)
            if spec.strategy == "recursive":
                values = np.concatenate((values, prediction))
        point = np.concatenate(out)[:steps]
        return ForecastResult(point, mean_values=point)
