"""Deterministic references; no uncertainty is inferred from point forecasts."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from .. import Covariates, TimeSeries

from dataclasses import dataclass, field

from .. import _validation as v
from ..interfaces.base import Diagnostics, FittedForecaster
from ..interfaces.data import prepare, reject_covariates
from ..interfaces.result import ForecastResult


@dataclass(frozen=True)
class BaselineForecaster:
    """Reusable baseline specification; method and seasonality determine its deterministic point forecast."""

    strategy: str = "last"
    period: int = 1
    window: int = 5
    missing: str = "raise"

    def __post_init__(self) -> None:
        v.choice(
            self.strategy,
            ("last", "drift", "seasonal_naive", "moving_mean", "seasonal_mean"),
            "strategy",
        )
        v.integer(self.period, "period", 1)
        v.integer(self.window, "window", 1)
        v.choice(self.missing, ("raise", "ffill"), "missing")

    def fit(
        self, data: TimeSeries | ArrayLike, *, covariates: Covariates | None = None
    ) -> FittedForecaster:
        history = prepare(data, covariates, missing=self.missing)
        reject_covariates(history)
        required = (
            2
            if self.strategy == "drift"
            else self.period
            if self.strategy.startswith("seasonal")
            else self.window
            if self.strategy == "moving_mean"
            else 1
        )
        if len(history) < required:
            raise ValueError(
                f"{self.strategy} requires at least {required} observations"
            )
        return FittedForecaster(
            _BaselineAdapter(history.values, self),
            history,
            self,
            Diagnostics("numpy", message="Deterministic baseline"),
            model_identity=type(self).__name__ + ":" + self.strategy,
        )


@dataclass(frozen=True)
class _BaselineAdapter:
    values: object
    spec: BaselineForecaster

    def forecast(self, steps, covariates, *, rng: np.random.Generator | None = None):
        y, s = np.asarray(self.values), self.spec
        if s.strategy == "last":
            point = np.repeat(y[-1:], steps, axis=0)
        elif s.strategy == "drift":
            point = y[-1] + np.arange(1, steps + 1)[:, None] * (y[-1] - y[0]) / (
                len(y) - 1
            )
        elif s.strategy == "moving_mean":
            point = np.repeat(y[-s.window :].mean(axis=0, keepdims=True), steps, axis=0)
        elif s.strategy == "seasonal_naive":
            point = y[len(y) - s.period + np.arange(steps) % s.period]
        else:
            means = np.array([y[i :: s.period].mean(axis=0) for i in range(s.period)])
            point = means[(len(y) + np.arange(steps)) % s.period]
        return ForecastResult(point, mean_values=point)


@dataclass(frozen=True)
class NaiveForecaster(BaselineForecaster):
    """Repeat the most recent observation for each target and requested future step."""

    strategy: str = field(default="last", init=False)


@dataclass(frozen=True)
class DriftForecaster(BaselineForecaster):
    """Extrapolate the mean historical change from the last observation; point forecasts alone declare no distribution."""

    strategy: str = field(default="drift", init=False)


@dataclass(frozen=True)
class SeasonalNaiveForecaster(BaselineForecaster):
    """Repeat the most recent observed seasonal cycle using an explicit integer period."""

    strategy: str = field(default="seasonal_naive", init=False)


@dataclass(frozen=True)
class MovingMeanForecaster(BaselineForecaster):
    """Forecast the recent-window mean for each target; window is counted in observations."""

    strategy: str = field(default="moving_mean", init=False)


@dataclass(frozen=True)
class SeasonalMeanForecaster(BaselineForecaster):
    """Forecast historical means at matching seasonal positions, with an explicit integer period."""

    strategy: str = field(default="seasonal_mean", init=False)
