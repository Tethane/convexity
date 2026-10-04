"""Point pooling and explicit mixtures of complete path laws."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from ..._typing import FloatArray

if TYPE_CHECKING:
    from .. import Covariates, Forecaster

from dataclasses import dataclass, replace
from typing import Any, cast

from .. import _validation as v
from ..interfaces.base import Diagnostics, FittedForecaster, fit
from ..interfaces.data import TimeSeries, as_series
from ..interfaces.result import ForecastResult


@dataclass(frozen=True)
class EnsembleForecaster:
    """Combine constituent forecasts with normalized weights and explicit uncertainty-combination semantics."""

    forecasters: tuple[Forecaster, ...]
    weights: FloatArray | Callable[[TimeSeries], ArrayLike] = cast("FloatArray", None)
    method: str = "mean"
    samples: int = 1000

    def __post_init__(self) -> None:
        models = tuple(self.forecasters)
        if not models:
            raise ValueError("ensemble requires at least one forecaster")
        object.__setattr__(self, "forecasters", models)
        v.choice(self.method, ("mean", "median", "mixture"), "method")
        if callable(self.weights):
            if self.method == "median":
                raise ValueError("median does not support dynamic weights")
            v.integer(self.samples, "samples", 1)
            return
        weights = (
            np.ones(len(models))
            if self.weights is None
            else v.array(self.weights, "weights", ndim=1)
        )
        if (
            weights.shape != (len(models),)
            or np.any(weights < 0)
            or not np.any(weights > 0)
        ):
            raise ValueError(
                "weights must be nonnegative, align with models, and have positive sum"
            )
        if self.method == "median" and not np.all(weights == weights[0]):
            raise ValueError("median ensembles require equal weights")
        weights = weights / weights.max()
        object.__setattr__(self, "weights", v.array(weights / weights.sum(), "weights"))
        v.integer(self.samples, "samples", 1)

    def fit(
        self, data: TimeSeries | ArrayLike, *, covariates: Covariates | None = None
    ):
        history = as_series(data, covariates)
        if callable(self.weights):
            fixed = replace(self, weights=self.weights(history))
            return replace(fixed.fit(history), specification=self)
        models = tuple(fit(history, model) for model in self.forecasters)
        return FittedForecaster(
            _EnsembleAdapter(models, self),
            history,
            self,
            Diagnostics("ensemble", all(m.diagnostics.converged for m in models)),
            {"weights": self.weights, "method": self.method},
            "ensemble." + self.method,
        )

    def __init__(
        self,
        forecasters: tuple[Forecaster, ...],
        weights: ArrayLike | Callable[[TimeSeries], ArrayLike] | None = None,
        method: str = "mean",
        samples: int = 1000,
    ) -> None:
        object.__setattr__(self, "forecasters", forecasters)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "samples", samples)
        self.__post_init__()


@dataclass(frozen=True)
class _EnsembleAdapter:
    models: tuple
    spec: EnsembleForecaster

    def forecast(self, steps, covariates, *, rng: Any = None):
        # Call checked adapters directly: the outer fitted facade supplies times,
        # including explicit irregular future labels and sparse requested rows.
        assert not callable(self.spec.weights)
        results = [m.adapter.forecast(steps, covariates, rng=rng) for m in self.models]
        if self.spec.method == "mixture":
            v.generator(rng)
            labels = rng.choice(
                len(results), size=self.spec.samples, p=self.spec.weights
            )
            paths = np.empty((self.spec.samples,) + results[0].shape)
            for i, result in enumerate(results):
                count = np.count_nonzero(labels == i)
                if count:
                    paths[labels == i] = result.sample(int(count), rng=rng)
            return ForecastResult.from_paths(
                paths, metadata={"mixture_weights": self.spec.weights}
            )
        points = np.stack([result.point for result in results])
        if self.spec.method == "median":
            return ForecastResult(np.median(points, axis=0))
        point = np.einsum("m,mhk->hk", self.spec.weights, points)
        mean = (
            np.einsum(
                "m,mhk->hk", self.spec.weights, np.stack([r.mean() for r in results])
            )
            if all("mean" in r.capabilities for r in results)
            else None
        )
        return ForecastResult(point, mean_values=mean)


def ensemble(
    forecasters: Sequence[Forecaster],
    *,
    weights: ArrayLike | Callable[[TimeSeries], ArrayLike] | None = None,
    method: str = "mean",
    samples: int = 1000,
) -> EnsembleForecaster:
    """Construct an EnsembleForecaster from compatible forecasting specifications.

    method=mean pools points, median takes coordinatewise medians, mixture pools
    complete path laws. Weights are nonnegative masses or a history-to-weights callback."""
    return EnsembleForecaster(
        tuple(forecasters), weights=weights, method=method, samples=samples
    )


@dataclass(frozen=True)
class StackingForecaster:
    """Learn nonnegative, sum-to-one weights from temporal out-of-fold predictions.

    Validation origins see only their own historical prefixes. A full-data
    refit follows weight selection; keep separate test data for final scoring.
    """

    forecasters: tuple[Forecaster, ...]
    horizon: int = 1
    initial_window: int | None = None
    step: int = 1
    window: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "forecasters", tuple(self.forecasters))
        if not self.forecasters:
            raise ValueError("stacking requires candidate forecasters")
        v.integer(self.horizon, "horizon", 1)
        v.integer(self.step, "step", 1)

    def fit(
        self, data: TimeSeries | ArrayLike, *, covariates: Covariates | None = None
    ):
        from scipy.optimize import minimize

        from ..evaluation import backtest
        from ..interfaces.base import ConvergenceError

        history = as_series(data, covariates)
        results = [
            backtest(
                m,
                history,
                horizon=self.horizon,
                initial_window=self.initial_window,
                step=self.step,
                window=self.window,
            )
            for m in self.forecasters
        ]
        x = np.column_stack(
            [
                np.concatenate([r.forecast.point.ravel() for r in b.records])
                for b in results
            ]
        )
        y = np.concatenate([r.actual.ravel() for r in results[0].records])
        count = len(results)
        result = minimize(
            lambda w: np.mean((x @ w - y) ** 2),
            np.ones(count) / count,
            jac=lambda w: 2 * x.T @ (x @ w - y) / len(y),
            method="SLSQP",
            bounds=[(0.0, 1.0)] * count,
            constraints={
                "type": "eq",
                "fun": lambda w: w.sum() - 1,
                "jac": lambda w: np.ones(count),
            },
            options={"ftol": 1e-12, "maxiter": 1000},
        )
        pooled = EnsembleForecaster(self.forecasters, weights=result.x).fit(history)
        fitted = replace(
            pooled,
            specification=self,
            model_identity="ensemble.stacking",
            diagnostics=Diagnostics(
                "scipy.SLSQP",
                bool(result.success),
                str(result.message),
                int(result.nit),
            ),
            parameters={
                **pooled.parameters,
                "selection_mse": float(result.fun),
                "validation_origins": results[0].origins,
            },
        )
        if not result.success:
            raise ConvergenceError(fitted)
        return fitted
