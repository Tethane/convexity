"""User extension contracts, fitted artifacts, and immutable refit updates."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy.typing import ArrayLike

from ..._typing import FloatArray

if TYPE_CHECKING:
    from .. import FittedForecaster
from copy import deepcopy
from dataclasses import dataclass, field, replace
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from os import PathLike
from typing import Protocol, cast
from uuid import uuid4

import numpy as np

from .. import _validation as v
from .data import Covariates, ForecastHorizon, TimeSeries, as_series
from .result import ForecastResult


class Forecaster(Protocol):
    """Fit supplied time-series observations into a reusable fitted forecasting contract."""

    def fit(
        self, data: TimeSeries | ArrayLike, *, covariates: Covariates | None = None
    ) -> "FittedForecaster": ...


class ForecastAdapter(Protocol):
    """Produce a ForecastResult for requested future steps; advertise only uncertainty actually provided by the model."""

    def forecast(
        self,
        steps: int,
        covariates: FloatArray | None,
        *,
        rng: np.random.Generator | None = None,
    ) -> ForecastResult:
        """Return dense steps 1..steps, using only supplied future-known data."""
        ...


@dataclass(frozen=True)
class Diagnostics:
    """Backend termination diagnostics. Inspect converged and message before using a fitted approximation; iterations may be unavailable."""

    backend: str = "user"
    converged: bool = True
    message: str = ""
    iterations: int | None = None
    warnings: tuple[str, ...] = ()
    residuals: FloatArray | None = cast("FloatArray | None", None)
    loss_history: FloatArray | None = cast("FloatArray | None", None)
    validation_history: FloatArray | None = cast("FloatArray | None", None)

    def __post_init__(self) -> None:
        v.boolean(self.converged, "converged")
        if self.iterations is not None:
            v.integer(self.iterations, "iterations")
        for name in ("loss_history", "validation_history"):
            if getattr(self, name) is not None:
                object.__setattr__(
                    self, name, v.array(getattr(self, name), name, ndim=1)
                )
        if self.residuals is not None:
            object.__setattr__(
                self, "residuals", v.matrix(self.residuals, "residuals", missing=True)
            )

    def __init__(
        self,
        backend: str = "user",
        converged: bool = True,
        message: str = "",
        iterations: int | None = None,
        warnings: tuple[str, ...] = (),
        residuals: ArrayLike | None = None,
        loss_history: ArrayLike | None = None,
        validation_history: ArrayLike | None = None,
    ) -> None:
        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "converged", converged)
        object.__setattr__(self, "message", message)
        object.__setattr__(self, "iterations", iterations)
        object.__setattr__(self, "warnings", warnings)
        object.__setattr__(self, "residuals", residuals)
        object.__setattr__(self, "loss_history", loss_history)
        object.__setattr__(self, "validation_history", validation_history)
        self.__post_init__()


class ConvergenceError(RuntimeError):
    """Raised when a checked numerical method does not converge; result retains the attempted solution and diagnostics."""

    def __init__(self, result: FittedForecaster) -> None:
        self.result = result
        super().__init__(result.diagnostics.message or "forecast fit did not converge")


@dataclass(frozen=True)
class FittedForecaster:
    """Fitted forecasting adapter with retained history, specification and diagnostics. Future covariates cover the maximum requested step."""

    adapter: ForecastAdapter = field(repr=False)
    history: TimeSeries
    specification: Forecaster | None = field(default=None, repr=False)
    diagnostics: Diagnostics = field(default_factory=Diagnostics)
    parameters: dict[str, object] = field(default_factory=dict)
    model_identity: str = "user"
    random_state: int | None = None
    artifact_identity: str = field(default_factory=lambda: str(uuid4()))
    dependency_versions: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.history, TimeSeries) or not callable(
            self.adapter.forecast
        ):
            raise TypeError(
                "fitted forecasters require TimeSeries history and a forecast adapter"
            )
        object.__setattr__(self, "parameters", deepcopy(self.parameters))
        if self.random_state is not None:
            v.seed(self.random_state)
        if not self.dependency_versions:
            versions = {}
            for name in (
                "convexity",
                "numpy",
                "scipy",
                "statsmodels",
                "scikit-learn",
                "arch",
                "torch",
                "sktime",
                "chronos-forecasting",
                "timesfm",
                "neuralforecast",
            ):
                try:
                    versions[name] = version(name)
                except PackageNotFoundError:
                    pass
            object.__setattr__(self, "dependency_versions", versions)

    @property
    def cutoff(self):
        return self.history.cutoff

    @property
    def nobs(self):
        return len(self.history)

    @property
    def schema(self):
        cov = self.history.covariates
        return {
            "targets": self.history.names,
            "timestamp_dtype": str(self.history.timestamps.dtype),
            **{
                name: 0
                if cov is None or getattr(cov, name) is None
                else getattr(cov, name).shape[-1]
                for name in ("known_future", "observed_past", "static")
            },
        }

    @property
    def training_digest(self):
        digest = sha256()
        for x in (self.history.values, self.history.timestamps):
            digest.update(str(x.dtype).encode())
            digest.update(str(x.shape).encode())
            digest.update(x.tobytes())
        if self.history.covariates is not None:
            for name in ("known_future", "observed_past", "static"):
                x = getattr(self.history.covariates, name)
                digest.update(name.encode())
                if x is not None:
                    digest.update(str(x.shape).encode())
                    digest.update(x.tobytes())
        digest.update(repr(self.history.names).encode())
        return digest.hexdigest()

    def forecast(
        self,
        horizon: int | ForecastHorizon | ArrayLike,
        *,
        covariates: Covariates | ArrayLike | None = None,
        rng: np.random.Generator | None = None,
    ):
        h = (
            horizon
            if isinstance(horizon, ForecastHorizon)
            else ForecastHorizon(horizon)
        )
        h = h.align(
            self.history,
            check_frequency=not getattr(self.adapter, "elapsed_time", False),
        )
        if rng is not None:
            v.generator(rng)
        if isinstance(covariates, Covariates):
            if covariates.observed_past is not None or covariates.static is not None:
                raise ValueError(
                    "forecast-time covariates may contain only known_future values"
                )
            covariates = covariates.known_future
        width = self.schema["known_future"]
        if width:
            if covariates is None:
                raise ValueError("future known covariates are required")
            covariates = v.matrix(covariates, "future covariates")
            if covariates.shape != (h.maximum, width):
                raise ValueError(
                    "future covariates must cover every step through the maximum horizon"
                )
        elif covariates is not None:
            raise ValueError("model was fitted without known-future covariates")
        if getattr(self.adapter, "elapsed_time", False):
            result = getattr(self.adapter, "forecast_at")(h, covariates, rng=rng)
            if not isinstance(result, ForecastResult) or result.shape != (
                len(h),
                self.history.dimension,
            ):
                raise ValueError("elapsed-time adapter returned incorrect shape")
            return replace(
                result,
                horizon=h,
                cutoff=self.cutoff,
                model_identity=self.model_identity,
                metadata={
                    **result.metadata,
                    "artifact_identity": self.artifact_identity,
                    "training_digest": self.training_digest,
                    "target_names": self.history.names,
                },
            )
        result = self.adapter.forecast(h.maximum, covariates, rng=rng)
        if not isinstance(result, ForecastResult) or result.shape != (
            h.maximum,
            self.history.dimension,
        ):
            raise ValueError(
                "adapter must return ForecastResult with shape (steps,targets)"
            )
        if result.horizon is None or not np.array_equal(
            result.horizon.steps, np.arange(1, h.maximum + 1)
        ):
            raise ValueError("adapter must return dense relative forecast steps")
        result = result.select(h.steps - 1, horizon=h)
        return replace(
            result,
            model_identity=self.model_identity,
            cutoff=self.cutoff,
            metadata={
                **result.metadata,
                "artifact_identity": self.artifact_identity,
                "training_digest": self.training_digest,
                "target_names": self.history.names,
            },
        )

    def predict(
        self,
        horizon: int | ForecastHorizon | ArrayLike,
        *,
        covariates: Covariates | ArrayLike | None = None,
        rng: np.random.Generator | None = None,
    ):
        return self.forecast(horizon, covariates=covariates, rng=rng).point

    def update(
        self,
        data: TimeSeries | ArrayLike,
        *,
        covariates: Covariates | None = None,
        mode: str = "refit",
    ):
        """Append observations and refit, incrementally filter, or warm-start."""
        v.choice(mode, ("refit", "incremental", "warm_start"), "mode")
        if self.specification is None:
            raise NotImplementedError(
                "a user artifact needs a specification to support refit updates"
            )
        history = concatenate_history(self.history, data, covariates)
        if mode != "refit":
            if not hasattr(self.adapter, "update"):
                raise NotImplementedError(
                    "this engine does not implement incremental/warm-start updates"
                )
            return getattr(self.adapter, "update")(history, mode=mode)
        return self.specification.fit(history)

    def save(self, path):
        """Save a Python pickle artifact; load only files from trusted sources."""
        import pickle
        from pathlib import Path

        Path(path).write_bytes(
            pickle.dumps({"format": "convexity.forecast.v1", "model": self}, protocol=5)
        )


def load_forecaster(path: str | PathLike[str]) -> FittedForecaster:
    """Restore a trusted pickle in a compatible Python/dependency environment."""
    import pickle
    from pathlib import Path

    payload = pickle.loads(Path(path).read_bytes())
    if (
        not isinstance(payload, dict)
        or payload.get("format") != "convexity.forecast.v1"
    ):
        raise ValueError("unsupported forecast artifact format")
    result = payload.get("model")
    if not isinstance(result, FittedForecaster):
        raise ValueError("artifact does not contain a fitted forecaster")
    return result


def fit(
    data: TimeSeries | ArrayLike,
    forecaster: Forecaster,
    *,
    covariates: Covariates | None = None,
) -> FittedForecaster:
    """Fit a Forecaster to TimeSeries or time-major values (time,target).

    Return FittedForecaster with history, diagnostics and checked forecast methods.
    Covariates distinguish known future information from observed-past inputs.
    Point predictions alone do not create a distribution or temporal dependence."""
    history = as_series(data, covariates)
    result = forecaster.fit(history)
    if (
        not isinstance(result, FittedForecaster)
        or result.history.names != history.names
        or result.cutoff != history.cutoff
    ):
        raise ValueError("forecaster must return an aligned FittedForecaster")
    return result


def concatenate_history(
    old_history, data: TimeSeries | ArrayLike, covariates: Covariates | None = None
):
    if isinstance(data, TimeSeries):
        new = as_series(data, covariates)
    else:
        values = v.matrix(data, "update values", missing=True)
        if old_history.frequency is None:
            raise ValueError(
                "irregular history updates require explicit TimeSeries timestamps"
            )
        times = old_history.cutoff + old_history.frequency * np.arange(
            1, len(values) + 1
        )
        new = TimeSeries(values, times, covariates, old_history.names)
    if new.names != old_history.names or new.timestamps[0] <= old_history.cutoff:
        raise ValueError("updates must preserve target names and follow the cutoff")
    oldcov, newcov = (
        old_history.covariates or Covariates(),
        new.covariates or Covariates(),
    )
    joined = {}
    for name in ("known_future", "observed_past", "static"):
        a, b = getattr(oldcov, name), getattr(newcov, name)
        if (a is None) != (b is None):
            raise ValueError("update covariate schema must match the fitted history")
        if a is not None:
            if a.shape[-1] != b.shape[-1] or (
                name == "static" and not np.array_equal(a, b)
            ):
                raise ValueError("update covariate schema or static values changed")
            joined[name] = a if name == "static" else np.concatenate((a, b))
    history = TimeSeries(
        np.concatenate((old_history.values, new.values)),
        np.concatenate((old_history.timestamps, new.timestamps)),
        Covariates(**joined) if joined else None,
        old_history.names,
    )
    return history
