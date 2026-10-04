"""Dedicated specialist architectures backed by NeuralForecast, including panels."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from .. import GlobalFittedForecaster
from os import PathLike

if TYPE_CHECKING:
    from neuralforecast import NeuralForecast
from copy import deepcopy
from dataclasses import dataclass, field
from hashlib import sha256
from importlib.metadata import version
from typing import TypedDict, Unpack, overload
from uuid import uuid4

import numpy as np

from .. import _validation as v
from ..interfaces.base import Diagnostics, FittedForecaster
from ..interfaces.data import (
    Covariates,
    ForecastHorizon,
    TimeSeries,
    as_series,
    prepare,
)


class TftforecasterOptions(TypedDict, total=False):
    """Optional keyword settings for TFTForecaster."""

    context_length: int
    max_horizon: int
    max_steps: int
    validation_size: int
    seed: int
    device: str
    quantiles: tuple[float, ...]
    model_options: Mapping[str, object]


class PatchtstforecasterOptions(TypedDict, total=False):
    """Optional keyword settings for PatchTSTForecaster."""

    context_length: int
    max_horizon: int
    max_steps: int
    validation_size: int
    seed: int
    device: str
    quantiles: tuple[float, ...]
    model_options: Mapping[str, object]


class NbeatsforecasterOptions(TypedDict, total=False):
    """Optional keyword settings for NBEATSForecaster."""

    context_length: int
    max_horizon: int
    max_steps: int
    validation_size: int
    seed: int
    device: str
    quantiles: tuple[float, ...]
    model_options: Mapping[str, object]


class NhitsforecasterOptions(TypedDict, total=False):
    """Optional keyword settings for NHiTSForecaster."""

    context_length: int
    max_horizon: int
    max_steps: int
    validation_size: int
    seed: int
    device: str
    quantiles: tuple[float, ...]
    model_options: Mapping[str, object]


class EncoderdecoderforecasterOptions(TypedDict, total=False):
    """Optional keyword settings for EncoderDecoderForecaster."""

    context_length: int
    max_horizon: int
    max_steps: int
    validation_size: int
    seed: int
    device: str
    quantiles: tuple[float, ...]
    model_options: Mapping[str, object]


@dataclass(frozen=True)
class Panel:
    """Separate histories; windows never cross series boundaries."""

    series: dict[str, TimeSeries]

    def __init__(self, series: Mapping[str, TimeSeries | ArrayLike]) -> None:
        object.__setattr__(self, "series", series)
        self.__post_init__()

    def __post_init__(self) -> None:
        if not self.series or any(not isinstance(k, str) or not k for k in self.series):
            raise ValueError("panel requires nonempty string series identifiers")
        series = {name: as_series(value) for name, value in self.series.items()}
        first = next(iter(series.values()))
        for value in series.values():
            if value.names != first.names:
                raise ValueError(
                    "panel histories must share target names and dimension"
                )
        object.__setattr__(self, "series", series)


@dataclass(frozen=True)
class NeuralForecastForecaster:
    """Wrap a supplied neural forecasting specification in the checked forecast result contract."""

    model: str = "TFT"
    context_length: int = 24
    max_horizon: int = 6
    max_steps: int = 100
    validation_size: int = 0
    seed: int = 0
    device: str = "cpu"
    quantiles: tuple[float, ...] = ()
    model_options: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        v.choice(self.model, ("TFT", "PatchTST", "NBEATS", "NHITS", "RNN"), "model")
        for name in ("context_length", "max_horizon", "max_steps"):
            v.integer(getattr(self, name), name, 1)
        v.integer(self.validation_size, "validation_size")
        v.seed(self.seed)
        if self.quantiles:
            object.__setattr__(self, "quantiles", tuple(v.levels(self.quantiles)))
        reserved = {
            "h",
            "input_size",
            "max_steps",
            "loss",
            "alias",
            "random_seed",
            "accelerator",
            "devices",
            "futr_exog_list",
            "hist_exog_list",
            "stat_exog_list",
        }
        if reserved & self.model_options.keys():
            raise ValueError(
                "model_options cannot override contract, data schema, or training controls"
            )
        object.__setattr__(self, "model_options", deepcopy(self.model_options))

    @overload
    def fit(
        self, data: Panel, *, covariates: None = None
    ) -> GlobalFittedForecaster: ...

    @overload
    def fit(
        self, data: TimeSeries | ArrayLike, *, covariates: Covariates | None = None
    ) -> FittedForecaster: ...

    def fit(
        self,
        data: Panel | TimeSeries | ArrayLike,
        *,
        covariates: Covariates | None = None,
    ):
        if isinstance(data, Panel):
            if covariates is not None:
                raise ValueError("panel covariates belong to each TimeSeries")
            return self.fit_panel(data)
        history = prepare(data, covariates)
        global_model = self.fit_panel(Panel({"series": history}))
        return FittedForecaster(
            _SpecialistAdapter(global_model),
            history,
            self,
            Diagnostics(
                "neuralforecast." + self.model, message="Shared sequence training"
            ),
            {"model_options": self.model_options},
            "neuralforecast." + self.model,
            self.seed,
        )

    def fit_panel(self, panel):
        from .._backends.specialist import train_panel

        return train_panel(self, panel if isinstance(panel, Panel) else Panel(panel))


@dataclass(frozen=True)
class GlobalFittedForecaster:
    """Fitted shared forecasting model for named series; outputs retain each series identity and horizon."""

    engine: NeuralForecast = field(repr=False)
    panel: Panel
    specification: NeuralForecastForecaster
    schema: dict[str, int]
    coordinates: tuple[tuple[str, str, int], ...]
    artifact_identity: str = field(default_factory=lambda: str(uuid4()))
    dependency_versions: dict[str, str] = field(
        default_factory=lambda: {
            name: version(name) for name in ("numpy", "torch", "neuralforecast")
        }
    )

    @property
    def model_identity(self):
        return "neuralforecast." + self.specification.model

    @property
    def random_state(self):
        return self.specification.seed

    @property
    def cutoff(self):
        return {name: series.cutoff for name, series in self.panel.series.items()}

    @property
    def training_digest(self):
        digest = sha256()
        for name, series in sorted(self.panel.series.items()):
            digest.update(name.encode())
            for array in (series.values, series.timestamps):
                digest.update(str(array.dtype).encode())
                digest.update(str(array.shape).encode())
                digest.update(array.tobytes())
            digest.update(repr(series.names).encode())
            if series.covariates is not None:
                for key in ("known_future", "observed_past", "static"):
                    value = getattr(series.covariates, key)
                    digest.update(key.encode())
                    if value is not None:
                        digest.update(str(value.shape).encode())
                        digest.update(value.tobytes())
        return digest.hexdigest()

    def save(self, path):
        """Persist a trusted Python artifact in a compatible backend environment."""
        import pickle
        from pathlib import Path

        Path(path).write_bytes(
            pickle.dumps(
                {"format": "convexity.forecast.panel.v1", "model": self}, protocol=5
            )
        )

    def forecast(
        self,
        horizon: int | ForecastHorizon | ArrayLike | None = None,
        *,
        covariates: Mapping[str, ArrayLike] | None = None,
    ):
        from .._backends.specialist import predict_panel

        return predict_panel(
            self,
            self.specification.max_horizon if horizon is None else horizon,
            covariates or {},
        )

    def update(self, panel, *, mode: str = "refit"):
        from .._backends.specialist import train_panel

        v.choice(mode, ("refit", "warm_start"), "mode")
        panel = panel if isinstance(panel, Panel) else Panel(panel)
        if set(panel.series) != set(self.panel.series):
            raise ValueError("panel updates must retain the fitted series identifiers")
        joined = {}
        for key, old in self.panel.series.items():
            # Reuse the central checked concatenation contract without fitting.
            from ..interfaces.base import concatenate_history

            joined[key] = concatenate_history(old, panel.series[key])
        return train_panel(
            self.specification, Panel(joined), self if mode == "warm_start" else None
        )


@dataclass(frozen=True)
class _SpecialistAdapter:
    fitted: GlobalFittedForecaster

    def forecast(self, steps, covariates, *, rng: np.random.Generator | None = None):
        if steps > self.fitted.specification.max_horizon:
            raise ValueError("forecast exceeds trained specialist horizon")
        from .._backends.specialist import predict_panel

        # The single-series facade aligns timestamps after dense prediction.
        return predict_panel(self.fitted, steps, {"series": covariates}, align=False)[
            "series"
        ]

    def update(self, history, *, mode):
        from .._backends.specialist import train_panel

        if mode != "warm_start":
            raise ValueError("specialist neural models support mode=warm_start")
        fitted = train_panel(
            self.fitted.specification, Panel({"series": history}), self.fitted
        )
        return FittedForecaster(
            _SpecialistAdapter(fitted),
            history,
            self.fitted.specification,
            Diagnostics("neuralforecast", message="Warm-start training"),
            model_identity="neuralforecast." + self.fitted.specification.model,
        )


def TFTForecaster(**kwargs: Unpack[TftforecasterOptions]) -> NeuralForecastForecaster:
    """Construct the TFT forecasting specification.

    NeuralForecastForecaster(model: 'str' = 'TFT', context_length: 'int' = 24, max_horizon: 'int' = 6, max_steps: 'int' = 100, validation_size: 'int' = 0, seed: 'int' = 0, device: 'str' = 'cpu', quantiles: 'tuple[float, ...]' = (), model_options: 'dict' = <factory>)
    Fit with forecasting.fit(TimeSeries(values), specification). Forecast rows
    are horizons and columns are targets; variances use squared target units.
    Optional settings: context_length, max_horizon, max_steps, validation_size, seed, device, quantiles, model_options."""
    return NeuralForecastForecaster(model="TFT", **kwargs)


def PatchTSTForecaster(
    **kwargs: Unpack[PatchtstforecasterOptions],
) -> NeuralForecastForecaster:
    """Construct the PatchTST forecasting specification.

    NeuralForecastForecaster(model: 'str' = 'TFT', context_length: 'int' = 24, max_horizon: 'int' = 6, max_steps: 'int' = 100, validation_size: 'int' = 0, seed: 'int' = 0, device: 'str' = 'cpu', quantiles: 'tuple[float, ...]' = (), model_options: 'dict' = <factory>)
    Fit with forecasting.fit(TimeSeries(values), specification). Forecast rows
    are horizons and columns are targets; variances use squared target units.
    Optional settings: context_length, max_horizon, max_steps, validation_size, seed, device, quantiles, model_options."""
    return NeuralForecastForecaster(model="PatchTST", **kwargs)


def NBEATSForecaster(
    **kwargs: Unpack[NbeatsforecasterOptions],
) -> NeuralForecastForecaster:
    """Construct the NBEATS forecasting specification.

    NeuralForecastForecaster(model: 'str' = 'TFT', context_length: 'int' = 24, max_horizon: 'int' = 6, max_steps: 'int' = 100, validation_size: 'int' = 0, seed: 'int' = 0, device: 'str' = 'cpu', quantiles: 'tuple[float, ...]' = (), model_options: 'dict' = <factory>)
    Fit with forecasting.fit(TimeSeries(values), specification). Forecast rows
    are horizons and columns are targets; variances use squared target units.
    Optional settings: context_length, max_horizon, max_steps, validation_size, seed, device, quantiles, model_options."""
    return NeuralForecastForecaster(model="NBEATS", **kwargs)


def NHiTSForecaster(
    **kwargs: Unpack[NhitsforecasterOptions],
) -> NeuralForecastForecaster:
    """Construct the NHiTS forecasting specification.

    NeuralForecastForecaster(model: 'str' = 'TFT', context_length: 'int' = 24, max_horizon: 'int' = 6, max_steps: 'int' = 100, validation_size: 'int' = 0, seed: 'int' = 0, device: 'str' = 'cpu', quantiles: 'tuple[float, ...]' = (), model_options: 'dict' = <factory>)
    Fit with forecasting.fit(TimeSeries(values), specification). Forecast rows
    are horizons and columns are targets; variances use squared target units.
    Optional settings: context_length, max_horizon, max_steps, validation_size, seed, device, quantiles, model_options."""
    return NeuralForecastForecaster(model="NHITS", **kwargs)


def EncoderDecoderForecaster(
    **kwargs: Unpack[EncoderdecoderforecasterOptions],
) -> NeuralForecastForecaster:
    """Construct the EncoderDecoder forecasting specification.

    NeuralForecastForecaster(model: 'str' = 'TFT', context_length: 'int' = 24, max_horizon: 'int' = 6, max_steps: 'int' = 100, validation_size: 'int' = 0, seed: 'int' = 0, device: 'str' = 'cpu', quantiles: 'tuple[float, ...]' = (), model_options: 'dict' = <factory>)
    Fit with forecasting.fit(TimeSeries(values), specification). Forecast rows
    are horizons and columns are targets; variances use squared target units.
    Optional settings: context_length, max_horizon, max_steps, validation_size, seed, device, quantiles, model_options."""
    return NeuralForecastForecaster(model="RNN", **kwargs)


def load_panel_forecaster(path: str | PathLike[str]) -> GlobalFittedForecaster:
    """Load a trusted shared-panel artifact saved by GlobalFittedForecaster."""
    import pickle
    from pathlib import Path

    payload = pickle.loads(Path(path).read_bytes())
    if (
        not isinstance(payload, dict)
        or payload.get("format") != "convexity.forecast.panel.v1"
    ):
        raise ValueError("unsupported panel artifact format")
    model = payload.get("model")
    if not isinstance(model, GlobalFittedForecaster):
        raise ValueError("artifact must contain a GlobalFittedForecaster")
    return model
