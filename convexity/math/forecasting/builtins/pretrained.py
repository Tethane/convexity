"""Managed local checkpoints and native Chronos-2 / TimesFM 2.5 adapters."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from os import PathLike
from typing import TYPE_CHECKING, Literal, cast

from numpy.typing import ArrayLike

from ..._typing import FloatArray
from ..interfaces.data import TimeSeries

if TYPE_CHECKING:
    from chronos import Chronos2Pipeline
    from timesfm import TimesFM_2p5_200M_torch

import numpy as np

if TYPE_CHECKING:
    from .. import Covariates, TimeSeries

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .. import _validation as v
from ..interfaces.base import Diagnostics, FittedForecaster
from ..interfaces.data import prepare, reject_covariates
from ..interfaces.result import ForecastResult
from .adapters import ChronosForecaster, TimesFMForecaster, _numpy


@dataclass(frozen=True)
class Checkpoint:
    """Resolved model repository, immutable revision and local artifact directory."""

    repository: str
    revision: str
    directory: str


def download_checkpoint(
    repository: str,
    *,
    revision: str | None = None,
    cache_dir: str | PathLike[str] | None = None,
    local_files_only: bool = False,
) -> Checkpoint:
    """Resolve a Hub revision into an immutable cached local checkpoint.

    No remote Python code is executed. The resolved commit is retained; offline
    reloads use the same directory rather than resolving a moving branch again.
    """
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repository,
        revision=revision,
        cache_dir=None if cache_dir is None else Path(cache_dir),
        local_files_only=local_files_only,
        allow_patterns=["*.json", "*.safetensors", "*.bin", "*.pt", "*.pth"],
    )
    return Checkpoint(repository, Path(path).name, str(Path(path).resolve()))


def load_pretrained(
    provider: Literal["chronos", "chronos_bolt", "chronos2", "timesfm3", "timesfm2.5"],
    *,
    repository: str | None = None,
    revision: str | None = None,
    cache_dir: str | PathLike[str] | None = None,
    local_files_only: bool = False,
    device: str = "cpu",
    **options: object,
) -> ChronosForecaster | Chronos2Forecaster | TimesFMForecaster | TimesFM25Forecaster:
    """Load a concrete provider model and wrap it in the forecasting contract."""
    defaults = {
        "chronos": "amazon/chronos-t5-tiny",
        "chronos_bolt": "amazon/chronos-bolt-tiny",
        "chronos2": "autogluon/chronos-2-small",
        "timesfm3": "google/timesfm-3.0-pytorch",
        "timesfm2.5": "google/timesfm-2.5-200m-pytorch",
    }
    v.choice(provider, tuple(defaults), "provider")
    checkpoint = download_checkpoint(
        repository or defaults[provider],
        revision=revision,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    identity = checkpoint.repository + "@" + checkpoint.revision
    if provider in ("chronos", "chronos_bolt", "chronos2"):
        import chronos

        factory = getattr(
            chronos,
            {
                "chronos": "ChronosPipeline",
                "chronos_bolt": "ChronosBoltPipeline",
                "chronos2": "Chronos2Pipeline",
            }[provider],
        )
        pipeline = factory.from_pretrained(checkpoint.directory, device_map=device)
        if provider == "chronos2":
            return cast(Callable[..., Chronos2Forecaster], Chronos2Forecaster)(
                pipeline, checkpoint=identity, **options
            )
        return cast(Callable[..., ChronosForecaster], ChronosForecaster)(
            pipeline, checkpoint=identity, **options
        )
    if provider == "timesfm3":
        from timesfm3.torch import TimesFM3Forecaster

        return TimesFMForecaster(
            TimesFM3Forecaster.from_pretrained(checkpoint.directory, device=device),
            checkpoint=identity,
        )
    import timesfm

    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(checkpoint.directory)
    import torch

    model.model.device = torch.device(device)
    model.model.device_count = 1
    model.model.to(device)
    options.setdefault("fix_quantile_crossing", True)
    options.setdefault("use_continuous_quantile_head", True)
    configuration = cast(Callable[..., timesfm.ForecastConfig], timesfm.ForecastConfig)
    model.compile(
        configuration(
            max_context=options.pop("max_context", 256),
            max_horizon=options.pop("max_horizon", 128),
            **options,
        )
    )
    return TimesFM25Forecaster(model, identity)


def chronos2_input(
    history, future: FloatArray | None = None, *, training: bool = False
):
    cov = history.covariates
    item: dict[str, Any] = {"target": np.array(history.values.T)}
    past, known = {}, {}
    if cov is not None:
        for name, prefix in [("known_future", "known"), ("observed_past", "past")]:
            values = getattr(cov, name)
            if values is not None:
                for j in range(values.shape[1]):
                    past[f"{prefix}{j}"] = np.array(values[:, j])
                    if name == "known_future":
                        if not training and future is None:
                            raise ValueError(
                                "future known covariates are required for prediction"
                            )
                        if training:
                            known[f"{prefix}{j}"] = None
                        else:
                            assert future is not None
                            known[f"{prefix}{j}"] = np.array(future[:, j])
    if past:
        item["past_covariates"] = past
    if known:
        item["future_covariates"] = known
    return [item]


@dataclass(frozen=True)
class Chronos2Forecaster:
    """Pretrained Chronos-2 specification with explicit marginal quantile levels and optional fine-tuning controls."""

    pipeline: Chronos2Pipeline
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
    checkpoint: str = "supplied-chronos2"
    finetune_steps: int = 0
    finetune_horizon: int = 6
    finetune_options: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "quantiles", tuple(v.levels(self.quantiles)))
        v.integer(self.finetune_steps, "finetune_steps")
        v.integer(self.finetune_horizon, "finetune_horizon", 1)
        reserved = {
            "inputs",
            "prediction_length",
            "num_steps",
        } & self.finetune_options.keys()
        if reserved:
            raise ValueError(
                "finetune_options cannot override training data or step controls"
            )

    def fit(
        self, data: TimeSeries | ArrayLike, *, covariates: Covariates | None = None
    ) -> FittedForecaster:
        history = prepare(data, covariates)
        reject_covariates(history, known=True, observed=True)
        pipeline = deepcopy(self.pipeline)
        if self.finetune_steps:
            options = dict(self.finetune_options)
            options.setdefault("batch_size", 8)
            options.setdefault("min_past", 4)
            options.setdefault("remove_printer_callback", True)
            options.setdefault("report_to", [])
            # Advanced trainer options are validated by the native provider.
            train = cast("Callable[..., Chronos2Pipeline]", pipeline.fit)
            pipeline = train(
                chronos2_input(history, training=True),
                prediction_length=self.finetune_horizon,
                num_steps=self.finetune_steps,
                **options,
            )
        return FittedForecaster(
            _Chronos2Adapter(pipeline, history, self),
            history,
            self,
            Diagnostics(
                "chronos2",
                message="Fine-tuned checkpoint"
                if self.finetune_steps
                else "Pretrained checkpoint",
            ),
            {"checkpoint": self.checkpoint, "finetune_steps": self.finetune_steps},
            "chronos2:" + self.checkpoint,
        )


@dataclass(frozen=True)
class _Chronos2Adapter:
    pipeline: Chronos2Pipeline
    history: TimeSeries
    spec: Chronos2Forecaster

    def forecast(self, steps, covariates, *, rng: np.random.Generator | None = None):
        quantiles, point = self.pipeline.predict_quantiles(
            chronos2_input(self.history, covariates),
            prediction_length=steps,
            quantile_levels=list(self.spec.quantiles),
        )
        q, p = _numpy(quantiles[0]), _numpy(point[0])
        k = self.history.dimension
        if q.shape != (k, steps, len(self.spec.quantiles)) or p.shape != (k, steps):
            raise ValueError("Chronos-2 returned unexpected target/horizon axes")
        return ForecastResult(
            p.T,
            quantile_levels=self.spec.quantiles,
            quantile_values=q.transpose(1, 0, 2),
        )


@dataclass(frozen=True)
class TimesFM25Forecaster:
    """Pretrained TimesFM 2.5 specification; its quantile head supplies marginal forecasts, not a joint path law."""

    model: TimesFM_2p5_200M_torch
    checkpoint: str = "supplied-timesfm2.5"

    def fit(
        self, data: TimeSeries | ArrayLike, *, covariates: Covariates | None = None
    ) -> FittedForecaster:
        history = prepare(data, covariates)
        reject_covariates(history)
        return FittedForecaster(
            _TimesFM25Adapter(deepcopy(self.model), history),
            history,
            self,
            Diagnostics("timesfm2.5"),
            {"checkpoint": self.checkpoint},
            "timesfm2.5:" + self.checkpoint,
        )


@dataclass(frozen=True)
class _TimesFM25Adapter:
    model: TimesFM_2p5_200M_torch
    history: TimeSeries

    def forecast(self, steps, covariates, *, rng: np.random.Generator | None = None):
        point, quantiles = self.model.forecast(
            horizon=steps, inputs=[np.array(col) for col in self.history.values.T]
        )
        point, quantiles = _numpy(point), _numpy(quantiles)
        # TimesFM 2.5's first quantile-output channel is its point prediction.
        if quantiles.shape != (self.history.dimension, steps, 10):
            raise ValueError(
                "TimesFM 2.5 requires its quantile head (point plus nine deciles)"
            )
        return ForecastResult(
            point.T,
            quantile_levels=np.arange(1, 10) / 10,
            quantile_values=quantiles[..., 1:].transpose(1, 0, 2),
        )
