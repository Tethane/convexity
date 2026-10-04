"""User engines, pretrained checkpoints, and optional sktime interoperability."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chronos import ChronosBoltPipeline, ChronosPipeline
    from sktime.forecasting.base import BaseForecaster
    from timesfm3.torch import TimesFM3Forecaster
from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike

from ..._typing import FloatArray

if TYPE_CHECKING:
    from .. import Covariates

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Protocol, cast

from .. import _validation as v
from ..interfaces.base import Diagnostics, FittedForecaster
from ..interfaces.data import TimeSeries, prepare, reject_covariates
from ..interfaces.result import ForecastResult


class ForecastFunction(Protocol):
    """Produce a checked result from history and H future known-covariate rows."""

    def __call__(
        self,
        history: TimeSeries,
        steps: int,
        future_known: FloatArray | None,
        /,
        *,
        rng: np.random.Generator | None,
    ) -> ForecastResult: ...


class PathSampler(Protocol):
    """Generate joint paths (size,H,target); rng is owned by the caller."""

    def __call__(
        self,
        history: TimeSeries,
        steps: int,
        future_known: FloatArray | None,
        /,
        *,
        size: int,
        rng: np.random.Generator,
    ) -> ArrayLike: ...


@dataclass(frozen=True)
class FoundationForecaster:
    """Wrap predictor(history, steps, future_known, rng=...) -> ForecastResult.

    The supplied engine owns preprocessing/checkpoints. fit snapshots context;
    it does not imply fine-tuning. No weights are downloaded by Convexity.
    """

    predictor: ForecastFunction
    identity: str = "pretrained"
    missing: str = "raise"

    def __post_init__(self) -> None:
        if not callable(self.predictor):
            raise TypeError("predictor must be callable")
        if not isinstance(self.identity, str) or not self.identity:
            raise ValueError("identity must be a nonempty string")
        v.choice(self.missing, ("raise", "ffill", "native"), "missing")

    def fit(
        self, data: TimeSeries | ArrayLike, *, covariates: Covariates | None = None
    ) -> FittedForecaster:
        history = prepare(data, covariates, missing=self.missing)
        return FittedForecaster(
            _CallableAdapter(deepcopy(self.predictor), history),
            history,
            self,
            Diagnostics("pretrained", message="Context only; supplied checkpoint"),
            model_identity=self.identity,
        )


@dataclass(frozen=True)
class _CallableAdapter:
    predictor: ForecastFunction
    history: TimeSeries

    def forecast(self, steps, covariates, *, rng: np.random.Generator | None = None):
        return self.predictor(self.history, steps, covariates, rng=rng)


@dataclass(frozen=True)
class GenerativeForecaster:
    """Wrap a trained diffusion/flow/VAE/autoregressive conditional path sampler.

    sampler(history, steps, future_known, size=..., rng=...) returns (size,H,k).
    An optional trainer(history) returns a fitted sampler during fit.
    """

    sampler: PathSampler | None = None
    trainer: Callable[[TimeSeries], PathSampler] | None = None
    samples: int = 1000
    identity: str = "generative"
    missing: str = "raise"

    def __post_init__(self) -> None:
        if (self.sampler is None) == (self.trainer is None):
            raise ValueError("supply exactly one of sampler or trainer")
        if not callable(self.sampler if self.trainer is None else self.trainer):
            raise TypeError("sampler/trainer must be callable")
        v.integer(self.samples, "samples", 1)
        v.choice(self.missing, ("raise", "ffill", "native"), "missing")

    def fit(
        self, data: TimeSeries | ArrayLike, *, covariates: Covariates | None = None
    ) -> FittedForecaster:
        history = prepare(data, covariates, missing=self.missing)
        sampler = (
            deepcopy(self.sampler) if self.trainer is None else self.trainer(history)
        )
        if not callable(sampler):
            raise TypeError("trainer must return a callable sampler")
        return FittedForecaster(
            _GenerativeAdapter(sampler, history, self.samples),
            history,
            self,
            Diagnostics("generative"),
            model_identity=self.identity,
        )


@dataclass(frozen=True)
class _GenerativeAdapter:
    sampler: PathSampler
    history: TimeSeries
    samples: int

    def forecast(self, steps, covariates, *, rng: np.random.Generator | None = None):
        rng = v.generator(rng)
        paths = self.sampler(
            self.history, steps, covariates, size=self.samples, rng=rng
        )
        out = v.array(paths, "generated paths", ndim=3)
        if out.shape != (self.samples, steps, self.history.dimension):
            raise ValueError("sampler must return (samples,steps,targets)")
        return ForecastResult.from_paths(out)


@dataclass(frozen=True)
class ChronosForecaster:
    """Original Chronos/Chronos-Bolt tensor APIs, with explicit output semantics.

    output=quantiles calls predict_quantiles. output=samples requires the
    original Chronos sampled-path API, not Bolt/Chronos-2 quantile tensors.
    """

    pipeline: ChronosPipeline | ChronosBoltPipeline
    output: str = "quantiles"
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
    samples: int = 100
    seed: int = 0
    checkpoint: str = "supplied-chronos-checkpoint"

    def __post_init__(self) -> None:
        v.choice(self.output, ("quantiles", "samples"), "output")
        object.__setattr__(self, "quantiles", tuple(v.levels(self.quantiles)))
        v.integer(self.samples, "samples", 1)
        v.seed(self.seed)
        kind = _chronos_kind(self.pipeline)
        if self.output == "samples" and kind != "samples":
            raise ValueError(
                "Chronos samples require a pipeline declaring forecast_type=SAMPLES"
            )
        if kind == "quantiles":
            available = getattr(self.pipeline, "quantiles", ())
            if not set(self.quantiles) <= set(available):
                raise ValueError(
                    "request only the quantile levels trained by the supplied Chronos pipeline"
                )

    def fit(
        self, data: TimeSeries | ArrayLike, *, covariates: Covariates | None = None
    ) -> FittedForecaster:
        history = prepare(data, covariates)
        reject_covariates(history)
        if history.dimension != 1:
            raise ValueError(
                "this Chronos adapter supports one target; use a custom predictor for Chronos-2"
            )
        adapter = _ChronosAdapter(deepcopy(self.pipeline), history, self)
        return FittedForecaster(
            adapter,
            history,
            self,
            Diagnostics("chronos", message="Supplied checkpoint; no fine-tuning"),
            {"checkpoint": self.checkpoint, "output": self.output},
            "chronos:" + self.checkpoint,
            self.seed,
        )


@dataclass(frozen=True)
class _ChronosAdapter:
    pipeline: ChronosPipeline | ChronosBoltPipeline
    history: TimeSeries
    spec: Any

    def forecast(self, steps, covariates, *, rng: np.random.Generator | None = None):
        import torch

        seed = self.spec.seed if rng is None else int(rng.integers(0, 2**32))
        devices = (
            list(range(torch.cuda.device_count()))
            if torch.cuda.is_initialized()
            else []
        )
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            torch.set_rng_state(torch.Generator().manual_seed(seed).get_state())
            for device in devices:
                torch.cuda.set_rng_state(
                    torch.Generator(device=f"cuda:{device}")
                    .manual_seed(seed)
                    .get_state(),
                    device,
                )
            context = torch.tensor(
                np.array(self.history.values[:, 0]), dtype=torch.float32
            )[None]
            if self.spec.output == "samples":
                raw = cast("ChronosPipeline", self.pipeline).predict(
                    context,
                    prediction_length=steps,
                    num_samples=self.spec.samples,
                    limit_prediction_length=True,
                )
                values = _numpy(raw)
                if values.shape != (1, self.spec.samples, steps):
                    raise ValueError(
                        "Chronos sample API must return (1,samples,horizon)"
                    )
                return ForecastResult.from_paths(values[0, :, :, None])
            options = (
                {"num_samples": self.spec.samples}
                if _chronos_kind(self.pipeline) == "samples"
                else {}
            )
            raw, mean = self.pipeline.predict_quantiles(
                context,
                prediction_length=steps,
                quantile_levels=list(self.spec.quantiles),
                limit_prediction_length=True,
                **options,
            )
            values, mean = _numpy(raw), _numpy(mean)
            if values.shape != (1, steps, len(self.spec.quantiles)) or mean.shape != (
                1,
                steps,
            ):
                raise ValueError(
                    "Chronos quantile API returned incompatible output shapes"
                )
            # Bolt calls its median a "mean" in this API. Retain it as a point
            # forecast; only the original sample engine supplies an estimated mean.
            mean_values = (
                mean[0, :, None] if _chronos_kind(self.pipeline) == "samples" else None
            )
            return ForecastResult(
                mean[0, :, None],
                mean_values=mean_values,
                quantile_levels=self.spec.quantiles,
                quantile_values=values[0, :, None, :],
            )


def _chronos_kind(pipeline):
    kind = getattr(pipeline, "forecast_type", None)
    return str(getattr(kind, "name", kind)).lower()


def _numpy(value):
    return np.asarray(
        value.detach().cpu().numpy() if hasattr(value, "detach") else value
    )


@dataclass(frozen=True)
class TimesFMForecaster:
    """TimesFM 3 predict_batch API with multivariate and dynamic covariates."""

    model: TimesFM3Forecaster
    checkpoint: str = "supplied-timesfm3-checkpoint"

    def fit(
        self, data: TimeSeries | ArrayLike, *, covariates: Covariates | None = None
    ) -> FittedForecaster:
        history = prepare(data, covariates)
        reject_covariates(history, known=True, observed=True)
        return FittedForecaster(
            _TimesFMAdapter(deepcopy(self.model), history),
            history,
            self,
            Diagnostics("timesfm3", message="Supplied checkpoint; no fine-tuning"),
            {"checkpoint": self.checkpoint},
            "timesfm3:" + self.checkpoint,
        )


@dataclass(frozen=True)
class _TimesFMAdapter:
    model: TimesFM3Forecaster
    history: TimeSeries

    def forecast(self, steps, covariates, *, rng: np.random.Generator | None = None):
        history = self.history
        context = history.values.T if history.dimension > 1 else history.values[:, 0]
        kwargs: dict[str, Any] = dict(
            horizon=steps, return_quantiles=True, use_symmetric_averaging=False
        )
        cov = history.covariates
        if cov is not None and cov.observed_past is not None:
            kwargs["past_only_covariates"] = [np.array(cov.observed_past.T)]
        if covariates is not None:
            assert (
                history.covariates is not None
                and history.covariates.known_future is not None
            )
            kwargs["past_future_covariates"] = [
                np.concatenate((history.covariates.known_future, covariates)).T
            ]
        outputs = list(self.model.predict_batch([np.array(context)], **kwargs))
        if len(outputs) != 1:
            raise ValueError("TimesFM must return one series forecast")
        point, quantiles = _numpy(outputs[0].forecast), _numpy(outputs[0].quantiles)
        expected = (steps,) if history.dimension == 1 else (history.dimension, steps)
        if point.shape != expected or quantiles.shape != expected + (9,):
            raise ValueError(
                "TimesFM 3 must return point and nine decile forecasts in target,time order"
            )
        point = point[:, None] if history.dimension == 1 else point.T
        quantiles = (
            quantiles[:, None, :]
            if history.dimension == 1
            else quantiles.transpose(1, 0, 2)
        )
        return ForecastResult(
            point, quantile_levels=np.arange(1, 10) / 10, quantile_values=quantiles
        )


@dataclass(frozen=True)
class SktimeForecaster:
    """Explicit sktime interoperability wrapper retaining checked forecast axes and supplied uncertainty semantics."""

    estimator: BaseForecaster
    max_horizon: int = 1
    quantiles: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        v.integer(self.max_horizon, "max_horizon", 1)
        if self.quantiles:
            object.__setattr__(self, "quantiles", tuple(v.levels(self.quantiles)))

    def fit(
        self, data: TimeSeries | ArrayLike, *, covariates: Covariates | None = None
    ) -> FittedForecaster:
        try:
            from importlib import import_module

            import pandas as pd

            ForecastingHorizon = import_module(
                "sktime.forecasting.base"
            ).ForecastingHorizon
        except ModuleNotFoundError as exc:
            raise ImportError(
                "sktime interoperability requires: pip install 'convexity[forecasting]' "
            ) from exc
        history = prepare(data, covariates)
        reject_covariates(history, known=True)
        model = self.estimator.clone()
        y = pd.DataFrame(history.values, columns=history.names)
        if history.dimension == 1:
            y = y.iloc[:, 0]
        cov = history.covariates
        x = (
            None
            if cov is None or cov.known_future is None
            else pd.DataFrame(cov.known_future)
        )
        model.fit(
            y=y,
            X=x,
            fh=ForecastingHorizon(np.arange(1, self.max_horizon + 1), is_relative=True),
        )
        return FittedForecaster(
            _SktimeAdapter(model, history, self),
            history,
            self,
            Diagnostics("sktime"),
            model_identity="sktime." + type(model).__name__,
        )


@dataclass(frozen=True)
class _SktimeAdapter:
    model: BaseForecaster
    history: TimeSeries
    spec: SktimeForecaster

    def forecast(self, steps, covariates, *, rng: np.random.Generator | None = None):
        from importlib import import_module

        import pandas as pd

        ForecastingHorizon = import_module("sktime.forecasting.base").ForecastingHorizon
        if steps > self.spec.max_horizon:
            raise ValueError("requested horizon exceeds the fitted sktime horizon")
        fh = ForecastingHorizon(np.arange(1, steps + 1), is_relative=True)
        x = (
            None
            if covariates is None
            else pd.DataFrame(
                covariates,
                index=np.arange(len(self.history), len(self.history) + steps),
            )
        )
        point = np.asarray(self.model.predict(fh=fh, X=x)).reshape(
            steps, self.history.dimension
        )
        if not self.spec.quantiles:
            return ForecastResult(point)
        raw = cast(
            "pd.DataFrame",
            self.model.predict_quantiles(fh=fh, X=x, alpha=list(self.spec.quantiles)),
        )
        if not isinstance(raw, pd.DataFrame):
            raise TypeError("sktime quantiles must be a pandas DataFrame")
        columns = [
            (name, level)
            for name in self.history.names
            for level in self.spec.quantiles
        ]
        # Univariate sktime adapters may rename the single series to 0.
        if self.history.dimension == 1:
            columns = [(raw.columns[0][0], level) for level in self.spec.quantiles]
        q = (
            raw.loc[:, columns]
            .to_numpy()
            .reshape(point.shape + (len(self.spec.quantiles),))
        )
        return ForecastResult(
            point, quantile_levels=self.spec.quantiles, quantile_values=q
        )
