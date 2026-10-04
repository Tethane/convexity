"""Optional sequence learning with user modules and explicit probabilistic heads."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from ..._typing import FloatArray

if TYPE_CHECKING:
    from .. import Covariates, TimeSeries

from typing import Protocol

from ..interfaces.result import ForecastResult

if TYPE_CHECKING:
    from torch import Tensor
    from torch.nn import Module

from dataclasses import dataclass

from .. import _validation as v


class ForecastDecoder(Protocol):
    """Decode a neural head without inventing uncertainty absent from the head."""

    def __call__(
        self,
        raw: Tensor,
        location: FloatArray,
        scale: FloatArray,
        *,
        rng: np.random.Generator | None,
    ) -> ForecastResult: ...


@dataclass(frozen=True)
class SequenceSchema:
    """Ordered lag/covariate feature layout for converting a time series into supervised rows."""

    context_length: int
    horizon: int
    targets: int
    known: int
    observed: int
    static: int
    channels: int


@dataclass(frozen=True)
class TorchForecaster:
    """A module factory receives SequenceSchema and returns module(batch).

    batch contains history, mask, past_known, past_observed, future_known, static.
    Output shape is (batch,horizon,targets,channels). Custom heads supply loss
    and decoder callbacks; see the forecasting guide for their contracts.
    """

    module_factory: Callable[[SequenceSchema], Module] | None = None
    architecture: str = "mlp"
    context_length: int = 12
    max_horizon: int = 1
    hidden_size: int = 32
    head: str = "point"
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    validation_size: int = 0
    patience: int = 10
    seed: int = 0
    device: str = "cpu"
    missing: str = "raise"
    loss: Callable[[Tensor, Tensor, Tensor], Tensor] | None = None
    decoder: ForecastDecoder | None = None
    output_channels: int = 1
    mixture_components: int = 3
    category_values: tuple[float, ...] = (0.0, 1.0)
    discretize: bool = False

    def __post_init__(self) -> None:
        v.choice(
            self.architecture,
            ("mlp", "cnn", "rnn", "gru", "lstm", "transformer"),
            "architecture",
        )
        v.choice(
            self.head,
            (
                "point",
                "gaussian",
                "student_t",
                "lognormal",
                "mixture",
                "categorical",
                "quantile",
                "custom",
            ),
            "head",
        )
        v.choice(self.missing, ("raise", "ffill", "mask"), "missing")
        for name in (
            "context_length",
            "max_horizon",
            "hidden_size",
            "epochs",
            "batch_size",
            "patience",
            "output_channels",
        ):
            v.integer(getattr(self, name), name, 1)
        v.integer(self.validation_size, "validation_size")
        v.seed(self.seed)
        v.integer(self.mixture_components, "mixture_components", 1)
        v.boolean(self.discretize, "discretize")
        values = v.array(self.category_values, "category_values", ndim=1)
        if len(values) < 2 or any(b <= a for a, b in zip(values, values[1:])):
            raise ValueError(
                "category_values must be strictly increasing with at least two values"
            )
        object.__setattr__(self, "category_values", tuple(values))
        v.scalar(self.learning_rate, "learning_rate", minimum=0, strict=True)
        object.__setattr__(self, "quantiles", tuple(v.levels(self.quantiles)))
        if self.module_factory is not None and not callable(self.module_factory):
            raise TypeError("module_factory must be callable")
        if self.head == "custom" and (
            not callable(self.loss) or not callable(self.decoder)
        ):
            raise ValueError("custom heads require loss and decoder callbacks")
        if self.head != "custom" and (
            self.loss is not None or self.decoder is not None
        ):
            raise ValueError("loss and decoder callbacks require head=custom")

    def fit(
        self, data: TimeSeries | ArrayLike, *, covariates: Covariates | None = None
    ):
        try:
            from .._backends.torch import train
        except ModuleNotFoundError as exc:
            if exc.name != "torch":
                raise
            raise ImportError(
                "Neural forecasting requires: pip install 'convexity[neural]'"
            ) from exc
        return train(self, data, covariates)
