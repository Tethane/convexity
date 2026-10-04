"""Conditional generative models trained on causal forecast windows."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from .. import Covariates, TimeSeries

from dataclasses import dataclass
from typing import TypedDict, Unpack

from .. import _validation as v


class FlowforecasterOptions(TypedDict, total=False):
    """Optional keyword settings for FlowForecaster."""

    context_length: int
    max_horizon: int
    hidden_size: int
    latent_size: int
    layers: int
    epochs: int
    batch_size: int
    learning_rate: float
    samples: int
    seed: int
    device: str
    missing: str
    kl_weight: float
    noise_min: float
    noise_max: float
    noise_levels: int
    langevin_steps: int
    langevin_rate: float


class VaeforecasterOptions(TypedDict, total=False):
    """Optional keyword settings for VAEForecaster."""

    context_length: int
    max_horizon: int
    hidden_size: int
    latent_size: int
    layers: int
    epochs: int
    batch_size: int
    learning_rate: float
    samples: int
    seed: int
    device: str
    missing: str
    kl_weight: float
    noise_min: float
    noise_max: float
    noise_levels: int
    langevin_steps: int
    langevin_rate: float


class AutoregressiveforecasterOptions(TypedDict, total=False):
    """Optional keyword settings for AutoregressiveForecaster."""

    context_length: int
    max_horizon: int
    hidden_size: int
    latent_size: int
    layers: int
    epochs: int
    batch_size: int
    learning_rate: float
    samples: int
    seed: int
    device: str
    missing: str
    kl_weight: float
    noise_min: float
    noise_max: float
    noise_levels: int
    langevin_steps: int
    langevin_rate: float


class ScoreforecasterOptions(TypedDict, total=False):
    """Optional keyword settings for ScoreForecaster."""

    context_length: int
    max_horizon: int
    hidden_size: int
    latent_size: int
    layers: int
    epochs: int
    batch_size: int
    learning_rate: float
    samples: int
    seed: int
    device: str
    missing: str
    kl_weight: float
    noise_min: float
    noise_max: float
    noise_levels: int
    langevin_steps: int
    langevin_rate: float


@dataclass(frozen=True)
class DiffusionForecaster:
    """Small conditional denoising network; a working research baseline.

    Gaussian forward noise, velocity prediction and the DDPM posterior-variance
    reverse process generate joint paths. Velocity avoids dividing network
    errors by near-zero terminal signal. Parameters/checkpoints are stored in
    the fitted model; a finite training budget does not establish calibration.
    """

    context_length: int = 12
    max_horizon: int = 3
    hidden_size: int = 64
    diffusion_steps: int = 50
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    samples: int = 500
    seed: int = 0
    device: str = "cpu"
    missing: str = "raise"

    def __post_init__(self) -> None:
        for name in (
            "context_length",
            "max_horizon",
            "hidden_size",
            "diffusion_steps",
            "epochs",
            "batch_size",
            "samples",
        ):
            v.integer(getattr(self, name), name, 1)
        if self.diffusion_steps < 2:
            raise ValueError("diffusion_steps must be at least two")
        v.scalar(self.learning_rate, "learning_rate", minimum=0, strict=True)
        v.seed(self.seed)
        v.choice(self.missing, ("raise", "ffill"), "missing")

    def fit(
        self, data: TimeSeries | ArrayLike, *, covariates: Covariates | None = None
    ):
        try:
            from .._backends.diffusion import train_diffusion
        except ModuleNotFoundError as exc:
            if exc.name != "torch":
                raise
            raise ImportError(
                "Diffusion forecasting requires: pip install 'convexity[neural]'"
            ) from exc
        return train_diffusion(self, data, covariates)


@dataclass(frozen=True)
class ConditionalGenerativeForecaster:
    """Train an actual conditional flow, VAE, autoregressive law, or score model."""

    model: str = "flow"
    context_length: int = 12
    max_horizon: int = 3
    hidden_size: int = 64
    latent_size: int = 8
    layers: int = 4
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    samples: int = 500
    seed: int = 0
    device: str = "cpu"
    missing: str = "raise"
    kl_weight: float = 1.0
    noise_min: float = 0.01
    noise_max: float = 5.0
    noise_levels: int = 20
    langevin_steps: int = 10
    langevin_rate: float = 2e-5

    def __post_init__(self) -> None:
        v.choice(self.model, ("flow", "vae", "autoregressive", "score"), "model")
        for name in (
            "context_length",
            "max_horizon",
            "hidden_size",
            "latent_size",
            "layers",
            "epochs",
            "batch_size",
            "samples",
            "noise_levels",
            "langevin_steps",
        ):
            v.integer(getattr(self, name), name, 1)
        for name in ("learning_rate", "noise_min", "noise_max", "langevin_rate"):
            v.scalar(getattr(self, name), name, minimum=0, strict=True)
        v.scalar(self.kl_weight, "kl_weight", minimum=0)
        if self.noise_min >= self.noise_max:
            raise ValueError("noise_min must be below noise_max")
        v.seed(self.seed)
        v.choice(self.missing, ("raise", "ffill"), "missing")

    def fit(
        self, data: TimeSeries | ArrayLike, *, covariates: Covariates | None = None
    ):
        from .._backends.generative import train_generative

        return train_generative(self, data, covariates)


def FlowForecaster(
    **kwargs: Unpack[FlowforecasterOptions],
) -> ConditionalGenerativeForecaster:
    """Construct the Flow forecasting specification.

    Train an actual conditional flow, VAE, autoregressive law, or score model.
    Fit with forecasting.fit(TimeSeries(values), specification). Forecast rows
    are horizons and columns are targets; variances use squared target units.
    Optional settings: context_length, max_horizon, hidden_size, latent_size, layers, epochs, batch_size, learning_rate, samples, seed, device, missing, kl_weight, noise_min, noise_max, noise_levels, langevin_steps, langevin_rate."""
    return ConditionalGenerativeForecaster(model="flow", **kwargs)


def VAEForecaster(
    **kwargs: Unpack[VaeforecasterOptions],
) -> ConditionalGenerativeForecaster:
    """Construct the VAE forecasting specification.

    Train an actual conditional flow, VAE, autoregressive law, or score model.
    Fit with forecasting.fit(TimeSeries(values), specification). Forecast rows
    are horizons and columns are targets; variances use squared target units.
    Optional settings: context_length, max_horizon, hidden_size, latent_size, layers, epochs, batch_size, learning_rate, samples, seed, device, missing, kl_weight, noise_min, noise_max, noise_levels, langevin_steps, langevin_rate."""
    return ConditionalGenerativeForecaster(model="vae", **kwargs)


def AutoregressiveForecaster(
    **kwargs: Unpack[AutoregressiveforecasterOptions],
) -> ConditionalGenerativeForecaster:
    """Construct the Autoregressive forecasting specification.

    Train an actual conditional flow, VAE, autoregressive law, or score model.
    Fit with forecasting.fit(TimeSeries(values), specification). Forecast rows
    are horizons and columns are targets; variances use squared target units.
    Optional settings: context_length, max_horizon, hidden_size, latent_size, layers, epochs, batch_size, learning_rate, samples, seed, device, missing, kl_weight, noise_min, noise_max, noise_levels, langevin_steps, langevin_rate."""
    return ConditionalGenerativeForecaster(model="autoregressive", **kwargs)


def ScoreForecaster(
    **kwargs: Unpack[ScoreforecasterOptions],
) -> ConditionalGenerativeForecaster:
    """Construct the Score forecasting specification.

    Train an actual conditional flow, VAE, autoregressive law, or score model.
    Fit with forecasting.fit(TimeSeries(values), specification). Forecast rows
    are horizons and columns are targets; variances use squared target units.
    Optional settings: context_length, max_horizon, hidden_size, latent_size, layers, epochs, batch_size, learning_rate, samples, seed, device, missing, kl_weight, noise_min, noise_max, noise_levels, langevin_steps, langevin_rate."""
    return ConditionalGenerativeForecaster(model="score", **kwargs)
