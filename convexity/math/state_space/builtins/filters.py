"""Backend selection with explicit inference capabilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ...forecasting import TimeSeries
from .. import _validation as v


@dataclass(frozen=True)
class KalmanFilter:
    """Exact Gaussian filtering for a compatible linear Gaussian state-space model."""

    def filter(self, model, observations: TimeSeries | ArrayLike):
        from .._backends.linear import run

        return run(model, observations, smooth=False)

    def smooth(self, model, observations: TimeSeries | ArrayLike):
        from .._backends.linear import run

        return run(model, observations, smooth=True)


@dataclass(frozen=True)
class ExtendedKalmanFilter:
    """Gaussian state approximation using local derivatives of nonlinear transition and observation maps."""

    def filter(self, model, observations: TimeSeries | ArrayLike):
        from .._backends.nonlinear import run

        return run(model, observations, method="extended", smooth=False)

    def smooth(self, model, observations: TimeSeries | ArrayLike):
        from .._backends.nonlinear import run

        return run(model, observations, method="extended", smooth=True)


@dataclass(frozen=True)
class UnscentedKalmanFilter:
    """Gaussian state approximation using a configured sigma-point rule for nonlinear maps."""

    alpha: float = 0.5
    beta: float = 2.0
    kappa: float = 0.0

    def __post_init__(self) -> None:
        v.scalar(self.alpha, "alpha", minimum=0, strict=True)
        v.scalar(self.beta, "beta", minimum=0)
        v.scalar(self.kappa, "kappa")

    def filter(self, model, observations: TimeSeries | ArrayLike):
        from .._backends.nonlinear import run

        return run(
            model,
            observations,
            method="unscented",
            smooth=False,
            alpha=self.alpha,
            beta=self.beta,
            kappa=self.kappa,
        )

    def smooth(self, model, observations: TimeSeries | ArrayLike):
        from .._backends.nonlinear import run

        return run(
            model,
            observations,
            method="unscented",
            smooth=True,
            alpha=self.alpha,
            beta=self.beta,
            kappa=self.kappa,
        )


@dataclass(frozen=True)
class ParticleFilter:
    """Monte Carlo state filtering with explicit particle count, resampling controls and caller RNG."""

    particles: int = 1000
    resampling: str = "systematic"
    ess_threshold: float = 0.5

    def __post_init__(self) -> None:
        v.integer(self.particles, "particles", 2)
        v.choice(
            self.resampling,
            ("systematic", "stratified", "multinomial", "residual"),
            "resampling",
        )
        threshold = v.scalar(self.ess_threshold, "ess_threshold", minimum=0)
        if threshold > 1:
            raise ValueError("ess_threshold must lie in [0,1]")

    def filter(
        self, model, observations: TimeSeries | ArrayLike, *, rng: np.random.Generator
    ):
        from .._backends.particle import run

        return run(model, observations, self, rng=rng)

    def smooth(
        self,
        model,
        observations: TimeSeries | ArrayLike,
        *,
        rng: np.random.Generator,
        paths: int = 100,
        method: str = "backward",
    ):
        """Filter then sample smoothing paths; backward FFBS or ancestral tracing."""
        from .._backends.particle import smooth_result

        v.integer(paths, "paths", 1)
        v.choice(method, ("backward", "ancestral"), "smoothing method")
        return smooth_result(
            self.filter(model, observations, rng=rng), paths, method, rng=rng
        )

    def smooth_result(
        self,
        result,
        *,
        rng: np.random.Generator,
        paths: int = 100,
        method: str = "backward",
    ):
        """Reuse a completed particle filter without resampling its forward pass."""
        from .._backends.particle import smooth_result

        return smooth_result(result, paths, method, rng=rng)
