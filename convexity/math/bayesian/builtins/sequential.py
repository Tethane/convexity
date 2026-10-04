"""Sequential joint inference with static parameter particles."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ...forecasting import TimeSeries
from .. import _validation as v
from ..interfaces.sequential import JointFilterResult, StateParameterModel


@dataclass(frozen=True)
class JointParticleFilter:
    """Bootstrap SMC on (theta,x), with theta constant along each path.

    No parameter jitter or artificial dynamics. Static parameter diversity can
    collapse over long series: inspect ESS and surviving distinct parameters.
    This bounded adapter is not SMC² or particle MCMC rejuvenation.
    """

    particles: int = 2000
    ess_threshold: float = 0.5
    resampling: str = "systematic"

    def __post_init__(self) -> None:
        v.integer(self.particles, "particles", 2)
        v.probability(self.ess_threshold, "ess_threshold", closed=True)
        v.choice(
            self.resampling,
            ("systematic", "stratified", "multinomial", "residual"),
            "resampling",
        )

    def filter(
        self,
        model: StateParameterModel,
        observations: TimeSeries | ArrayLike,
        *,
        rng: np.random.Generator,
    ):
        from .._backends.sequential import run

        return run(model, observations, self, rng=rng)

    def smooth(
        self,
        model: StateParameterModel,
        observations: TimeSeries | ArrayLike,
        *,
        paths: int = 100,
        rng: np.random.Generator,
    ) -> JointFilterResult:
        from ...state_space import ParticleFilter

        v.integer(paths, "paths", 1)
        result = self.filter(model, observations, rng=rng)
        smoothed = ParticleFilter().smooth_result(
            result.filtering, paths=paths, method="ancestral", rng=rng
        )
        return JointFilterResult(smoothed, result.parameter_names)
