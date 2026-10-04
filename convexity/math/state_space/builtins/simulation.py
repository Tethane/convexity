"""Generated states and observations using the same model kernels as inference."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from .. import StateSpaceModel


import numpy as np

from ...forecasting.interfaces import TimeSeries
from .. import _validation as v
from ..interfaces.results import SimulationResult


def simulate(
    model: StateSpaceModel, timestamps: ArrayLike, *, rng: np.random.Generator
) -> SimulationResult:
    """One joint path with known initial time; no burn-in or implicit transition."""
    v.generator(rng)
    times = np.asarray(timestamps)
    if times.ndim != 1 or not len(times):
        raise ValueError("timestamps must be a nonempty vector")
    series = TimeSeries(np.zeros((len(times), model.observation_dimension)), times)
    steps = model.steps(series)
    state = model.initial_distribution().sample(1, rng=rng)
    states, observations = [], []
    for step in steps:
        if step.index:
            state = model.transition.sample(state, step, rng=rng)
        state = v.shaped(state, (1, model.state_dimension), "sampled state")
        observation = v.shaped(
            model.observation.sample(state, step, rng=rng),
            (1, model.observation_dimension),
            "sampled observation",
        )
        states.append(state[0])
        observations.append(observation[0])
    return SimulationResult(TimeSeries(observations, times), states)
