"""Structural extension points: no registration or inheritance required."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike

from ..._typing import FloatArray
from ...forecasting import TimeSeries
from .. import _validation as v
from .results import RegimeResult


class RegimeInference(Protocol):
    """Infer latent-regime probabilities under the supplied model, retaining explicit regime order."""

    def filter(self, data: TimeSeries | ArrayLike) -> RegimeResult: ...


def infer_regimes(data: TimeSeries | ArrayLike, model: RegimeInference) -> RegimeResult:
    """Filter observations through a RegimeInference implementation.

    Rows are time; return state probabilities and predictive log likelihoods.
    Smoothed probabilities, when available, condition on the complete sample."""
    result = model.filter(data)
    if not isinstance(result, RegimeResult):
        raise TypeError("filter must return a RegimeResult")
    return result


class Intensity(Protocol):
    """Deterministic rate per time unit and its exact/numerical integral."""

    def intensity(self, time: float) -> float: ...
    def integrated(self, start: float, end: float) -> float: ...


class EventProcess(Protocol):
    """Draw sorted, strictly increasing event times on (start,end]."""

    def sample(
        self, start: float, end: float, *, rng: np.random.Generator
    ) -> ArrayLike: ...


def sample_events(
    process: EventProcess, start: float, end: float, *, rng: np.random.Generator
) -> FloatArray:
    """Sample ordered event times in (start,end] with an explicit NumPy Generator.

    Return finite times in the same units as the process intensity; seeded generators
    make stochastic calls reproducible."""
    start, end = v.interval(start, end)
    v.generator(rng)
    result = v.times(process.sample(start, end, rng=rng))
    if (result <= start).any() or (result > end).any():
        raise ValueError("process returned times outside (start,end]")
    return result
