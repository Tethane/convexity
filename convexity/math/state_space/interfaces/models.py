"""Structural transition/observation contracts; no backend inheritance required."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from ..._typing import FloatArray

if TYPE_CHECKING:
    from ...distributions import Distribution
    from ...forecasting import TimeSeries
    from .results import FilterResult, GaussianState
from dataclasses import dataclass
from typing import Protocol

from .. import _validation as v


@dataclass(frozen=True)
class TimeStep:
    """Destination row, timestamp, and elapsed duration from the previous row.

    The initial law lives at row zero, where dt=0 and no transition is applied.
    Discrete models use dt=1 thereafter. Numeric elapsed units are user-defined.
    """

    index: int
    time: float | int | np.datetime64
    dt: float

    def __post_init__(self) -> None:
        v.integer(self.index, "step index")
        v.scalar(self.dt, "step dt", minimum=0)
        time = np.asarray(self.time)
        if time.ndim != 0 or time.dtype.kind not in "iufM" or not np.isfinite(time):
            raise ValueError("step time must be a finite numeric or datetime scalar")


class TransitionModel(Protocol):
    """Latent-state transition mechanism with explicit state dimension and time-step convention."""
    @property
    def dimension(self) -> int: ...

    def sample(
        self, states: ArrayLike, step: TimeStep, *, rng: np.random.Generator
    ) -> FloatArray: ...


class TransitionDensityModel(TransitionModel, Protocol):
    """Optional density capability required by backward particle smoothing."""

    def log_density(
        self, values: ArrayLike, states: ArrayLike, step: TimeStep
    ) -> FloatArray: ...


class ObservationModel(Protocol):
    """State-conditioned observation mechanism with an explicit observed dimension and time-step convention."""
    @property
    def state_dimension(self) -> int: ...

    @property
    def dimension(self) -> int: ...

    def sample(
        self, states: ArrayLike, step: TimeStep, *, rng: np.random.Generator
    ) -> FloatArray: ...

    def log_density(
        self, value: ArrayLike, states: ArrayLike, step: TimeStep
    ) -> FloatArray:
        """One log likelihood per state, marginalizing NaN observation coordinates."""
        ...


@dataclass(frozen=True)
class StateSpaceModel:
    """Initial vector law and independent Markov transition/observation kernels.

    Gaussian algorithms additionally require the built-in Gaussian kernels (or
    structural equivalents with mean, covariance, and, for EKF, jacobian).
    Covariates may be captured explicitly by time-indexed kernel callbacks.
    """

    initial: GaussianState | Distribution
    transition: TransitionModel
    observation: ObservationModel
    time_mode: str = "discrete"
    time_unit: np.timedelta64 | None = None
    state_names: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        from ...distributions import Distribution
        from .results import GaussianState

        if not isinstance(self.initial, (GaussianState, Distribution)):
            raise TypeError("initial must be GaussianState or a vector Distribution")
        shape = (
            self.initial.mean.shape
            if isinstance(self.initial, GaussianState)
            else self.initial.event_shape
        )
        if len(shape) != 1:
            raise ValueError(
                "initial must be a vector law, including one-dimensional states"
            )
        d = shape[0]
        v.integer(self.transition.dimension, "transition dimension", 1)
        v.integer(self.observation.dimension, "observation dimension", 1)
        if self.transition.dimension != d or self.observation.state_dimension != d:
            raise ValueError(
                "initial, transition, and observation state dimensions must agree"
            )
        for kernel in (self.transition, self.observation):
            if not callable(kernel.sample):
                raise TypeError("kernels must implement sample")
        if not callable(self.observation.log_density):
            raise TypeError("observation must implement log_density")
        v.choice(self.time_mode, ("discrete", "elapsed"), "time_mode")
        if self.time_mode == "discrete" and self.time_unit is not None:
            raise ValueError("time_unit applies only to elapsed models")
        names = (
            tuple(f"x{i}" for i in range(d))
            if self.state_names is None
            else tuple(self.state_names)
        )
        if (
            len(names) != d
            or len(set(names)) != d
            or any(not isinstance(n, str) or not n for n in names)
        ):
            raise ValueError(
                "state_names must contain one unique nonempty name per state"
            )
        object.__setattr__(self, "state_names", names)

    @property
    def state_dimension(self):
        return self.transition.dimension

    @property
    def observation_dimension(self):
        return self.observation.dimension

    def steps(self, series: TimeSeries):
        durations = v.elapsed(series.timestamps, self.time_mode, self.time_unit)
        return tuple(
            TimeStep(i, t, float(dt))
            for i, (t, dt) in enumerate(zip(series.timestamps, durations))
        )

    def initial_distribution(self):
        from .results import GaussianState

        return (
            self.initial.distribution()
            if isinstance(self.initial, GaussianState)
            else self.initial
        )


class StateEstimator[**P](Protocol):
    """Estimator whose additional arguments are retained by ``filter_states``."""

    def filter(
        self,
        model: StateSpaceModel,
        observations: TimeSeries | ArrayLike,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> FilterResult: ...


def filter_states[**P](
    model: StateSpaceModel,
    observations: TimeSeries | ArrayLike,
    estimator: StateEstimator[P],
    *args: P.args,
    **kwargs: P.kwargs,
) -> FilterResult:
    """Run a built-in or user estimator and check the shared result contract."""
    from .results import FilterResult

    series = v.data(observations, model.observation_dimension)
    result = estimator.filter(model, series, *args, **kwargs)
    if not isinstance(result, FilterResult):
        raise TypeError("estimator must return FilterResult")
    if result.filtered.mean.shape != (len(series), model.state_dimension):
        raise ValueError("filtered states must align with data and model dimensions")
    if not np.array_equal(
        result.observations.timestamps, series.timestamps
    ) or not np.array_equal(result.observations.values, series.values, equal_nan=True):
        raise ValueError("result observations must match the supplied series")
    return result
