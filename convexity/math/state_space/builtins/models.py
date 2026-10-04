"""Composable linear, additive Gaussian, and arbitrary distribution kernels."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from numpy.typing import ArrayLike

from ..._typing import FloatArray

if TYPE_CHECKING:
    from .. import TimeStep


from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import cast

import numpy as np

if TYPE_CHECKING:
    from ...distributions import Distribution
from .. import _validation as v
from ..interfaces.models import StateSpaceModel, TimeStep
from ..interfaces.results import GaussianState


def _log_gaussian(residuals, covariance):
    """Positive-definite densities only: degenerate measurements need constraints."""
    from scipy.linalg import cho_factor, cho_solve

    covariance = v.covariance(covariance, residuals.shape[-1], positive=True)
    factor = cho_factor(covariance, lower=True)
    quadratic = np.sum(residuals * cho_solve(factor, residuals.T).T, axis=-1)
    return -0.5 * (
        residuals.shape[-1] * np.log(2 * np.pi)
        + 2 * np.log(np.diag(factor[0])).sum()
        + quadratic
    )


class _GaussianKernel(ABC):
    """Vectorized outer batches, checked one-state user functions."""

    dimension: int

    @property
    @abstractmethod
    def input_dimension(self): ...

    @abstractmethod
    def mean(self, state, step) -> FloatArray: ...

    @abstractmethod
    def covariance(self, step) -> FloatArray: ...

    def _means(self, states, step):
        return np.array([self.mean(row, step) for row in states])

    def distribution(self, state, step):
        return GaussianState(
            self.mean(state, step), self.covariance(step)
        ).distribution()

    def sample(self, states: ArrayLike, step, *, rng: np.random.Generator):
        v.generator(rng)
        x = v.states(states, self.input_dimension)
        means = self._means(x, step)
        from ..._backends.gaussian import sample

        noise = sample(np.zeros(self.dimension), self.covariance(step), len(x), rng)
        return means + noise

    def log_density(self, values: ArrayLike, states: ArrayLike, step):
        x = v.states(states, self.input_dimension)
        means = self._means(x, step)
        values = v.array(values, "values")
        if values.shape not in ((self.dimension,), means.shape):
            raise ValueError("values must be one vector or one vector per state")
        return _log_gaussian(values - means, self.covariance(step))


class _GaussianObservation(_GaussianKernel):
    state_dimension: int

    @property
    def input_dimension(self):
        return self.state_dimension

    def log_density(self, value, states: ArrayLike, step):
        x = v.states(states, self.state_dimension)
        value = np.asarray(value, dtype=float)
        if value.shape != (self.dimension,) or np.any(np.isinf(value)):
            raise ValueError(
                "observation must be a vector with finite or NaN coordinates"
            )
        keep = np.flatnonzero(~np.isnan(value))
        if not len(keep):
            return np.zeros(len(x))
        means = self._means(x, step)
        covariance = self.covariance(step)[np.ix_(keep, keep)]
        return _log_gaussian(value[keep] - means[:, keep], covariance)


@dataclass(frozen=True)
class LinearGaussianTransition(_GaussianKernel):
    """x[t] = matrix(step) @ x[t-1] + offset(step) + N(0,noise(step))."""

    matrix: FloatArray | Callable[[TimeStep], ArrayLike]
    noise: FloatArray | Callable[[TimeStep], ArrayLike]
    offset: FloatArray | Callable[[TimeStep], ArrayLike] = cast(
        "FloatArray | Callable[[TimeStep], ArrayLike]", None
    )
    dimension: int = cast("int", None)

    def __post_init__(self) -> None:
        d = self.dimension
        if d is None:
            if callable(self.matrix):
                raise ValueError("time-dependent matrices require dimension")
            raw = v.array(self.matrix, "matrix", ndim=2)
            d = len(raw)
        d = v.integer(d, "dimension", 1)
        object.__setattr__(self, "dimension", d)
        for name, shape, cov in (
            ("matrix", (d, d), False),
            ("noise", (d, d), True),
            ("offset", (d,), False),
        ):
            value = (
                np.zeros(d)
                if name == "offset" and self.offset is None
                else getattr(self, name)
            )
            object.__setattr__(self, name, v.parameter(value, shape, name, cov=cov))

    @property
    def input_dimension(self):
        return self.dimension

    def coefficients(self, step: TimeStep):
        d = self.dimension
        return (
            v.evaluate(self.matrix, step, (d, d), "matrix"),
            v.evaluate(self.offset, step, (d,), "offset"),
        )

    def mean(self, state: ArrayLike, step: TimeStep):
        matrix, offset = self.coefficients(step)
        return matrix @ v.shaped(state, (self.dimension,), "state") + offset

    def _means(self, states, step):
        matrix, offset = self.coefficients(step)
        return states @ matrix.T + offset

    def covariance(self, step: TimeStep):
        return v.evaluate(
            self.noise,
            step,
            (self.dimension, self.dimension),
            "transition noise",
            cov=True,
        )

    def jacobian(self, state, step: TimeStep):
        return self.coefficients(step)[0]

    def __init__(
        self,
        matrix: ArrayLike | Callable[[TimeStep], ArrayLike],
        noise: ArrayLike | Callable[[TimeStep], ArrayLike],
        offset: ArrayLike | Callable[[TimeStep], ArrayLike] | None = None,
        dimension: int | None = None,
    ) -> None:
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "noise", noise)
        object.__setattr__(self, "offset", offset)
        object.__setattr__(self, "dimension", dimension)
        self.__post_init__()


@dataclass(frozen=True)
class LinearGaussianObservation(_GaussianObservation):
    """y[t] = matrix(step) @ x[t] + offset(step) + N(0,noise(step))."""

    matrix: FloatArray | Callable[[TimeStep], ArrayLike]
    noise: FloatArray | Callable[[TimeStep], ArrayLike]
    offset: FloatArray | Callable[[TimeStep], ArrayLike] = cast(
        "FloatArray | Callable[[TimeStep], ArrayLike]", None
    )
    state_dimension: int = cast("int", None)
    dimension: int = cast("int", None)

    def __post_init__(self) -> None:
        k, d = self.dimension, self.state_dimension
        if k is None or d is None:
            if callable(self.matrix):
                raise ValueError(
                    "time-dependent matrices require dimension and state_dimension"
                )
            raw = v.array(self.matrix, "matrix", ndim=2)
            k = raw.shape[0] if k is None else k
            d = raw.shape[1] if d is None else d
        k, d = v.integer(k, "dimension", 1), v.integer(d, "state_dimension", 1)
        object.__setattr__(self, "dimension", k)
        object.__setattr__(self, "state_dimension", d)
        for name, shape, cov in (
            ("matrix", (k, d), False),
            ("noise", (k, k), True),
            ("offset", (k,), False),
        ):
            value = (
                np.zeros(k)
                if name == "offset" and self.offset is None
                else getattr(self, name)
            )
            object.__setattr__(self, name, v.parameter(value, shape, name, cov=cov))

    def coefficients(self, step: TimeStep):
        return (
            v.evaluate(
                self.matrix, step, (self.dimension, self.state_dimension), "matrix"
            ),
            v.evaluate(self.offset, step, (self.dimension,), "offset"),
        )

    def mean(self, state: ArrayLike, step: TimeStep):
        matrix, offset = self.coefficients(step)
        return matrix @ v.shaped(state, (self.state_dimension,), "state") + offset

    def _means(self, states, step):
        matrix, offset = self.coefficients(step)
        return states @ matrix.T + offset

    def covariance(self, step: TimeStep):
        return v.evaluate(
            self.noise,
            step,
            (self.dimension, self.dimension),
            "observation noise",
            cov=True,
        )

    def jacobian(self, state, step: TimeStep):
        return self.coefficients(step)[0]

    def __init__(
        self,
        matrix: ArrayLike | Callable[[TimeStep], ArrayLike],
        noise: ArrayLike | Callable[[TimeStep], ArrayLike],
        offset: ArrayLike | Callable[[TimeStep], ArrayLike] | None = None,
        state_dimension: int | None = None,
        dimension: int | None = None,
    ) -> None:
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "noise", noise)
        object.__setattr__(self, "offset", offset)
        object.__setattr__(self, "state_dimension", state_dimension)
        object.__setattr__(self, "dimension", dimension)
        self.__post_init__()


@dataclass(frozen=True)
class GaussianTransition(_GaussianKernel):
    """Additive Gaussian dynamics; function(state, step), optional derivative."""

    dimension: int
    function: Callable[[FloatArray, TimeStep], ArrayLike]
    noise: ArrayLike | Callable[[TimeStep], ArrayLike]
    derivative: Callable[[FloatArray, TimeStep], ArrayLike] | None = None

    def __post_init__(self) -> None:
        d = v.integer(self.dimension, "dimension", 1)
        if not callable(self.function) or (
            self.derivative is not None and not callable(self.derivative)
        ):
            raise TypeError("function and supplied derivative must be callable")
        object.__setattr__(
            self, "noise", v.parameter(self.noise, (d, d), "noise", cov=True)
        )

    @property
    def input_dimension(self):
        return self.dimension

    def mean(self, state: ArrayLike, step: TimeStep) -> FloatArray:
        state = v.shaped(state, (self.dimension,), "state")
        return v.shaped(
            self.function(state, step), (self.dimension,), "transition mean"
        )

    def covariance(self, step: TimeStep):
        return v.evaluate(
            self.noise,
            step,
            (self.dimension, self.dimension),
            "transition noise",
            cov=True,
        )

    def jacobian(self, state: ArrayLike, step: TimeStep) -> FloatArray:
        if self.derivative is None:
            raise NotImplementedError(
                "extended filtering requires an explicit transition derivative"
            )
        state = v.shaped(state, (self.dimension,), "state")
        return v.shaped(
            self.derivative(state, step),
            (self.dimension, self.dimension),
            "transition Jacobian",
        )


@dataclass(frozen=True)
class GaussianObservation(_GaussianObservation):
    """Additive Gaussian observations; function(state, step), optional derivative."""

    state_dimension: int
    dimension: int
    function: Callable[[FloatArray, TimeStep], ArrayLike]
    noise: ArrayLike | Callable[[TimeStep], ArrayLike]
    derivative: Callable[[FloatArray, TimeStep], ArrayLike] | None = None

    def __post_init__(self) -> None:
        v.integer(self.state_dimension, "state_dimension", 1)
        k = v.integer(self.dimension, "dimension", 1)
        if not callable(self.function) or (
            self.derivative is not None and not callable(self.derivative)
        ):
            raise TypeError("function and supplied derivative must be callable")
        object.__setattr__(
            self, "noise", v.parameter(self.noise, (k, k), "noise", cov=True)
        )

    def mean(self, state: ArrayLike, step: TimeStep) -> FloatArray:
        state = v.shaped(state, (self.state_dimension,), "state")
        return v.shaped(
            self.function(state, step), (self.dimension,), "observation mean"
        )

    def covariance(self, step: TimeStep):
        return v.evaluate(
            self.noise,
            step,
            (self.dimension, self.dimension),
            "observation noise",
            cov=True,
        )

    def jacobian(self, state: ArrayLike, step: TimeStep) -> FloatArray:
        if self.derivative is None:
            raise NotImplementedError(
                "extended filtering requires an explicit observation derivative"
            )
        state = v.shaped(state, (self.state_dimension,), "state")
        return v.shaped(
            self.derivative(state, step),
            (self.dimension, self.state_dimension),
            "observation Jacobian",
        )


def _law(callback, state, step, dimension):
    from ...distributions import Distribution

    law = callback(state, step)
    if not isinstance(law, Distribution) or law.event_shape != (dimension,):
        raise ValueError(
            "kernel callback must return a vector Distribution of the declared dimension"
        )
    return law


@dataclass(frozen=True)
class DistributionTransition:
    """Any conditional Convexity vector law; scalar states still use event (1,)."""

    dimension: int
    law: Callable[[FloatArray, TimeStep], Distribution]

    def __post_init__(self) -> None:
        v.integer(self.dimension, "dimension", 1)
        if not callable(self.law):
            raise TypeError("law must be callable")

    def distribution(self, state: ArrayLike, step: TimeStep):
        return _law(
            self.law, v.shaped(state, (self.dimension,), "state"), step, self.dimension
        )

    def sample(self, states: ArrayLike, step: TimeStep, *, rng: np.random.Generator):
        v.generator(rng)
        return np.array(
            [
                self.distribution(row, step).sample(rng=rng)
                for row in v.states(states, self.dimension)
            ]
        )

    def log_density(self, values: ArrayLike, states: ArrayLike, step: TimeStep):
        states = v.states(states, self.dimension)
        values = v.array(values, "values")
        if values.shape == (self.dimension,):
            values = np.broadcast_to(values, states.shape)
        values = v.shaped(values, states.shape, "values")
        return np.array(
            [
                float(self.distribution(x, step).log_density(y))
                for x, y in zip(states, values)
            ]
        )


@dataclass(frozen=True)
class DistributionObservation:
    """Observation law supplied as a state/time-dependent Distribution callback."""

    state_dimension: int
    dimension: int
    law: Callable[[FloatArray, TimeStep], Distribution]

    def __post_init__(self) -> None:
        v.integer(self.state_dimension, "state_dimension", 1)
        v.integer(self.dimension, "dimension", 1)
        if not callable(self.law):
            raise TypeError("law must be callable")

    def distribution(self, state: ArrayLike, step: TimeStep):
        return _law(
            self.law,
            v.shaped(state, (self.state_dimension,), "state"),
            step,
            self.dimension,
        )

    def sample(self, states: ArrayLike, step: TimeStep, *, rng: np.random.Generator):
        v.generator(rng)
        return np.array(
            [
                self.distribution(row, step).sample(rng=rng)
                for row in v.states(states, self.state_dimension)
            ]
        )

    def log_density(self, value, states: ArrayLike, step: TimeStep):
        x = v.states(states, self.state_dimension)
        value = np.asarray(value, dtype=float)
        if value.shape != (self.dimension,) or np.isinf(value).any():
            raise ValueError("observation must have finite or NaN coordinates")
        keep = np.flatnonzero(~np.isnan(value))
        if not len(keep):
            return np.zeros(len(x))
        result = []
        for row in x:
            law = self.distribution(row, step)
            if len(keep) != self.dimension:
                law = law.marginal(keep)  # Explicit capability failure if unsupported.
            result.append(float(law.log_density(value[keep])))
        return np.array(result)


def linear_gaussian(
    initial: GaussianState | Distribution,
    transition: ArrayLike | Callable[[TimeStep], ArrayLike],
    observation: ArrayLike | Callable[[TimeStep], ArrayLike],
    process_covariance: ArrayLike | Callable[[TimeStep], ArrayLike],
    observation_covariance: ArrayLike | Callable[[TimeStep], ArrayLike],
    *,
    transition_offset: ArrayLike | Callable[[TimeStep], ArrayLike] | None = None,
    observation_offset: ArrayLike | Callable[[TimeStep], ArrayLike] | None = None,
    time_mode: str = "discrete",
    time_unit: np.timedelta64 | None = None,
    state_names: Sequence[str] | None = None,
    observation_dimension: int | None = None,
) -> StateSpaceModel:
    """Convenience composition; callable matrices require explicit dimensions."""
    d = (
        len(initial.mean)
        if isinstance(initial, GaussianState)
        else initial.event_shape[0]
    )
    return StateSpaceModel(
        initial,
        LinearGaussianTransition(transition, process_covariance, transition_offset, d),
        LinearGaussianObservation(
            observation,
            observation_covariance,
            observation_offset,
            d,
            observation_dimension,
        ),
        time_mode,
        time_unit,
        None if state_names is None else tuple(state_names),
    )
