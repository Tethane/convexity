"""Path-shaped laws and finite weighted joint scenarios."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import ArrayLike

from ..._typing import (
    FloatArray,
    SampleSize,
)
from ...distributions import Distribution, empirical
from .. import _validation as v


@dataclass(frozen=True)
class PathDistribution:
    """A Distribution on time-major flattened coordinates, with explicit times.

    One event is (time,*state_shape). Density is with respect to the underlying
    law's measure; singular Gaussian laws retain their supported-subspace density.
    This wrapper never manufactures temporal dependence from marginal intervals.
    """

    times: FloatArray
    state_shape: tuple[int, ...]
    law: Distribution

    def __post_init__(self) -> None:
        object.__setattr__(self, "times", v.times(self.times))
        v.shape(self.state_shape)
        if not isinstance(self.law, Distribution) or self.law.event_shape != (
            int(np.prod(self.event_shape)),
        ):
            raise ValueError(
                "law must be a Distribution over time-major flattened path coordinates"
            )

    @property
    def event_shape(self):
        return (len(self.times),) + self.state_shape

    @property
    def capabilities(self):
        return self.law.capabilities

    def sample(self, size: SampleSize = (), *, rng: np.random.Generator) -> FloatArray:
        size = v.sample_shape(size)
        return v.shaped(
            self.law.sample(size, rng=rng).reshape(size + self.event_shape),
            size + self.event_shape,
            "samples",
        )

    def log_density(self, paths: ArrayLike):
        values = v.array(paths, "paths", empty=True)
        n = len(self.event_shape)
        if values.ndim < n or values.shape[-n:] != self.event_shape:
            raise ValueError("paths must have trailing event_shape")
        return self.law.log_density(
            values.reshape(values.shape[:-n] + self.law.event_shape)
        )

    def mean(self):
        return self.law.mean().reshape(self.event_shape)

    def variance(self):
        return self.law.variance().reshape(self.event_shape)

    def covariance(self):
        """Time-major (time*dimension,time*dimension) joint covariance."""
        return self.law.covariance()

    def marginal(self, index: int):
        """Vector law at one grid index (one coordinate for scalar states)."""
        index = v.integer(index, "index")
        if index >= len(self.times):
            raise ValueError("time index out of range")
        d = int(np.prod(self.state_shape))
        return self.law.marginal(np.arange(index * d, (index + 1) * d))

    def expect(self, function):
        return self.law.expect(
            lambda x: function(np.asarray(x).reshape(self.event_shape))
        )

    def probability(self, event):
        return self.law.probability(lambda x: event(x.reshape(self.event_shape)))

    def map(self, function):
        return self.law.map(lambda x: function(np.asarray(x).reshape(self.event_shape)))

    def __init__(
        self,
        times: ArrayLike,
        state_shape: tuple[int, ...],
        law: Distribution,
    ) -> None:
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "state_shape", state_shape)
        object.__setattr__(self, "law", law)
        self.__post_init__()


@dataclass(frozen=True)
class PathScenarios:
    """Finite joint law: values (scenario,time[,state]), normalized masses.

    `exact` describes the source sampler, never claims that this finite law is
    the continuous model law. `sampling` describes row dependence before any
    reweighting. No Monte Carlo error is inferred from weights or effective size.
    """

    times: FloatArray
    values: FloatArray
    weights: FloatArray = cast("FloatArray", None)
    sampling: str = "unspecified"
    method: str = "supplied"
    backend: str = "numpy"
    exact: bool = False

    def __post_init__(self) -> None:
        times = v.times(self.times)
        values = v.array(self.values, "values")
        if values.ndim not in (2, 3) or values.shape[1] != len(times):
            raise ValueError(
                "values must have shape (scenario,time) or (scenario,time,state)"
            )
        mass = v.weights(self.weights, len(values))
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "weights", v.array(mass / mass.sum(), "weights"))
        v.sampling(self.sampling)
        if self.sampling == "antithetic" and len(values) % 2:
            raise ValueError("antithetic scenarios require an even number of rows")
        v.label(self.method, "method")
        v.label(self.backend, "backend")
        v.boolean(self.exact, "exact")

    @classmethod
    def from_process_paths(
        cls, paths, *, weights: ArrayLike | None = None, sampling: str = "unspecified"
    ):
        from ...processes import ProcessPaths

        if not isinstance(paths, ProcessPaths):
            raise TypeError("paths must be ProcessPaths")
        return cls(
            paths.times,
            paths.values,
            weights,
            sampling,
            paths.method,
            paths.backend,
            paths.exact,
        )

    @property
    def state_shape(self):
        return self.values.shape[2:]

    @property
    def effective_size(self) -> float:
        """Weight concentration 1/sum(w²); not independent sample count."""
        return float(1 / np.sum(self.weights**2))

    def distribution(self) -> PathDistribution:
        law = empirical(self.values.reshape(len(self.values), -1), self.weights)
        return PathDistribution(self.times, self.state_shape, law)

    def mean(self):
        return self.distribution().mean()

    def covariance(self):
        return self.distribution().covariance()

    def expect(self, function):
        return self.distribution().expect(function)

    def probability(self, event):
        return self.distribution().probability(event)

    def map(self, function):
        return self.distribution().map(function)

    def reweight(self, likelihood: ArrayLike) -> PathScenarios:
        """Multiply masses by supplied finite nonnegative relative likelihoods."""
        factors = v.shaped(likelihood, self.weights.shape, "likelihood")
        if np.any(factors < 0) or not np.any(factors > 0):
            raise ValueError("likelihood must be nonnegative with positive mass")
        return PathScenarios(
            self.times,
            self.values,
            self.weights * (factors / factors.max()),
            method=self.method,
            backend=self.backend,
            exact=self.exact,
        )

    def resample(self, draws: int, *, rng: np.random.Generator) -> PathScenarios:
        """IID resampling of entire rows preserves within-path dependence."""
        draws = v.integer(draws, "draws", 1)
        values = self.distribution().sample(draws, rng=rng)
        return PathScenarios(
            self.times, values, sampling="iid", method="empirical-resampling"
        )

    def to_forecast(self, *, horizon=None, model_identity: str = "simulation"):
        """All supplied times become forecast rows; caller chooses whether to omit t0."""
        from ...forecasting import ForecastResult

        law = self.distribution().law
        point = self.mean().reshape(len(self.times), -1)
        return ForecastResult(
            point, horizon=horizon, law=law, model_identity=model_identity
        )

    def __init__(
        self,
        times: ArrayLike,
        values: ArrayLike,
        weights: ArrayLike | None = None,
        sampling: str = "unspecified",
        method: str = "supplied",
        backend: str = "numpy",
        exact: bool = False,
    ) -> None:
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "sampling", sampling)
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "exact", exact)
        self.__post_init__()
