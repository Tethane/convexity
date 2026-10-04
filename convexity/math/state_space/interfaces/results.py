"""Owned latent-state marginals and explicit filtering/smoothing results."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from particles.smoothing import ParticleHistory
from dataclasses import dataclass, field
from typing import Literal, cast

import numpy as np
from numpy.typing import ArrayLike

from ..._typing import BoolArray, FloatArray
from ...forecasting import TimeSeries
from .. import _validation as v

if TYPE_CHECKING:
    from ...distributions import Distribution


@dataclass(frozen=True)
class _GaussianLaw:
    location: FloatArray
    matrix: FloatArray
    measure: Literal["continuous"] = "continuous"
    capabilities = frozenset(
        {"mean", "variance", "covariance", "sample", "log_density", "marginal"}
    )

    @property
    def event_shape(self):
        return self.location.shape

    def evaluate(self, operation, *args, **kwargs):
        if operation == "mean":
            return self.location
        if operation == "variance":
            return np.diag(self.matrix)
        if operation == "covariance":
            return self.matrix
        if operation == "sample":
            from ..._backends.gaussian import sample

            return sample(self.location, self.matrix, args[0], kwargs["rng"])
        if operation == "marginal":
            ix = args[0]
            return GaussianState(
                self.location[ix], self.matrix[np.ix_(ix, ix)]
            ).distribution()
        from ..._backends.gaussian import logpdf

        return logpdf(args[0], self.location, self.matrix)

    def __init__(
        self,
        location: ArrayLike,
        matrix: ArrayLike,
        measure: Literal["continuous"] = "continuous",
    ) -> None:
        object.__setattr__(self, "location", location)
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "measure", measure)


@dataclass(frozen=True)
class GaussianState:
    """A vector latent state with a positive-semidefinite covariance.

    Singular Gaussians are supported. Their density uses the SciPy convention
    on the affine support; a fully deterministic state is an empirical atom.
    """

    mean: FloatArray
    covariance: FloatArray

    def __post_init__(self) -> None:
        mean = v.array(self.mean, "state mean", ndim=1)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "covariance", v.covariance(self.covariance, len(mean)))

    def distribution(self) -> Distribution:
        from ...distributions import Distribution, empirical

        if not np.any(self.covariance):
            return empirical(self.mean[None, :])
        return Distribution(_GaussianLaw(self.mean, self.covariance))

    def __init__(
        self,
        mean: ArrayLike,
        covariance: ArrayLike,
    ) -> None:
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "covariance", covariance)
        self.__post_init__()


@dataclass(frozen=True)
class StateSequence:
    """Time-indexed marginals, never an implicitly independent joint path law.

    Means (n,d), covariances (n,d,d); optional particles (n,N,d), weights (n,N).
    Particle moments are computed from those exact normalized masses.
    """

    mean: FloatArray
    covariance: FloatArray
    particles: FloatArray | None = cast("FloatArray | None", None)
    weights: FloatArray | None = cast("FloatArray | None", None)

    def __post_init__(self) -> None:
        mean = v.array(self.mean, "means", ndim=2)
        n, d = mean.shape
        cov = v.shaped(self.covariance, (n, d, d), "covariances")
        cov = v.array([v.covariance(c, d) for c in cov], "covariances")
        if (self.particles is None) != (self.weights is None):
            raise ValueError("particles and weights must be supplied together")
        if self.particles is not None:
            x = v.array(self.particles, "particles", ndim=3)
            w = v.shaped(self.weights, x.shape[:2], "weights")
            if (
                x.shape[0] != n
                or x.shape[2] != d
                or np.any(w < 0)
                or not np.allclose(w.sum(axis=1), 1)
            ):
                raise ValueError(
                    "particles must align with means and weights must sum to one"
                )
            w = v.array(w / w.sum(axis=1, keepdims=True), "weights")
            expected = np.einsum("tn,tnd->td", w, x)
            centered = x - expected[:, None, :]
            expected_cov = np.einsum("tn,tni,tnj->tij", w, centered, centered)
            if not np.allclose(mean, expected) or not np.allclose(cov, expected_cov):
                raise ValueError("moments must agree with weighted particles")
            object.__setattr__(self, "particles", x)
            object.__setattr__(self, "weights", w)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "covariance", cov)

    @classmethod
    def from_particles(cls, particles: ArrayLike, weights: ArrayLike):
        x = v.array(particles, "particles", ndim=3)
        w = v.shaped(weights, x.shape[:2], "weights")
        if np.any(w < 0) or np.any(w.sum(axis=1) <= 0):
            raise ValueError("weights must have positive nonnegative mass in each row")
        w = w / w.sum(axis=1, keepdims=True)
        mean = np.einsum("tn,tnd->td", w, x)
        centered = x - mean[:, None, :]
        cov = np.einsum("tn,tni,tnj->tij", w, centered, centered)
        return cls(mean, cov, x, w)

    def distribution(self, index: int) -> Distribution:
        from ...distributions import empirical

        i = v.integer(index, "index")
        if i >= len(self.mean):
            raise ValueError("state index out of range")
        if self.particles is not None:
            assert self.weights is not None
            return empirical(self.particles[i], self.weights[i])
        return GaussianState(self.mean[i], self.covariance[i]).distribution()

    def __init__(
        self,
        mean: ArrayLike,
        covariance: ArrayLike,
        particles: ArrayLike | None = None,
        weights: ArrayLike | None = None,
    ) -> None:
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "particles", particles)
        object.__setattr__(self, "weights", weights)
        self.__post_init__()


@dataclass(frozen=True)
class FilterResult:
    """Filtered state laws and observations, with likelihood/diagnostics and optional smoothing information."""

    observations: TimeSeries
    filtered: StateSequence
    log_likelihoods: FloatArray
    backend: str
    method: str
    predicted: StateSequence | None = None
    smoothed: StateSequence | None = None
    observation_mean: FloatArray | None = cast("FloatArray | None", None)
    observation_covariance: FloatArray | None = cast("FloatArray | None", None)
    paths: FloatArray | None = cast("FloatArray | None", None)
    effective_sample_size: FloatArray | None = cast("FloatArray | None", None)
    resampled: BoolArray | None = cast("BoolArray | None", None)
    state_names: tuple[str, ...] | None = None
    approximation: str = "exact Gaussian"
    _history: ParticleHistory | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.observations, TimeSeries) or not isinstance(
            self.filtered, StateSequence
        ):
            raise TypeError(
                "observations and filtered must be TimeSeries and StateSequence"
            )
        shape = self.filtered.mean.shape
        n, d = shape
        if n != len(self.observations):
            raise ValueError("states must align with observations")
        for name in ("predicted", "smoothed"):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, StateSequence) or value.mean.shape != shape
            ):
                raise ValueError(f"{name} must align with filtered states")
        ll = v.shaped(self.log_likelihoods, (n,), "log_likelihoods")
        object.__setattr__(self, "log_likelihoods", ll)
        if (self.observation_mean is None) != (self.observation_covariance is None):
            raise ValueError(
                "observation predictive means and covariances must be supplied together"
            )
        if self.observation_mean is not None:
            k = self.observations.dimension
            object.__setattr__(
                self,
                "observation_mean",
                v.shaped(self.observation_mean, (n, k), "observation mean"),
            )
            cov = v.shaped(
                self.observation_covariance, (n, k, k), "observation covariance"
            )
            object.__setattr__(
                self,
                "observation_covariance",
                v.array([v.covariance(c, k) for c in cov], "observation covariance"),
            )
        if self.paths is not None:
            paths = v.array(self.paths, "paths", ndim=3)
            if paths.shape[1:] != shape or self.smoothed is None:
                raise ValueError(
                    "smoothing paths must have shape (draw,time,state) and smoothed marginals"
                )
            path_mean = paths.mean(axis=0)
            centered = paths - path_mean
            path_cov = np.einsum("mti,mtj->tij", centered, centered) / len(paths)
            if not np.allclose(path_mean, self.smoothed.mean) or not np.allclose(
                path_cov, self.smoothed.covariance
            ):
                raise ValueError("smoothed moments must agree with empirical paths")
            object.__setattr__(self, "paths", paths)
        if self.effective_sample_size is not None:
            ess = v.shaped(self.effective_sample_size, (n,), "effective_sample_size")
            if np.any(ess < 1 - 1e-10):
                raise ValueError("effective_sample_size must be at least one")
            if self.filtered.weights is not None and not np.allclose(
                ess, 1 / np.sum(self.filtered.weights**2, axis=1)
            ):
                raise ValueError(
                    "effective_sample_size must agree with filtered weights"
                )
            object.__setattr__(self, "effective_sample_size", ess)
        if self.resampled is not None:
            rs = np.array(self.resampled, copy=True)
            if rs.shape != (n,) or rs.dtype.kind != "b":
                raise ValueError(
                    "resampled must be a boolean vector aligned with observations"
                )
            rs.setflags(write=False)
            object.__setattr__(self, "resampled", rs)
        for name in ("backend", "method", "approximation"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a nonempty string")
        names = (
            tuple(f"x{i}" for i in range(d))
            if self.state_names is None
            else tuple(self.state_names)
        )
        if (
            len(names) != d
            or len(set(names)) != d
            or any(not isinstance(x, str) or not x for x in names)
        ):
            raise ValueError("state_names must align with state coordinates")
        object.__setattr__(self, "state_names", names)

    @property
    def log_likelihood(self) -> float:
        return float(self.log_likelihoods.sum())

    @property
    def timestamps(self):
        return self.observations.timestamps

    @property
    def innovations(self):
        if self.observation_mean is None:
            raise NotImplementedError(
                "this result has no Gaussian predictive observations"
            )
        out = self.observations.values - self.observation_mean
        out.setflags(write=False)
        return out

    def path_distribution(self):
        from ...distributions import empirical

        if self.paths is None:
            raise NotImplementedError(
                "marginal state estimates do not define a joint path law"
            )
        return empirical(self.paths.reshape(len(self.paths), -1))

    def __init__(
        self,
        observations: TimeSeries,
        filtered: StateSequence,
        log_likelihoods: ArrayLike,
        backend: str,
        method: str,
        predicted: StateSequence | None = None,
        smoothed: StateSequence | None = None,
        observation_mean: ArrayLike | None = None,
        observation_covariance: ArrayLike | None = None,
        paths: ArrayLike | None = None,
        effective_sample_size: ArrayLike | None = None,
        resampled: ArrayLike | None = None,
        state_names: tuple[str, ...] | None = None,
        approximation: str = "exact Gaussian",
        _history: ParticleHistory | None = None,
    ) -> None:
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "filtered", filtered)
        object.__setattr__(self, "log_likelihoods", log_likelihoods)
        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "predicted", predicted)
        object.__setattr__(self, "smoothed", smoothed)
        object.__setattr__(self, "observation_mean", observation_mean)
        object.__setattr__(self, "observation_covariance", observation_covariance)
        object.__setattr__(self, "paths", paths)
        object.__setattr__(self, "effective_sample_size", effective_sample_size)
        object.__setattr__(self, "resampled", resampled)
        object.__setattr__(self, "state_names", state_names)
        object.__setattr__(self, "approximation", approximation)
        object.__setattr__(self, "_history", _history)
        self.__post_init__()


@dataclass(frozen=True)
class SimulationResult:
    """Simulated latent states and observations aligned to the declared state-space time coordinates."""

    observations: TimeSeries
    states: FloatArray

    def __post_init__(self) -> None:
        states = v.array(self.states, "states", ndim=2)
        if len(states) != len(self.observations):
            raise ValueError("states must align with observations")
        object.__setattr__(self, "states", states)

    def __init__(
        self,
        observations: TimeSeries,
        states: ArrayLike,
    ) -> None:
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "states", states)
        self.__post_init__()
