"""Finite-activity independent Poisson jumps, with explicit compensation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from ..._typing import (
    FloatArray,
    SampleSize,
)

if TYPE_CHECKING:
    from ...distributions import Distribution
from .. import _validation as v
from .._backends.laws import atom, gaussian
from ..interfaces import Transition
from .base import Diffusion
from .gaussian import _parameters


@dataclass(frozen=True)
class AdditiveJumpDiffusion(Diffusion):
    """dX=mu dt+B dW+J dN, N has constant intensity; marks are IID.

    mu is the uncompensated continuous drift. Scalar loading has shape (m,),
    vector loading (d,m). Existing Distribution marks supply size/shape laws.
    No common-reference density is asserted for this potentially mixed kernel.
    """

    mu: FloatArray
    loading: FloatArray
    intensity: float
    jumps: Distribution
    methods = ("exact",)

    def __post_init__(self) -> None:
        from ...distributions import Distribution

        mu = v.array(self.mu, "mu")
        v.shape(mu.shape)
        loading = v.array(self.loading, "loading")
        if loading.ndim != mu.ndim + 1 or loading.shape[:-1] != mu.shape:
            raise ValueError("loading must have shape event_shape + (noise_dimension,)")
        if (
            not isinstance(self.jumps, Distribution)
            or self.jumps.event_shape != mu.shape
        ):
            raise ValueError("jumps must be a Distribution with the drift event_shape")
        object.__setattr__(self, "mu", mu)
        object.__setattr__(self, "loading", loading)
        object.__setattr__(
            self, "intensity", v.scalar(self.intensity, "intensity", minimum=0)
        )

    @property
    def event_shape(self):
        return self.mu.shape

    @property
    def noise_dimension(self):
        return self.loading.shape[-1]

    def drift(self, time: float, state):
        v.scalar(time, "time")
        self._state(state)
        return self.mu

    def diffusion(self, time: float, state):
        v.scalar(time, "time")
        self._state(state)
        return self.loading

    def _exact(self, state, dt, time):
        covariance = (
            self.loading @ self.loading.T
            if self.event_shape
            else np.sum(self.loading**2)
        ) * dt
        continuous = gaussian(state + self.mu * dt, covariance)
        mass = self.intensity * dt
        if mass == 0:
            return Transition.from_distribution(continuous)

        def sample(size: SampleSize, rng: np.random.Generator):
            result = np.array(continuous.sample(size, rng=rng), copy=True)
            counts = rng.poisson(mass, size=int(np.prod(size)))
            marks = self.jumps.sample(int(counts.sum()), rng=rng)
            flat = result.reshape((len(counts),) + self.event_shape)
            np.add.at(flat, np.repeat(np.arange(len(counts)), counts), marks)
            return result

        mean, cov = None, None
        if "mean" in self.jumps.capabilities:
            jm = self.jumps.mean()
            if np.all(np.isfinite(jm)):
                mean = state + self.mu * dt + mass * jm
                if "covariance" in self.jumps.capabilities:
                    jc = self.jumps.covariance()
                    if np.all(np.isfinite(jc)):
                        cov = covariance + mass * (
                            jc + (np.outer(jm, jm) if self.event_shape else jm**2)
                        )
        return Transition(
            self.event_shape,
            "exact",
            "numpy/scipy",
            True,
            sample,
            _mean=mean,
            _covariance=cov,
        )

    def generator(
        self,
        time: float,
        state,
        gradient: ArrayLike | Callable[[FloatArray], ArrayLike],
        hessian: ArrayLike | Callable[[FloatArray], ArrayLike],
        *,
        function,
        jump_expectation=None,
    ):
        """Lf=continuous part+lambda E[f(x+J)-f(x)].

        Supply jump_expectation explicitly if the mark law cannot integrate f.
        It must be the expectation of the difference, without the intensity.
        """
        x = self._state(state)
        continuous = super().generator(time, x, gradient, hessian)
        if not callable(function):
            raise TypeError("function must be callable")
        if self.intensity == 0:
            return continuous
        difference = (
            self.jumps.expect(
                lambda jump: (
                    v.scalar(function(x + jump), "function")
                    - v.scalar(function(x), "function")
                )
            )
            if jump_expectation is None
            else jump_expectation
        )
        return v.scalar(
            continuous + self.intensity * v.scalar(difference, "jump_expectation"),
            "generator",
        )

    def __init__(
        self,
        mu: ArrayLike,
        loading: ArrayLike,
        intensity: float,
        jumps: Distribution,
    ) -> None:
        object.__setattr__(self, "mu", mu)
        object.__setattr__(self, "loading", loading)
        object.__setattr__(self, "intensity", intensity)
        object.__setattr__(self, "jumps", jumps)
        self.__post_init__()


@dataclass(frozen=True)
class MertonJumpDiffusion(Diffusion):
    """dS/S-=(mu-lambda E[e^J-1])dt+sigma dW+(e^J-1)dN.

    J~Normal(jump_mean,jump_sigma). mu is total expected proportional drift.
    Exact terminal sampling conditions on Poisson counts; no mixture truncation.
    """

    mu: float
    sigma: float
    intensity: float
    jump_mean: float
    jump_sigma: float
    methods = ("exact",)

    def __post_init__(self) -> None:
        _parameters(
            self,
            [
                ("mu", None, False),
                ("sigma", 0, False),
                ("intensity", 0, False),
                ("jump_mean", None, False),
                ("jump_sigma", 0, False),
            ],
        )
        v.scalar(self.compensator, "jump compensator")

    @property
    def compensator(self):
        return np.expm1(self.jump_mean + 0.5 * self.jump_sigma**2)

    def _state(self, state):
        state = super()._state(state)
        if state < 0:
            raise ValueError("Merton state must be nonnegative")
        return state

    def drift(self, time: float, state) -> FloatArray:
        v.scalar(time, "time")
        return v.array(
            (self.mu - self.intensity * self.compensator) * self._state(state), "drift"
        )

    def diffusion(self, time: float, state) -> FloatArray:
        v.scalar(time, "time")
        return v.array([self.sigma * self._state(state)], "diffusion")

    def _exact(self, state, dt, time):
        if state == 0:
            return Transition.from_distribution(atom(state), backend="numpy")
        if self.intensity == 0:
            from .positive import GeometricBrownianMotion

            return GeometricBrownianMotion(self.mu, self.sigma).transition(
                state, dt, time=time
            )
        drift = (self.mu - self.intensity * self.compensator - 0.5 * self.sigma**2) * dt

        def sample(size: SampleSize, rng: np.random.Generator):
            counts = rng.poisson(self.intensity * dt, size=size)
            logs = (
                drift
                + counts * self.jump_mean
                + np.sqrt(self.sigma**2 * dt + counts * self.jump_sigma**2)
                * rng.standard_normal(size)
            )
            return np.exp(np.log(state) + logs)

        mean = np.exp(np.log(state) + self.mu * dt)
        jump_second = (
            np.exp(2 * self.jump_mean + self.jump_sigma**2)
            * np.expm1(self.jump_sigma**2)
            + self.compensator**2
        )
        variance = mean**2 * np.expm1(
            (self.sigma**2 + self.intensity * jump_second) * dt
        )
        return Transition(
            (), "exact", "numpy", True, sample, _mean=mean, _covariance=variance
        )

    def generator(
        self,
        time: float,
        state,
        gradient: ArrayLike | Callable[[FloatArray], ArrayLike],
        hessian: ArrayLike | Callable[[FloatArray], ArrayLike],
        *,
        function,
        jump_expectation=None,
    ):
        x = self._state(state)
        continuous = super().generator(time, x, gradient, hessian)
        if not callable(function):
            raise TypeError("function must be callable")
        if self.intensity == 0:
            return continuous
        if jump_expectation is None:
            if self.jump_sigma == 0:
                difference = v.scalar(
                    function(x * np.exp(self.jump_mean)), "function"
                ) - v.scalar(function(x), "function")
            else:
                from scipy.integrate import quad
                from scipy.special import ndtri

                # Integrate over probability rather than infinite log-jump
                # coordinates, where exp(jump) can overflow even for log f.
                initial = v.scalar(function(x), "function")
                difference = quad(
                    lambda probability: (
                        v.scalar(
                            function(
                                x
                                * np.exp(
                                    self.jump_mean
                                    + self.jump_sigma * ndtri(probability)
                                )
                            ),
                            "function",
                        )
                        - initial
                    ),
                    0.0,
                    1.0,
                    epsabs=1e-9,
                    epsrel=1e-9,
                    limit=200,
                )[0]
        else:
            difference = jump_expectation
        return v.scalar(
            continuous + self.intensity * v.scalar(difference, "jump_expectation"),
            "generator",
        )
