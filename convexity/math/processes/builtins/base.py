"""Shared Ito discretization and optional conveniences for built-in processes."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike

from ..._typing import FloatArray
from .. import _validation as v
from .._backends.laws import gaussian
from ..interfaces import Transition, diffusion_generator, sample_paths


class Diffusion:
    event_shape: tuple = ()
    noise_dimension: int = 1
    methods = ("exact", "euler", "sri2")

    def _state(self, state):
        return v.shaped(state, self.event_shape, "state")

    def drift(self, time: float, state):
        raise NotImplementedError

    def diffusion(self, time: float, state):
        raise NotImplementedError

    def _exact(self, state, dt, time):
        raise NotImplementedError(
            "this model has no exact transition; choose euler or sri2"
        )

    def transition(
        self,
        state,
        dt: float,
        *,
        time: float = 0.0,
        method: str = "exact",
        substeps: int = 1,
    ):
        x = self._state(state)
        time, dt = v.interval(time, dt)
        substeps = v.integer(substeps, "substeps", 1)
        v.choice(method, self.methods, "method")
        if method == "exact" and substeps != 1:
            raise ValueError("substeps applies only to discretized transitions")
        if dt == 0:
            return Transition.identity(x)
        if method == "exact":
            return self._exact(x, dt, time)
        if method == "euler" and substeps == 1:
            drift = v.shaped(self.drift(time, x), self.event_shape, "drift")
            sigma = v.shaped(
                self.diffusion(time, x),
                self.event_shape + (self.noise_dimension,),
                "diffusion",
            )
            cov = sigma @ sigma.T if self.event_shape else np.sum(sigma**2)
            return Transition.from_distribution(
                gaussian(x + dt * drift, dt * cov), method="euler", exact=False
            )
        from .._backends.sde import integrate

        return Transition(
            self.event_shape,
            method,
            "sdeint",
            False,
            lambda size, rng: integrate(self, x, time, dt, substeps, method, size, rng),
        )

    def generator(
        self,
        time: float,
        state,
        gradient: ArrayLike | Callable[[FloatArray], ArrayLike],
        hessian: ArrayLike | Callable[[FloatArray], ArrayLike],
    ):
        return diffusion_generator(self, time, state, gradient, hessian)

    def sample(
        self,
        initial: ArrayLike,
        times,
        *,
        draws: int = 1,
        rng: np.random.Generator,
        method: str = "exact",
        substeps: int = 1,
    ):
        return sample_paths(
            self, initial, times, draws=draws, rng=rng, method=method, substeps=substeps
        )
