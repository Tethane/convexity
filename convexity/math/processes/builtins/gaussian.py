"""Brownian motion and exact affine Gaussian dynamics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
from numpy.typing import ArrayLike

from ..._typing import (
    FloatArray,
)
from .. import _validation as v
from .._backends.laws import gaussian
from ..interfaces import Transition
from .base import Diffusion


def _parameters(obj, specifications):
    for name, minimum, strict in specifications:
        object.__setattr__(
            obj,
            name,
            v.scalar(getattr(obj, name), name, minimum=minimum, strict=strict),
        )


@dataclass(frozen=True)
class ArithmeticBrownianMotion(Diffusion):
    """dX = mu dt + sigma dW. sigma is standard deviation per sqrt(time)."""

    mu: float = 0.0
    sigma: float = 1.0

    def __post_init__(self) -> None:
        _parameters(self, [("mu", None, False), ("sigma", 0, False)])

    def drift(self, time: float, state) -> FloatArray:
        v.scalar(time, "time")
        self._state(state)
        return v.array(self.mu, "drift")

    def diffusion(self, time: float, state) -> FloatArray:
        v.scalar(time, "time")
        self._state(state)
        return v.array([self.sigma], "diffusion")

    def _exact(self, state, dt, time):
        return Transition.from_distribution(
            gaussian(state + self.mu * dt, self.sigma**2 * dt)
        )


@dataclass(frozen=True)
class BrownianMotion(ArithmeticBrownianMotion):
    """Standard scalar Wiener process; the initial level is supplied separately."""

    mu: float = field(default=0.0, init=False)
    sigma: float = field(default=1.0, init=False)


@dataclass(frozen=True)
class OrnsteinUhlenbeck(Diffusion):
    """dX = kappa(theta-X) dt + sigma dW; kappa=0 is driftless Brownian motion."""

    kappa: float
    theta: float = 0.0
    sigma: float = 1.0

    def __post_init__(self) -> None:
        _parameters(
            self, [("kappa", 0, False), ("theta", None, False), ("sigma", 0, False)]
        )

    def drift(self, time: float, state) -> FloatArray:
        v.scalar(time, "time")
        state = self._state(state)
        return v.array(
            0.0 if self.kappa == 0 else self.kappa * (self.theta - state), "drift"
        )

    def diffusion(self, time: float, state) -> FloatArray:
        v.scalar(time, "time")
        self._state(state)
        return v.array([self.sigma], "diffusion")

    def _exact(self, state, dt, time):
        decay = np.exp(-self.kappa * dt)
        mass = -np.expm1(-self.kappa * dt)
        mean = decay * state + mass * self.theta
        variance = self.sigma**2 * (
            dt
            if self.kappa == 0
            else 0.5 * (-np.expm1(-2 * self.kappa * dt)) / self.kappa
        )
        return Transition.from_distribution(gaussian(mean, variance))

    def stationary(self):
        if self.kappa == 0:
            raise ValueError("a unique stationary OU law requires kappa > 0")
        return gaussian(self.theta, self.sigma**2 / (2 * self.kappa))


@dataclass(frozen=True)
class LinearDiffusion(Diffusion):
    """dX=(A X+b)dt+B dW with vector state, including singular A and noise.

    The exact semigroup reuses the family-10 SciPy matrix-exponential adapter.
    """

    matrix: FloatArray
    loading: FloatArray
    forcing: FloatArray = cast("FloatArray", None)
    _kernel: Any = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        a = v.array(self.matrix, "matrix", ndim=2)
        a = v.shaped(a, (len(a), len(a)), "matrix")
        b = v.array(self.loading, "loading", ndim=2)
        if b.shape[0] != len(a):
            raise ValueError(
                "loading must have shape (state_dimension,noise_dimension)"
            )
        forcing = np.zeros(len(a)) if self.forcing is None else self.forcing
        forcing = v.shaped(forcing, (len(a),), "forcing")
        object.__setattr__(self, "matrix", a)
        object.__setattr__(self, "loading", b)
        object.__setattr__(self, "forcing", forcing)
        from ...state_space import continuous_linear_transition

        object.__setattr__(
            self, "_kernel", continuous_linear_transition(a, b @ b.T, forcing=forcing)
        )

    @property
    def event_shape(self):
        return (len(self.matrix),)

    @property
    def noise_dimension(self):
        return self.loading.shape[1]

    def drift(self, time: float, state) -> FloatArray:
        v.scalar(time, "time")
        return v.array(self.matrix @ self._state(state) + self.forcing, "drift")

    def diffusion(self, time: float, state):
        v.scalar(time, "time")
        self._state(state)
        return self.loading

    def coefficients(self, dt: float):
        """Return (F, offset, Q) for X[t+dt]=F X[t]+offset+N(0,Q)."""
        from ...state_space import TimeStep

        dt = v.scalar(dt, "dt", minimum=0)
        step = TimeStep(1, dt, dt)
        matrix, offset = self._kernel.coefficients(step)
        return matrix, offset, self._kernel.covariance(step)

    def _exact(self, state, dt, time):
        matrix, offset, covariance = self.coefficients(dt)
        return Transition.from_distribution(
            gaussian(matrix @ state + offset, covariance)
        )

    def stationary(self):
        from scipy.linalg import solve_continuous_lyapunov

        if np.any(np.real(np.linalg.eigvals(self.matrix)) >= 0):
            raise ValueError("a unique stationary law requires a Hurwitz drift matrix")
        mean = np.linalg.solve(self.matrix, -self.forcing)
        covariance = solve_continuous_lyapunov(
            self.matrix, -self.loading @ self.loading.T
        )
        return gaussian(mean, covariance)

    def __init__(
        self,
        matrix: ArrayLike,
        loading: ArrayLike,
        forcing: ArrayLike | None = None,
    ) -> None:
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "loading", loading)
        object.__setattr__(self, "forcing", forcing)
        self.__post_init__()


class VectorOrnsteinUhlenbeck(LinearDiffusion):
    """dX=K(theta-X)dt+B dW; stability requires eigenvalues(K).real > 0."""

    def __init__(self, kappa: ArrayLike, theta: ArrayLike, loading: ArrayLike) -> None:
        theta = v.array(theta, "theta", ndim=1)
        kappa = v.shaped(kappa, (len(theta), len(theta)), "kappa")
        super().__init__(-kappa, loading, kappa @ theta)
