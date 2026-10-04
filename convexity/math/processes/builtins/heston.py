"""Heston coefficients and explicitly approximate QuantLib QE transitions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ..._typing import FloatArray
from .. import _validation as v
from ..interfaces import Transition
from .base import Diffusion
from .gaussian import _parameters


@dataclass(frozen=True)
class Heston(Diffusion):
    """dS=mu S dt+sqrt(V) S dW1; dV=kappa(theta-V)dt+sigma sqrt(V)dW2.

    State is (S,V), corr(dW1,dW2)=rho. mu is supplied by the caller; this
    object assigns no pricing/physical measure. QE is not an exact joint law.
    """

    mu: float
    kappa: float
    theta: float
    sigma: float
    rho: float
    event_shape = (2,)
    noise_dimension = 2
    methods = ("qe",)

    def __post_init__(self) -> None:
        _parameters(
            self,
            [
                ("mu", None, False),
                ("kappa", 0, True),
                ("theta", 0, True),
                ("sigma", 0, True),
                ("rho", None, False),
            ],
        )
        if abs(self.rho) > 1:
            raise ValueError("rho must lie in [-1,1]")

    def _state(self, state):
        state = super()._state(state)
        if state[0] <= 0 or state[1] < 0:
            raise ValueError("Heston requires positive spot and nonnegative variance")
        return state

    def drift(self, time: float, state) -> FloatArray:
        v.scalar(time, "time")
        spot, variance = self._state(state)
        return v.array([self.mu * spot, self.kappa * (self.theta - variance)], "drift")

    def diffusion(self, time: float, state) -> FloatArray:
        v.scalar(time, "time")
        spot, variance = self._state(state)
        return v.array(
            np.sqrt(variance)
            * np.array(
                [
                    [spot, 0.0],
                    [self.sigma * self.rho, self.sigma * np.sqrt(1 - self.rho**2)],
                ]
            ),
            "diffusion",
        )

    def transition(
        self,
        state,
        dt: float,
        *,
        time: float = 0.0,
        method: str = "qe",
        substeps: int = 1,
    ):
        state = self._state(state)
        time, dt = v.interval(time, dt)
        v.choice(method, self.methods, "method")
        substeps = v.integer(substeps, "substeps", 1)
        if dt == 0:
            return Transition.identity(state)
        from .._backends.quantlib import heston_kernel, heston_sample

        process = heston_kernel(self, state)
        return Transition(
            (2,),
            "qe",
            "QuantLib",
            False,
            lambda size, rng: heston_sample(process, state, dt, substeps, size, rng),
        )

    def sample(
        self,
        initial: ArrayLike,
        times,
        *,
        draws: int = 1,
        rng: np.random.Generator,
        method: str = "qe",
        substeps: int = 1,
    ):
        return super().sample(
            initial, times, draws=draws, rng=rng, method=method, substeps=substeps
        )
