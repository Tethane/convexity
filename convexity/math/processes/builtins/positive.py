"""Multiplicative and square-root diffusions in level coordinates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..._typing import FloatArray
from .. import _validation as v
from .._backends.laws import atom, cir_law
from ..interfaces import Transition
from .base import Diffusion
from .gaussian import _parameters


@dataclass(frozen=True)
class GeometricBrownianMotion(Diffusion):
    """dS=mu S dt+sigma S dW. mu is the expected proportional drift."""

    mu: float = 0.0
    sigma: float = 1.0

    def __post_init__(self) -> None:
        _parameters(self, [("mu", None, False), ("sigma", 0, False)])

    def _state(self, state):
        state = super()._state(state)
        if state < 0:
            raise ValueError("GBM state must be nonnegative")
        return state

    def drift(self, time: float, state) -> FloatArray:
        v.scalar(time, "time")
        return v.array(self.mu * self._state(state), "drift")

    def diffusion(self, time: float, state) -> FloatArray:
        v.scalar(time, "time")
        return v.array([self.sigma * self._state(state)], "diffusion")

    def _exact(self, state, dt, time):
        from ...distributions import lognormal

        if state == 0:
            law = atom(state)
        elif self.sigma == 0:
            law = atom(np.exp(np.log(state) + self.mu * dt))
        else:
            law = lognormal(
                np.log(state) + (self.mu - 0.5 * self.sigma**2) * dt,
                self.sigma * np.sqrt(dt),
            )
        return Transition.from_distribution(law)


@dataclass(frozen=True)
class CIR(Diffusion):
    """dX=kappa(theta-X)dt+sigma sqrt(X)dW; exact noncentral chi-square law.

    kappa,theta > 0; sigma >= 0. Feller violations are allowed. theta=0's
    mixed atom/continuous transition is outside this constructor's scope.
    """

    kappa: float
    theta: float
    sigma: float
    methods = ("exact",)

    def __post_init__(self) -> None:
        _parameters(self, [("kappa", 0, True), ("theta", 0, True), ("sigma", 0, False)])

    @property
    def feller_satisfied(self):
        return 2 * self.kappa * self.theta >= self.sigma**2

    def _state(self, state):
        state = super()._state(state)
        if state < 0:
            raise ValueError("CIR state must be nonnegative")
        return state

    def drift(self, time: float, state) -> FloatArray:
        v.scalar(time, "time")
        return v.array(self.kappa * (self.theta - self._state(state)), "drift")

    def diffusion(self, time: float, state) -> FloatArray:
        v.scalar(time, "time")
        return v.array([self.sigma * np.sqrt(self._state(state))], "diffusion")

    def _exact(self, state, dt, time):
        law = (
            atom(
                state * np.exp(-self.kappa * dt)
                + self.theta * (-np.expm1(-self.kappa * dt))
            )
            if self.sigma == 0
            else cir_law(state, dt, self.kappa, self.theta, self.sigma)
        )
        return Transition.from_distribution(law)

    def stationary(self):
        from scipy.stats import gamma

        from ...distributions import Distribution
        from ...distributions._backends.scipy import ScalarAdapter

        if self.sigma == 0:
            return atom(self.theta)
        return Distribution(
            ScalarAdapter(
                gamma(
                    2 * self.kappa * self.theta / self.sigma**2,
                    scale=self.sigma**2 / (2 * self.kappa),
                )
            )
        )
