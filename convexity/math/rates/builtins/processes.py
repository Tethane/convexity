"""Rate dynamics conforming to the existing stochastic-process interface."""

from __future__ import annotations

from dataclasses import dataclass

from ...processes.builtins.base import Diffusion
from .._backends import quantlib as _backend
from .models import ShortRateModel


@dataclass(frozen=True)
class HullWhiteProcess(Diffusion):
    """Gaussian mean-reverting short-rate process tied to an initial term structure."""

    model: ShortRateModel

    def __post_init__(self) -> None:
        if self.model.kind != "hull_white" or self.model.measure != "Q":
            raise ValueError(
                "HullWhiteProcess requires a curve-fitted Q Hull–White model"
            )

    def drift(self, time: float, state):
        return _backend.hull_white_coefficient(
            self.model, time, self._state(state), "drift"
        )

    def diffusion(self, time: float, state):
        return _backend.hull_white_coefficient(
            self.model, time, self._state(state), "diffusion"
        )

    def _exact(self, state, dt, time):
        return _backend.hull_white_transition(self.model, state, dt, time)
