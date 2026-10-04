"""User-supplied time-dependent Ito coefficients, with no registration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from numpy.typing import ArrayLike

from ..._typing import FloatArray
from .. import _validation as v
from .base import Diffusion


@dataclass(frozen=True)
class ItoDiffusion(Diffusion):
    """Callbacks drift(time,state), diffusion(time,state); independent Wiener drivers.

    For scalar state (), diffusion returns (m,). For vector (d,), it returns
    (d,m). Correlation belongs in this loading, not in an implicit driver law.
    Coefficients are Ito, not Stratonovich. Callbacks receive owned read-only states.
    """

    drift_function: Callable[[float, FloatArray], ArrayLike]
    diffusion_function: Callable[[float, FloatArray], ArrayLike]
    event_shape: tuple[int, ...] = ()
    noise_dimension: int = 1
    methods = ("euler", "sri2")

    def __post_init__(self) -> None:
        v.shape(self.event_shape)
        v.integer(self.noise_dimension, "noise_dimension", 1)
        if not callable(self.drift_function) or not callable(self.diffusion_function):
            raise TypeError("drift_function and diffusion_function must be callable")

    def drift(self, time: float, state) -> FloatArray:
        return v.shaped(
            self.drift_function(v.scalar(time, "time"), self._state(state)),
            self.event_shape,
            "drift",
        )

    def diffusion(self, time: float, state) -> FloatArray:
        return v.shaped(
            self.diffusion_function(v.scalar(time, "time"), self._state(state)),
            self.event_shape + (self.noise_dimension,),
            "diffusion",
        )
