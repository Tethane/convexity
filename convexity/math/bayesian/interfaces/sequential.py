"""Joint static-parameter/latent-state filtering contracts."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy.typing import ArrayLike

from ..._typing import (
    FloatArray,
)

if TYPE_CHECKING:
    from ...distributions import Distribution
    from ...state_space import FilterResult, StateSpaceModel
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .. import _validation as v


@dataclass(frozen=True)
class StateParameterModel:
    """A proper vector parameter prior and theta -> StateSpaceModel factory.

    reference is used only to check dimensions/time conventions, never as a
    plug-in estimate. Each parameter draw receives its own conditional state prior.
    """

    prior: Distribution
    model_factory: Callable[[FloatArray], StateSpaceModel]
    reference: FloatArray
    parameter_names: tuple[str, ...] | None = None
    _prototype: StateSpaceModel = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        from ...distributions import Distribution
        from ...state_space import StateSpaceModel

        if not isinstance(self.prior, Distribution) or len(self.prior.event_shape) != 1:
            raise TypeError("parameter prior must be a vector Distribution")
        if not callable(self.model_factory):
            raise TypeError("model_factory must be callable")
        reference = v.array(self.reference, "reference", ndim=1)
        if reference.shape != self.prior.event_shape:
            raise ValueError("reference must match parameter prior shape")
        prototype = self.model_factory(reference)
        if not isinstance(prototype, StateSpaceModel):
            raise TypeError("model_factory must return StateSpaceModel")
        object.__setattr__(self, "reference", reference)
        object.__setattr__(self, "_prototype", prototype)
        object.__setattr__(
            self, "parameter_names", v.names(self.parameter_names, len(reference))
        )

    def conditional_model(self, theta: ArrayLike):
        from ...state_space import StateSpaceModel

        point = v.array(theta, "parameters", ndim=1)
        if point.shape != self.prior.event_shape:
            raise ValueError("parameters must match prior event shape")
        model = self.model_factory(point)
        if not isinstance(model, StateSpaceModel):
            raise TypeError("model_factory must return StateSpaceModel")
        ref = self._prototype
        for key in (
            "state_dimension",
            "observation_dimension",
            "time_mode",
            "time_unit",
            "state_names",
        ):
            if getattr(model, key) != getattr(ref, key):
                raise ValueError(f"conditional models must preserve {key}")
        return model

    def __init__(
        self,
        prior: Distribution,
        model_factory: Callable[[FloatArray], StateSpaceModel],
        reference: ArrayLike,
        parameter_names: tuple[str, ...] | None = None,
    ) -> None:
        object.__setattr__(self, "prior", prior)
        object.__setattr__(self, "model_factory", model_factory)
        object.__setattr__(self, "reference", reference)
        object.__setattr__(self, "parameter_names", parameter_names)
        self.__post_init__()


@dataclass(frozen=True)
class JointFilterResult:
    """Augmented filter coordinates are (parameters, states), in that order."""

    filtering: FilterResult
    parameter_names: tuple[str, ...]

    def __post_init__(self) -> None:
        from ...state_space import FilterResult

        if (
            not isinstance(self.filtering, FilterResult)
            or self.filtering.filtered.particles is None
        ):
            raise TypeError("filtering must be a particle FilterResult")
        names = v.names(self.parameter_names, len(self.parameter_names))
        if not names or len(names) >= self.filtering.filtered.mean.shape[1]:
            raise ValueError("joint results need both parameter and state coordinates")
        object.__setattr__(self, "parameter_names", names)

    def joint_posterior(self, index, *, smoothed: bool = False):
        v.boolean(smoothed, "smoothed")
        sequence = self.filtering.smoothed if smoothed else self.filtering.filtered
        if sequence is None:
            raise ValueError("smoothed states are unavailable")
        return sequence.distribution(index)

    def parameter_posterior(self, index):
        return self.joint_posterior(index).marginal(
            np.arange(len(self.parameter_names))
        )

    def state_posterior(self, index, *, smoothed: bool = False):
        law = self.joint_posterior(index, smoothed=smoothed)
        return law.marginal(np.arange(len(self.parameter_names), law.event_shape[0]))
