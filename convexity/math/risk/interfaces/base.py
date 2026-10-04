"""Finite loss laws and risk contributions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, cast

from numpy.typing import ArrayLike

from ... import _decision as v
from ..._typing import FloatArray


@dataclass(frozen=True)
class LossScenarios:
    """Scalar losses (scenario,), positive means adverse; normalized masses."""

    losses: FloatArray
    weights: FloatArray = cast("FloatArray", None)

    def __post_init__(self) -> None:
        x = v.array(self.losses, "losses", ndim=1)
        object.__setattr__(self, "losses", x)
        object.__setattr__(self, "weights", v.weights(self.weights, len(x)))

    def __init__(
        self,
        losses: ArrayLike,
        weights: ArrayLike | None = None,
    ) -> None:
        object.__setattr__(self, "losses", losses)
        object.__setattr__(self, "weights", weights)
        self.__post_init__()


@dataclass(frozen=True)
class RiskContributions:
    """Risk allocated across the declared coordinates; the allocation convention depends on the chosen risk functional."""

    value: float
    marginal: FloatArray
    component: FloatArray

    def __post_init__(self) -> None:
        v.scalar(self.value, "value")
        m, c = (
            v.array(self.marginal, "marginal", ndim=1),
            v.array(self.component, "component", ndim=1),
        )
        if m.shape != c.shape:
            raise ValueError("contributions must align")
        object.__setattr__(self, "marginal", m)
        object.__setattr__(self, "component", c)

    def __init__(
        self,
        value: float,
        marginal: ArrayLike,
        component: ArrayLike,
    ) -> None:
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "marginal", marginal)
        object.__setattr__(self, "component", component)
        self.__post_init__()


class RiskMeasure(Protocol):
    """Evaluate risk of supplied outcomes under the family's loss/return sign convention."""
    def __call__(self, law: LossScenarios) -> float: ...
