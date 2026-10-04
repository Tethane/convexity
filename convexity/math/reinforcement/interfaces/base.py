"""Finite RL results and policy extension points; no tensor imports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypeVar, cast

from numpy.typing import ArrayLike

from ... import _decision as v
from ..._typing import FloatArray, IntArray


@dataclass(frozen=True)
class QResult:
    """Learned state-action values and training information in the declared finite state/action order."""

    q_values: FloatArray
    policy: IntArray
    residual: float
    iterations: int
    coefficients: FloatArray | None = cast("FloatArray | None", None)

    def __post_init__(self) -> None:
        q = v.array(self.q_values, "q values", ndim=2)
        object.__setattr__(self, "q_values", q)
        object.__setattr__(self, "policy", v.indices(self.policy, (len(q),), "policy"))
        v.scalar(self.residual, "residual", minimum=0)
        v.integer(self.iterations, "iterations", 1)
        if self.coefficients is not None:
            object.__setattr__(
                self, "coefficients", v.array(self.coefficients, "coefficients", ndim=1)
            )

    def __init__(
        self,
        q_values: ArrayLike,
        policy: ArrayLike,
        residual: float,
        iterations: int,
        coefficients: ArrayLike | None = None,
    ) -> None:
        object.__setattr__(self, "q_values", q_values)
        object.__setattr__(self, "policy", policy)
        object.__setattr__(self, "residual", residual)
        object.__setattr__(self, "iterations", iterations)
        object.__setattr__(self, "coefficients", coefficients)
        self.__post_init__()


State = TypeVar("State", contravariant=True)
Action = TypeVar("Action", covariant=True)


class Agent(Protocol[State, Action]):
    """Map a state to an action; generic parameters retain the user-defined state and action types."""
    def action(self, state: State) -> Action: ...
