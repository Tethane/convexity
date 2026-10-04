"""Finite-state and simulated optimal stopping contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, cast

import numpy as np
from numpy.typing import ArrayLike

from ..._finance import array
from ..._typing import BoolArray, FloatArray


@dataclass(frozen=True)
class StoppingProblem:
    """Exercise rewards and state paths for a stopping decision under the supplied discount convention."""

    rewards: FloatArray
    transitions: FloatArray
    discounts: FloatArray
    exercise: BoolArray = cast("BoolArray", None)

    def __post_init__(self) -> None:
        r = array(self.rewards, "rewards", ndim=2)
        n, k = r.shape
        p = array(self.transitions, "transitions", ndim=3)
        discounts = array(self.discounts, "discounts")
        if discounts.ndim == 1 and discounts.shape == (n - 1,):
            discounts = discounts[:, None]
        d = np.broadcast_to(discounts, (n - 1, k)).copy()
        if (
            n < 2
            or p.shape != (n - 1, k, k)
            or np.any(p < 0)
            or not np.allclose(p.sum(axis=2), 1, rtol=1e-10, atol=1e-12)
            or np.any(d <= 0)
        ):
            raise ValueError(
                "transitions must be stochastic (time-1,state,state); discounts positive"
            )
        e = (
            np.ones(n, dtype=bool)
            if self.exercise is None
            else np.asarray(self.exercise)
        )
        if e.dtype.kind != "b" or e.shape != (n,) or not e[-1]:
            raise ValueError("boolean exercise mask must include maturity")
        e = e.copy()
        e.setflags(write=False)
        for name, x in [
            ("rewards", r),
            ("transitions", p),
            ("discounts", array(d, "discounts")),
            ("exercise", e),
        ]:
            object.__setattr__(self, name, x)

    def __init__(
        self,
        rewards: ArrayLike,
        transitions: ArrayLike,
        discounts: ArrayLike,
        exercise: ArrayLike | None = None,
    ) -> None:
        object.__setattr__(self, "rewards", rewards)
        object.__setattr__(self, "transitions", transitions)
        object.__setattr__(self, "discounts", discounts)
        object.__setattr__(self, "exercise", exercise)
        self.__post_init__()


@dataclass(frozen=True)
class StoppingResult:
    """Estimated stopping values and policy information; inspect training/evaluation semantics before pricing."""

    values: FloatArray
    continuation: FloatArray
    exercise_region: BoolArray

    def __post_init__(self) -> None:
        v = array(self.values, "values", ndim=2)
        c = array(self.continuation, "continuation", ndim=2)
        e = np.asarray(self.exercise_region)
        if c.shape != v.shape or e.shape != v.shape or e.dtype.kind != "b":
            raise ValueError("stopping result arrays must align")
        e = e.copy()
        e.setflags(write=False)
        object.__setattr__(self, "values", v)
        object.__setattr__(self, "continuation", c)
        object.__setattr__(self, "exercise_region", e)

    def __init__(
        self,
        values: ArrayLike,
        continuation: ArrayLike,
        exercise_region: ArrayLike,
    ) -> None:
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "continuation", continuation)
        object.__setattr__(self, "exercise_region", exercise_region)
        self.__post_init__()


class StoppingSolver(Protocol):
    """Solve the supplied stopping problem and retain policy/value semantics in StoppingResult."""
    def solve(self, problem: StoppingProblem) -> StoppingResult: ...
