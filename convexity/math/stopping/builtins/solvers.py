"""Snell recursion and independently evaluated least-squares exercise policies."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from .. import ExercisePolicy


from dataclasses import dataclass

from ..._finance import integer
from .._backends import numpy as _numpy
from .._backends import sklearn as _sklearn


class ContinuationRegressor(Protocol):
    """Predict continuation values from state rows; one scalar value per row."""
    def predict(self, states: ArrayLike, /) -> ArrayLike: ...


@dataclass(frozen=True)
class BackwardInduction:
    """Solve the supplied finite stopping problem backward over its exercise times."""

    def solve(self, problem):
        return _numpy.snell(self, problem)


def _inputs(states, rewards, discounts, exercise):
    return _sklearn._inputs(states, rewards, discounts, exercise)


@dataclass(frozen=True)
class ExercisePolicy:
    """Fitted continuation regressions; evaluate on independent Q paths.

    Validation output is a feasible-policy lower estimate, not a dual upper
    bound. Standard error assumes IID validation paths and fixed learned policy.
    """

    regressions: tuple[ContinuationRegressor | None, ...]
    exercise: tuple[bool, ...]
    state_dimension: int

    def __post_init__(self) -> None:
        integer(self.state_dimension, "state dimension", 1)
        object.__setattr__(self, "regressions", tuple(self.regressions))
        object.__setattr__(self, "exercise", tuple(self.exercise))
        if (
            len(self.exercise) < 2
            or len(self.regressions) != len(self.exercise) - 1
            or any(type(value) is not bool for value in self.exercise)
            or not self.exercise[-1]
        ):
            raise ValueError(
                "policy requires aligned regressions and boolean exercise dates including maturity"
            )
        if any(
            value is not None and not callable(getattr(value, "predict", None))
            for value in self.regressions
        ):
            raise TypeError("continuation regressions must implement predict")

    def evaluate(self, states, rewards, discounts):
        return _sklearn.evaluate_policy(self, states, rewards, discounts)


def least_squares_policy(
    states: ArrayLike,
    rewards: ArrayLike,
    discounts: ArrayLike,
    *,
    degree: int = 2,
    exercise: ArrayLike | None = None,
) -> ExercisePolicy:
    """Longstaff–Schwartz fit on Q paths; use independent paths for valuation.

    Nonnegative exercise rewards are required. State includes all information
    needed to make the contract Markov (e.g. running average for Asian claims).
    """
    return _sklearn.least_squares_policy(
        states, rewards, discounts, degree=degree, exercise=exercise
    )
