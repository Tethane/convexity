"""Independent-path evaluation of a fixed stopping policy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ..._finance import ResultRecord, array, scalar
from ..._typing import FloatArray, IntArray


@dataclass(frozen=True)
class PolicyEvaluation(ResultRecord):
    """Stopping-policy values on supplied paths, with standard error, exercise indices and discounted payoffs. Independent evaluation paths are the caller's responsibility."""

    value: float
    standard_error: float
    exercise_indices: IntArray
    discounted_payoffs: FloatArray

    def __post_init__(self) -> None:
        scalar(self.value, "value")
        scalar(self.standard_error, "standard error", minimum=0)
        pay = array(self.discounted_payoffs, "payoffs", ndim=1)
        indices = array(self.exercise_indices, "exercise indices", ndim=1)
        if (
            indices.shape != pay.shape
            or np.any(indices < 0)
            or np.any(indices != np.floor(indices))
        ):
            raise ValueError("exercise indices must align with discounted payoffs")
        indices = indices.astype(np.int64)
        indices.setflags(write=False)
        object.__setattr__(self, "discounted_payoffs", pay)
        object.__setattr__(self, "exercise_indices", indices)

    def __init__(
        self,
        value: float,
        standard_error: float,
        exercise_indices: ArrayLike,
        discounted_payoffs: ArrayLike,
    ) -> None:
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "standard_error", standard_error)
        object.__setattr__(self, "exercise_indices", exercise_indices)
        object.__setattr__(self, "discounted_payoffs", discounted_payoffs)
        self.__post_init__()
