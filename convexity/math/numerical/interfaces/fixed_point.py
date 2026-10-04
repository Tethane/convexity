from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike

from ..._typing import (
    FloatArray,
)
from .. import _validation as v
from .base import check_array_result


@dataclass(frozen=True)
class FixedPointProblem:
    """Vector map f and initial x0 for solving f(x) = x."""

    function: Callable[[FloatArray], ArrayLike]
    x0: FloatArray

    def __post_init__(self) -> None:
        v.function(self.function)
        x = v.array(self.x0, "x0", real=True)
        if x.ndim > 1:
            raise ValueError("x0 must be a scalar or vector")
        object.__setattr__(self, "x0", x)

    def validate_result(self, result: object) -> None:
        check_array_result(result, np.shape(self.x0), real=True)

    def __init__(
        self,
        function: Callable[[FloatArray], ArrayLike],
        x0: ArrayLike,
    ) -> None:
        object.__setattr__(self, "function", function)
        object.__setattr__(self, "x0", x0)
        self.__post_init__()
