from dataclasses import dataclass
from typing import Any, Callable
import numpy as np
from .. import _validation as v
from .base import check_array_result


@dataclass(frozen=True)
class FixedPointProblem:
    function: Callable
    x0: Any

    def __post_init__(self):
        v.function(self.function)
        x = v.array(self.x0, "x0", real=True)
        if x.ndim > 1:
            raise ValueError("x0 must be a scalar or vector")
        object.__setattr__(self, "x0", x)

    def validate_result(self, result):
        check_array_result(result, np.shape(self.x0), real=True)
