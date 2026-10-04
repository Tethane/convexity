"""Forward-rate simulations carry their actual pricing measure."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ..._finance import ResultRecord, array
from ..._typing import FloatArray


@dataclass(frozen=True)
class RatePaths(ResultRecord):
    """Simulated rate coordinates and time grid, with model-dependent state meanings."""

    values: FloatArray
    weights: FloatArray
    times: FloatArray
    measure: str = "terminal_forward"

    def __post_init__(self) -> None:
        x = array(self.values, "values", ndim=3)
        w = array(self.weights, "weights", ndim=1)
        t = array(self.times, "times", ndim=1)
        if (
            x.shape[:2] != (len(w), len(t))
            or np.any(w < 0)
            or not np.any(w > 0)
            or np.any(np.diff(t) <= 0)
            or t[0] < 0
        ):
            raise ValueError("forward paths, weights and increasing times must align")
        if self.measure != "terminal_forward":
            raise ValueError(
                "native market-model paths use the terminal forward measure"
            )
        for n, v in [("values", x), ("weights", w), ("times", t)]:
            object.__setattr__(self, n, v)

    def __init__(
        self,
        values: ArrayLike,
        weights: ArrayLike,
        times: ArrayLike,
        measure: str = "terminal_forward",
    ) -> None:
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "measure", measure)
        self.__post_init__()
