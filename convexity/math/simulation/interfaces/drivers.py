"""Reusable Brownian increments for common-random-number comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike

from ..._typing import (
    FloatArray,
)
from .. import _validation as v
from .paths import PathScenarios


class PathSampler(Protocol):
    """Structural extension: return a checked PathScenarios from sample()."""

    def sample(
        self,
        initial: ArrayLike,
        times: ArrayLike,
        *,
        draws: int,
        rng: np.random.Generator,
    ) -> PathScenarios: ...


@dataclass(frozen=True)
class BrownianIncrements:
    """Independent Wiener coordinates, values (draw,interval,noise_dimension).

    Covariance within each row is dt*I. Correlation belongs in model loadings.
    Supplied arrays are trusted as driver realizations, not statistically tested.
    Antithetic order is all base rows followed by their negatives.
    """

    times: FloatArray
    values: FloatArray
    sampling: str = "unspecified"

    def __post_init__(self) -> None:
        times = v.times(self.times)
        values = v.array(self.values, "increments", ndim=3, empty=True)
        if (
            values.shape[0] < 1
            or values.shape[1] != len(times) - 1
            or values.shape[2] < 1
        ):
            raise ValueError(
                "increments must have shape (draw,len(times)-1,noise_dimension)"
            )
        v.sampling(self.sampling)
        if self.sampling == "antithetic":
            n = len(values) // 2
            if len(values) % 2 or not np.array_equal(values[:n], -values[n:]):
                raise ValueError(
                    "antithetic rows must be base rows followed by their negatives"
                )
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "values", values)

    def coarsen(self, indices) -> BrownianIncrements:
        """Sum adjacent increments; retain both endpoints and all driver coupling."""
        raw = np.asarray(indices)
        if (
            raw.ndim != 1
            or raw.dtype.kind not in "iu"
            or len(raw) < 1
            or raw[0] != 0
            or raw[-1] != len(self.times) - 1
            or np.any(np.diff(raw.astype(float)) <= 0)
        ):
            raise ValueError("indices must increase and include both grid endpoints")
        values = np.empty((len(self.values), len(raw) - 1, self.values.shape[2]))
        for j, (left, right) in enumerate(zip(raw[:-1], raw[1:])):
            values[:, j] = self.values[:, left:right].sum(axis=1)
        return BrownianIncrements(self.times[raw], values, self.sampling)

    def __init__(
        self,
        times: ArrayLike,
        values: ArrayLike,
        sampling: str = "unspecified",
    ) -> None:
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "sampling", sampling)
        self.__post_init__()
