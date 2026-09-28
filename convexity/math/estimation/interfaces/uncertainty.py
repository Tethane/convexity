"""Parameter intervals and resampling results with explicit interpretation."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .. import _validation as v
from .base import Estimate


@dataclass(frozen=True, init=False)
class Interval:
    lower: NDArray[Any]
    upper: NDArray[Any]
    level: float
    method: str
    kind: str = "confidence"
    names: tuple[str, ...] = ()

    def __init__(self, lower: ArrayLike, upper: ArrayLike, level: float, method: str,
                 kind: str = "confidence", names=None):
        lower = v.array(lower, "interval lower", ndim=1, finite=False)
        upper = v.array(upper, "interval upper", ndim=1, finite=False)
        if lower.shape != upper.shape or np.any(lower > upper):
            raise ValueError("interval endpoints must have matching shapes and be ordered")
        v.choice(kind, ("confidence", "credible"), "interval kind")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "level", v.level(level))
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "names", v.names(names, len(lower)))


@dataclass(frozen=True)
class BootstrapResult:
    estimate: Estimate
    interval: Interval
    draws: NDArray[Any]

    def __post_init__(self):
        if not isinstance(self.estimate, Estimate) or not isinstance(self.interval, Interval):
            raise TypeError("bootstrap requires an Estimate and Interval")
        draws = v.array(self.draws, "bootstrap draws", ndim=2)
        if (draws.shape[0] < 2 or draws.shape[1] != len(self.estimate.parameters)
                or self.estimate.names != self.interval.names):
            raise ValueError("bootstrap draws must have shape (resamples,p) aligned with intervals")
        object.__setattr__(self, "draws", draws)
