"""Statistical diagnostics state their null hypothesis and uncertainty method."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol

from numpy.typing import ArrayLike

from ... import _decision as v
from ..._typing import FloatArray


@dataclass(frozen=True)
class TestResult:
    """Statistical test output with statistic, p-value and method information; interpret under its null assumptions."""

    statistic: FloatArray
    pvalue: FloatArray
    method: str
    null: str
    details: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        s, p = (
            v.array(self.statistic, "statistic", finite=False),
            v.array(self.pvalue, "pvalue"),
        )
        if s.shape != p.shape or (p < 0).any() or (p > 1).any():
            raise ValueError("statistics and pvalues must align; pvalues in [0,1]")
        object.__setattr__(self, "statistic", s)
        object.__setattr__(self, "pvalue", p)
        object.__setattr__(self, "details", MappingProxyType(dict(self.details or {})))

    def __init__(
        self,
        statistic: ArrayLike,
        pvalue: ArrayLike,
        method: str,
        null: str,
        details: Mapping[str, object] | None = None,
    ) -> None:
        object.__setattr__(self, "statistic", statistic)
        object.__setattr__(self, "pvalue", pvalue)
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "null", null)
        object.__setattr__(self, "details", details)
        self.__post_init__()


@dataclass(frozen=True)
class BootstrapResult:
    """Bootstrap statistic replicates and uncertainty summaries under the requested resampling scheme."""

    estimate: FloatArray
    samples: FloatArray
    lower: FloatArray
    upper: FloatArray
    level: float
    block_length: int

    def __post_init__(self) -> None:
        for name in ("estimate", "samples", "lower", "upper"):
            object.__setattr__(self, name, v.array(getattr(self, name), name))
        if (
            self.samples.shape[1:] != self.estimate.shape
            or self.lower.shape != self.estimate.shape
            or self.upper.shape != self.estimate.shape
        ):
            raise ValueError("bootstrap statistic axes must align")
        v.probability(self.level, "level", open_interval=True)
        v.integer(self.block_length, "block length", 1)

    def __init__(
        self,
        estimate: ArrayLike,
        samples: ArrayLike,
        lower: ArrayLike,
        upper: ArrayLike,
        level: float,
        block_length: int,
    ) -> None:
        object.__setattr__(self, "estimate", estimate)
        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "level", level)
        object.__setattr__(self, "block_length", block_length)
        self.__post_init__()


class Score(Protocol):
    """Evaluate predictions against observations using the declared statistical scoring convention."""

    def __call__(self, prediction: ArrayLike, actual: ArrayLike) -> float: ...
