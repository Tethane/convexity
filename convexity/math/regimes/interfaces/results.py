"""Owned regime probabilities and marked event realizations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import ArrayLike

from ..._typing import FloatArray, IndexArray, IntArray
from .. import _validation as v


@dataclass(frozen=True)
class RegimeResult:
    """Rows are observation times, columns are integer regime labels.

    predicted uses observations strictly before the row; filtered includes that
    row; smoothed uses the entire supplied sample. Labels have no imposed order.
    """

    predicted: FloatArray
    filtered: FloatArray
    log_likelihoods: FloatArray
    smoothed: FloatArray | None = cast("FloatArray | None", None)
    pair_probabilities: FloatArray | None = cast("FloatArray | None", None)
    timestamps: IndexArray | None = cast("IndexArray | None", None)

    def __post_init__(self) -> None:
        if self.predicted is None or self.filtered is None:
            raise ValueError("predicted and filtered probabilities are required")
        for name in ("predicted", "filtered", "smoothed"):
            x = getattr(self, name)
            if x is not None:
                x = v.probabilities(x, name, ndim=2)
                if name != "predicted" and x.shape != self.predicted.shape:
                    raise ValueError("regime probabilities must have matching shapes")
                object.__setattr__(self, name, x)
        n, k = self.filtered.shape
        ll = v.array(self.log_likelihoods, "log likelihoods", ndim=1)
        if ll.shape != (n,):
            raise ValueError("log likelihoods must align with probability rows")
        object.__setattr__(self, "log_likelihoods", ll)
        if self.pair_probabilities is not None:
            pairs = v.array(self.pair_probabilities, "pair probabilities", empty=True)
            if (
                self.smoothed is None
                or pairs.shape != (n - 1, k, k)
                or np.any(pairs < 0)
                or not np.allclose(pairs.sum(axis=2), self.smoothed[:-1])
                or not np.allclose(pairs.sum(axis=1), self.smoothed[1:])
            ):
                raise ValueError(
                    "pair probabilities must agree with adjacent smoothed marginals"
                )
            object.__setattr__(self, "pair_probabilities", pairs)
        if self.timestamps is not None:
            from ...forecasting._validation import index

            object.__setattr__(self, "timestamps", index(self.timestamps, n))

    @property
    def log_likelihood(self) -> float:
        return float(self.log_likelihoods.sum())

    def distribution(self, index, *, kind: str = "filtered"):
        from ...distributions import empirical

        v.choice(kind, ("predicted", "filtered", "smoothed"), "kind")
        index = v.state(index, len(self.filtered))
        values = getattr(self, kind)
        if values is None:
            raise ValueError(f"{kind} probabilities are unavailable")
        return empirical(np.arange(values.shape[1]), values[index])

    def __init__(
        self,
        predicted: ArrayLike,
        filtered: ArrayLike,
        log_likelihoods: ArrayLike,
        smoothed: ArrayLike | None = None,
        pair_probabilities: ArrayLike | None = None,
        timestamps: ArrayLike | None = None,
    ) -> None:
        object.__setattr__(self, "predicted", predicted)
        object.__setattr__(self, "filtered", filtered)
        object.__setattr__(self, "log_likelihoods", log_likelihoods)
        object.__setattr__(self, "smoothed", smoothed)
        object.__setattr__(self, "pair_probabilities", pair_probabilities)
        object.__setattr__(self, "timestamps", timestamps)
        self.__post_init__()


@dataclass(frozen=True)
class RegimePaths:
    """Joint draws: regimes (draw,time), values (draw,time,*event_shape)."""

    regimes: IntArray
    values: FloatArray

    def __post_init__(self) -> None:
        r = v.array(self.regimes, "regimes", ndim=2)
        x = v.array(self.values, "values")
        if (
            np.any(r < 0)
            or np.any(r != np.floor(r))
            or x.ndim not in (2, 3)
            or x.shape[:2] != r.shape
        ):
            raise ValueError(
                "integer regime paths and values must align on draw,time axes"
            )
        r = r.astype(np.int64)
        r.setflags(write=False)
        object.__setattr__(self, "regimes", r)
        object.__setattr__(self, "values", x)

    def distribution(self):
        from ...distributions import empirical

        return empirical(self.values.reshape(len(self.values), -1))

    def __init__(
        self,
        regimes: ArrayLike,
        values: ArrayLike,
    ) -> None:
        object.__setattr__(self, "regimes", regimes)
        object.__setattr__(self, "values", values)
        self.__post_init__()


@dataclass(frozen=True)
class MarkedEvents:
    """A realization on (start,end], with scalar/vector marks at ordered times."""

    times: FloatArray
    marks: FloatArray
    start: float
    end: float

    def __post_init__(self) -> None:
        start, end = v.interval(self.start, self.end)
        times = v.times(self.times, strict=False)
        marks = v.array(self.marks, "marks", empty=True)
        if (
            marks.ndim not in (1, 2)
            or len(marks) != len(times)
            or (marks.ndim == 2 and marks.shape[1] == 0)
            or np.any(times <= start)
            or np.any(times > end)
        ):
            raise ValueError("marks must align with times in (start,end]")
        for name, value in (
            ("times", times),
            ("marks", marks),
            ("start", start),
            ("end", end),
        ):
            object.__setattr__(self, name, value)

    @property
    def total(self) -> FloatArray:
        return v.array(self.marks.sum(axis=0), "total")

    def __init__(
        self,
        times: ArrayLike,
        marks: ArrayLike,
        start: float,
        end: float,
    ) -> None:
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "marks", marks)
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)
        self.__post_init__()
