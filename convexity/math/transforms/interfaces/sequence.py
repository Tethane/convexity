"""Aligned feature matrices and the boundary conditions of differencing."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ..._typing import FloatArray, IntArray
from .. import _validation as v


@dataclass(frozen=True, init=False)
class FeatureMatrix:
    """Constructed feature rows with retained column names and row alignment."""

    values: FloatArray
    rows: IntArray
    columns: tuple[str, ...]

    def __init__(
        self, values: ArrayLike, rows: ArrayLike, columns: tuple[str, ...]
    ) -> None:
        x = v.samples(values)
        raw = np.asarray(rows)
        if (
            raw.dtype.kind not in "iu"
            or raw.shape != (len(x),)
            or np.any(raw < 0)
            or np.any(raw > np.iinfo(np.int64).max)
            or np.any(raw[1:] <= raw[:-1])
        ):
            raise ValueError(
                "rows must be strictly increasing nonnegative integer positions"
            )
        indices = np.array(raw, dtype=np.int64)
        indices.setflags(write=False)
        object.__setattr__(self, "values", x)
        object.__setattr__(self, "rows", indices)
        object.__setattr__(self, "columns", v.names(columns, x.shape[1]))


@dataclass(frozen=True, init=False)
class DifferenceResult:
    """D_lag**order x and leading/trailing seeds for every integration level."""

    values: FloatArray
    initial: tuple[FloatArray, ...]
    trailing: tuple[FloatArray, ...]
    lag: int

    def __init__(
        self,
        values: ArrayLike,
        initial: Sequence[ArrayLike],
        trailing: Sequence[ArrayLike],
        lag: int,
    ) -> None:
        delta = v.series(values)
        lag = v.integer(lag, "lag", 1)
        first = tuple(v.series(x) for x in initial)
        last = tuple(v.series(x) for x in trailing)
        if not first or len(first) != len(last):
            raise ValueError(
                "one initial and trailing block is required per difference order"
            )
        if any(x.shape != (lag,) + delta.shape[1:] for x in first + last):
            raise ValueError(
                "boundary blocks must have shape (lag,) + values.shape[1:]"
            )
        for name, value in dict(
            values=delta, initial=first, trailing=last, lag=lag
        ).items():
            object.__setattr__(self, name, value)

    @property
    def order(self):
        return len(self.initial)

    @property
    def row_offset(self):
        return self.lag * self.order

    @property
    def rows(self):
        result = np.arange(self.row_offset, self.row_offset + len(self.values))
        result.setflags(write=False)
        return result

    def inverse_transform(
        self, values: ArrayLike | None = None, *, continuation: bool = False
    ) -> FloatArray:
        """Restore a sequence, or integrate future differences using trailing seeds.

        Ordinary inversion includes initial rows. Continuation returns only
        new rows and never mutates the stored boundary conditions.
        """
        from .._backends.sequence import integrate

        v.boolean(continuation, "continuation")
        if continuation and values is None:
            raise ValueError("continuation requires explicit future differences")
        delta = self.values if values is None else v.series(values)
        if delta.shape[1:] != self.values.shape[1:]:
            raise ValueError(
                "difference coordinates must match the fitted boundary conditions"
            )
        boundaries = self.trailing if continuation else self.initial
        return v.array(
            integrate(delta, boundaries, continuation=continuation), "integrated values"
        )
