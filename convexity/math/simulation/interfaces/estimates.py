"""Estimated integration error is separate from path dispersion."""

from __future__ import annotations

from dataclasses import dataclass

from numpy.typing import ArrayLike

from ..._typing import FloatArray
from .. import _validation as v


@dataclass(frozen=True)
class MonteCarloEstimate:
    """Monte Carlo estimate with sampling uncertainty and retained simulation diagnostics."""

    value: FloatArray
    standard_error: FloatArray | None
    independent_units: int
    method: str

    def __post_init__(self) -> None:
        value = v.array(self.value, "value")
        object.__setattr__(self, "value", value)
        if self.standard_error is not None:
            error = v.shaped(self.standard_error, value.shape, "standard_error")
            if (error < 0).any():
                raise ValueError("standard_error must be nonnegative")
            object.__setattr__(self, "standard_error", error)
        v.integer(self.independent_units, "independent_units")
        v.label(self.method, "method")

    def __init__(
        self,
        value: ArrayLike,
        standard_error: ArrayLike | None,
        independent_units: int,
        method: str,
    ) -> None:
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "standard_error", standard_error)
        object.__setattr__(self, "independent_units", independent_units)
        object.__setattr__(self, "method", method)
        self.__post_init__()
