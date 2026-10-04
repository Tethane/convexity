"""Owned joint driver and finite-memory results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import ArrayLike

from ..._typing import (
    FloatArray,
)
from ...processes import ProcessPaths
from .. import _validation as v


@dataclass(frozen=True)
class VolterraPaths:
    """Joint X(t) and the increments of the same driving Brownian motion.

    factors, when present, have shape (draw,time,factor). Exactness refers to
    the supplied kernel, never to a different kernel it may approximate.
    """

    times: FloatArray
    values: FloatArray
    brownian_increments: FloatArray
    method: str
    exact: bool
    factors: FloatArray | None = cast("FloatArray | None", None)
    backend: str = "numpy/scipy"

    def __post_init__(self) -> None:
        times = v.times(self.times)
        values = v.array(self.values, "values", ndim=2)
        if (
            values.shape[0] < 1
            or values.shape[1] != len(times)
            or np.any(values[:, 0] != 0)
        ):
            raise ValueError("values must have shape (draw,time) and start at zero")
        increments = v.shaped(
            self.brownian_increments,
            (len(values), len(times) - 1),
            "brownian_increments",
        )
        if self.factors is not None:
            factors = v.array(self.factors, "factors", ndim=3)
            if factors.shape[:2] != values.shape or np.any(factors[:, 0] != 0):
                raise ValueError(
                    "factors must have shape (draw,time,factor) and start at zero"
                )
            object.__setattr__(self, "factors", factors)
        v.boolean(self.exact, "exact")
        v.label(self.method, "method")
        v.label(self.backend, "backend")
        for name, value in (
            ("times", times),
            ("values", values),
            ("brownian_increments", increments),
        ):
            object.__setattr__(self, name, value)

    def paths(self) -> ProcessPaths:
        return ProcessPaths(
            self.times, self.values, self.method, self.backend, self.exact
        )

    def __init__(
        self,
        times: ArrayLike,
        values: ArrayLike,
        brownian_increments: ArrayLike,
        method: str,
        exact: bool,
        factors: ArrayLike | None = None,
        backend: str = "numpy/scipy",
    ) -> None:
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "brownian_increments", brownian_increments)
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "exact", exact)
        object.__setattr__(self, "factors", factors)
        object.__setattr__(self, "backend", backend)
        self.__post_init__()


@dataclass(frozen=True)
class KernelError:
    """Integrated squared kernel error equals coupled terminal driver MSE."""

    horizon: float
    squared_error: float
    target_energy: float
    quadrature_error: float

    def __post_init__(self) -> None:
        for name in ("horizon", "squared_error", "target_energy", "quadrature_error"):
            object.__setattr__(
                self,
                name,
                v.scalar(
                    getattr(self, name), name, minimum=0, strict=name == "horizon"
                ),
            )

    @property
    def l2_error(self) -> float:
        return float(np.sqrt(self.squared_error))

    @property
    def relative_l2_error(self):
        if self.target_energy == 0:
            return 0.0 if self.squared_error == 0 else float("inf")
        return float(np.sqrt(self.squared_error / self.target_energy))

    def require(self, tolerance: float):
        tolerance = v.scalar(tolerance, "tolerance", minimum=0)
        if self.relative_l2_error > tolerance:
            raise ValueError(
                f"relative L2 kernel error {self.relative_l2_error:.6g} exceeds {tolerance:.6g}"
            )
        return self
