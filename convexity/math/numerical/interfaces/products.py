from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..._provider_types import NativeTensor

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike

from ..._typing import (
    FloatArray,
)
from .. import _validation as v
from .base import ArrayResult, check_array_result


@dataclass(frozen=True)
class DerivativeProductProblem:
    """Callback, evaluation point and direction for a JVP or VJP."""

    function: Callable[[FloatArray], ArrayLike] | Callable[[NativeTensor], NativeTensor]
    x: FloatArray
    vector: FloatArray
    kind: str = "jvp"

    def __post_init__(self) -> None:
        v.function(self.function)
        v.choice(self.kind, ("jvp", "vjp"), "kind")
        x, vector = (
            v.array(self.x, "x", ndim=1, real=True),
            v.array(self.vector, "vector", ndim=1, real=True),
        )
        if self.kind == "jvp" and vector.shape != x.shape:
            raise ValueError("JVP vector must have the input shape")
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "vector", vector)

    def validate_result(self, result: object) -> None:
        if self.kind == "vjp":
            check_array_result(result, np.shape(self.x), real=True)
        elif (
            not isinstance(result, ArrayResult)
            or result.value.ndim != 1
            or np.iscomplexobj(result.value)
        ):
            raise ValueError("JVP tool must return a real vector ArrayResult")

    def __init__(
        self,
        function: Callable[[FloatArray], ArrayLike] | Callable[[NativeTensor], NativeTensor],
        x: ArrayLike,
        vector: ArrayLike,
        kind: str = "jvp",
    ) -> None:
        object.__setattr__(self, "function", function)
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "vector", vector)
        object.__setattr__(self, "kind", kind)
        self.__post_init__()


@dataclass(frozen=True)
class DerivativeCheck:
    """Comparison to a converged finite-difference reference, component by component."""

    passed: bool
    actual: FloatArray
    reference: FloatArray
    absolute_error: FloatArray
    scaled_error: FloatArray

    def __post_init__(self) -> None:
        v.boolean(self.passed, "passed")
        for name in ("actual", "reference", "absolute_error", "scaled_error"):
            object.__setattr__(
                self, name, v.array(getattr(self, name), name, real=True)
            )
        if any(
            getattr(self, name).shape != self.actual.shape
            for name in ("reference", "absolute_error", "scaled_error")
        ):
            raise ValueError("derivative check arrays must have equal shapes")

    def __init__(
        self,
        passed: bool,
        actual: ArrayLike,
        reference: ArrayLike,
        absolute_error: ArrayLike,
        scaled_error: ArrayLike,
    ) -> None:
        object.__setattr__(self, "passed", passed)
        object.__setattr__(self, "actual", actual)
        object.__setattr__(self, "reference", reference)
        object.__setattr__(self, "absolute_error", absolute_error)
        object.__setattr__(self, "scaled_error", scaled_error)
        self.__post_init__()
