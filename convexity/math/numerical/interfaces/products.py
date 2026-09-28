from dataclasses import dataclass
from typing import Any, Callable
import numpy as np
from .. import _validation as v
from .base import ArrayResult, check_array_result


@dataclass(frozen=True)
class DerivativeProductProblem:
    function: Callable
    x: Any
    vector: Any
    kind: str = "jvp"

    def __post_init__(self):
        v.function(self.function)
        v.choice(self.kind, ("jvp", "vjp"), "kind")
        x, vector = v.array(self.x, "x", ndim=1, real=True), v.array(self.vector, "vector", ndim=1, real=True)
        if self.kind == "jvp" and vector.shape != x.shape:
            raise ValueError("JVP vector must have the input shape")
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "vector", vector)

    def validate_result(self, result):
        if self.kind == "vjp":
            check_array_result(result, np.shape(self.x), real=True)
        elif not isinstance(result, ArrayResult) or result.value.ndim != 1 or np.iscomplexobj(result.value):
            raise ValueError("JVP tool must return a real vector ArrayResult")


@dataclass(frozen=True)
class DerivativeCheck:
    """Comparison to a converged finite-difference reference, component by component."""
    passed: bool
    actual: Any
    reference: Any
    absolute_error: Any
    scaled_error: Any

    def __post_init__(self):
        v.boolean(self.passed, "passed")
        for name in ("actual", "reference", "absolute_error", "scaled_error"):
            object.__setattr__(self, name, v.array(getattr(self, name), name, real=True))
        if any(getattr(self, name).shape != self.actual.shape for name in ("reference", "absolute_error", "scaled_error")):
            raise ValueError("derivative check arrays must have equal shapes")
