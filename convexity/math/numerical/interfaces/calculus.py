"""Quadrature and differentiation contracts for scalar/vector functions."""

from dataclasses import dataclass
from typing import Any, Callable, Literal

import numpy as np
from numpy.typing import ArrayLike

from .. import _validation as v
from .base import ArrayResult, check_array_result

DifferentiationKind = Literal["derivative", "gradient", "jacobian", "hessian"]


@dataclass(frozen=True)
class QuadratureProblem:
    """Integral of a real scalar function over [a, b]; infinities are allowed."""

    function: Callable[[float], float]
    a: float
    b: float

    def __post_init__(self) -> None:
        v.function(self.function)
        a = v.scalar(self.a, "a", finite=False)
        b = v.scalar(self.b, "b", finite=False)
        if a == b and not np.isfinite(a):
            raise ValueError("identical infinite bounds do not define an integral")
        object.__setattr__(self, "a", a)
        object.__setattr__(self, "b", b)

    def validate_result(self, result: Any) -> None:
        check_array_result(result, (), real=True)


@dataclass(frozen=True)
class DifferentiationProblem:
    """One evaluation point; scalar for derivative, (n,) for other operations.

    NumPy callbacks consume one point at a time. Autodiff callbacks consume a
    PyTorch tensor of the same shape and must return a tensor. A Jacobian's
    axes are (output, input); Hessians and gradients require scalar output.
    """

    function: Callable[..., Any]
    x: ArrayLike
    kind: DifferentiationKind = "gradient"

    def __post_init__(self) -> None:
        v.function(self.function)
        v.choice(self.kind, ("derivative", "gradient", "jacobian", "hessian"), "kind")
        x = v.array(self.x, "x", ndim=0 if self.kind == "derivative" else 1, real=True)
        object.__setattr__(self, "x", x)

    def validate_result(self, result: Any) -> None:
        shape = np.shape(self.x)
        if self.kind == "jacobian":
            if not isinstance(result, ArrayResult):
                raise TypeError("tool must return ArrayResult")
            if result.value.ndim != 2 or result.value.shape[1:] != shape:
                raise ValueError("Jacobian must have shape (outputs, inputs)")
            check_array_result(result, result.value.shape, real=True)
        else:
            check_array_result(result, shape + shape if self.kind == "hessian" else shape,
                               real=True)
