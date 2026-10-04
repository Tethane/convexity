"""Real scalar roots and square systems of real nonlinear equations."""

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike

from .. import _validation as v
from .base import check_array_result


@dataclass(frozen=True)
class ScalarRootProblem:
    """Find f(x)=0 using a bracket or initial guesses (depending on the tool)."""

    function: Callable[[float], float]
    bracket: tuple[float, float] | None = None
    x0: float | None = None
    x1: float | None = None
    derivative: Callable[[float], float] | None = None

    def __post_init__(self) -> None:
        v.function(self.function)
        if self.bracket is not None:
            bracket = v.array(self.bracket, "bracket", ndim=1, real=True)
            if bracket.shape != (2,) or bracket[0] >= bracket[1]:
                raise ValueError("bracket must contain two increasing finite endpoints")
            object.__setattr__(self, "bracket", (float(bracket[0]), float(bracket[1])))
        for name in ("x0", "x1"):
            if getattr(self, name) is not None:
                object.__setattr__(self, name, v.scalar(getattr(self, name), name))
        if self.bracket is None and self.x0 is None:
            raise ValueError("supply a bracket or x0")
        if self.derivative is not None:
            v.function(self.derivative, "derivative")

    def validate_result(self, result: object) -> None:
        check_array_result(result, (), real=True)


@dataclass(frozen=True)
class RootSystem:
    """f: R^n -> R^n, starting at x0; optional Jacobian has shape (n, n)."""

    function: Callable[..., ArrayLike]
    x0: ArrayLike
    jacobian: Callable[..., ArrayLike] | None = None

    def __post_init__(self) -> None:
        v.function(self.function)
        object.__setattr__(self, "x0", v.array(self.x0, "x0", ndim=1, real=True))
        if self.jacobian is not None:
            v.function(self.jacobian, "jacobian")

    def validate_result(self, result: object) -> None:
        check_array_result(result, np.shape(self.x0), real=True)
