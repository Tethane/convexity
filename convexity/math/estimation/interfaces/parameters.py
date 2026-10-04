"""Named parameter coordinates and explicit parameter constraints."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import ArrayLike

from ..._typing import FloatArray
from .. import _validation as v


@dataclass(frozen=True)
class ParameterSpace:
    """A finite initial vector and closed box bounds (scalar or shape (p,)).

    Equal bounds are excluded: fix a parameter in the model callback instead.
    Positivity uses an explicit positive lower bound or log coordinates.
    """

    initial: FloatArray
    names: tuple[str, ...] = cast("tuple[str, ...]", None)
    lower: FloatArray = cast("FloatArray", -np.inf)
    upper: FloatArray = cast("FloatArray", np.inf)

    def __post_init__(self) -> None:
        initial = v.array(self.initial, "initial", ndim=1)
        lower, upper = v.bounds(self.lower, self.upper, len(initial), strict=True)
        if np.any((initial < lower) | (initial > upper)):
            raise ValueError("initial parameters must lie within bounds")
        for name, value in dict(
            initial=initial,
            lower=lower,
            upper=upper,
            names=v.names(self.names, len(initial)),
        ).items():
            object.__setattr__(self, name, value)

    @property
    def size(self):
        return np.size(self.initial)

    def point(self, value: ArrayLike):
        result = v.array(value, "parameters", ndim=1)
        if result.shape != (self.size,):
            raise ValueError(f"parameters must have shape ({self.size},)")
        return result

    def violation(self, value) -> float:
        point = self.point(value)
        return float(
            max(0.0, np.max(np.asarray(self.lower) - point), np.max(point - self.upper))
        )

    def at_boundary(self, value, tolerance: float = 1e-6) -> bool:
        point = self.point(value)
        scale = tolerance * (1 + np.abs(point))
        return bool(
            np.any((point - self.lower <= scale) | (self.upper - point <= scale))
        )

    def __init__(
        self,
        initial: ArrayLike,
        names: Sequence[str] | None = None,
        lower: ArrayLike = -float("inf"),
        upper: ArrayLike = float("inf"),
    ) -> None:
        object.__setattr__(self, "initial", initial)
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        self.__post_init__()


@dataclass(frozen=True)
class LinearConstraint:
    """lower <= matrix @ theta <= upper; equal endpoints impose equality."""

    matrix: FloatArray
    lower: FloatArray = cast("FloatArray", -np.inf)
    upper: FloatArray = cast("FloatArray", np.inf)

    def __post_init__(self) -> None:
        matrix = v.array(self.matrix, "constraint matrix", ndim=2)
        lower, upper = v.bounds(self.lower, self.upper, matrix.shape[0])
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    def evaluate(self, theta: ArrayLike):
        return np.asarray(self.matrix) @ v.array(theta, "parameters", ndim=1)

    def __init__(
        self,
        matrix: ArrayLike,
        lower: ArrayLike = -float("inf"),
        upper: ArrayLike = float("inf"),
    ) -> None:
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        self.__post_init__()


@dataclass(frozen=True)
class NonlinearConstraint:
    """lower <= function(theta) <= upper; function returns shape (m,)."""

    function: Callable[[FloatArray], ArrayLike]
    lower: FloatArray
    upper: FloatArray
    jacobian: Callable[[FloatArray], ArrayLike] | None = None

    def __post_init__(self) -> None:
        v.function(self.function, "constraint function")
        if self.jacobian is not None:
            v.function(self.jacobian, "constraint Jacobian")
        lo, hi = np.broadcast_arrays(
            v.array(self.lower, "lower", finite=False),
            v.array(self.upper, "upper", finite=False),
        )
        lower, upper = v.bounds(lo.reshape(-1), hi.reshape(-1), lo.size)
        if lo.ndim > 1:
            raise ValueError("constraint bounds must be scalars or vectors")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    def evaluate(self, theta: ArrayLike):
        value = v.array(
            self.function(v.array(theta, "parameters", ndim=1)),
            "constraint output",
            ndim=1,
        )
        if value.shape != np.shape(self.lower):
            raise ValueError("constraint output must match bounds")
        return value

    def derivative(self, theta: ArrayLike):
        assert self.jacobian is not None
        value = v.array(
            self.jacobian(v.array(theta, "parameters", ndim=1)),
            "constraint Jacobian",
            ndim=2,
        )
        if value.shape != (np.size(self.lower), np.size(theta)):
            raise ValueError("constraint Jacobian must have shape (m,p)")
        return value

    def __init__(
        self,
        function: Callable[[FloatArray], ArrayLike],
        lower: ArrayLike,
        upper: ArrayLike,
        jacobian: Callable[[FloatArray], ArrayLike] | None = None,
    ) -> None:
        object.__setattr__(self, "function", function)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "jacobian", jacobian)
        self.__post_init__()


def as_space(value):
    return value if isinstance(value, ParameterSpace) else ParameterSpace(value)
