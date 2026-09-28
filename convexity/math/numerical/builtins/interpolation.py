"""Canonical linear, cubic, and shape-preserving PCHIP interpolation."""

from dataclasses import dataclass
from typing import Literal

from numpy.typing import ArrayLike

from .. import _validation as v
from .._backends import interpolation
from ..interfaces import Interpolant, InterpolationProblem, NumericalTool, run


@dataclass(frozen=True)
class Interpolation:
    """Cubic uses not-a-knot boundaries; PCHIP preserves monotone sample shape."""

    method: Literal["linear", "cubic", "pchip"] = "linear"

    def __post_init__(self) -> None:
        v.choice(self.method, ("linear", "cubic", "pchip"), "method")

    def __call__(self, problem: InterpolationProblem) -> Interpolant:
        return interpolation.fit(problem, method=self.method)


def interpolate(x: ArrayLike, y: ArrayLike, *, extrapolate: bool = False,
                tool: NumericalTool[InterpolationProblem, Interpolant] | None = None) -> Interpolant:
    """Fit a callable interpolant; out-of-domain queries raise unless opted in."""
    return run(InterpolationProblem(x, y, extrapolate), tool if tool is not None else Interpolation())
