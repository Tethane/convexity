from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from numpy.typing import ArrayLike

from ..._typing import FloatArray
from .. import _validation as v
from .._backends import fixed_point as backend
from ..interfaces import ArrayResult, FixedPointProblem, NumericalTool
from ..interfaces.base import run


@dataclass(frozen=True)
class FixedPointSolver:
    """Iterate a vector map toward x = f(x); termination depends on the configured tolerance."""

    method: str = "del2"
    xtol: float = 1e-10
    residual_tol: float = 1e-8
    maxiter: int = 500

    def __post_init__(self) -> None:
        v.choice(self.method, ("del2", "iteration"), "method")
        v.positive(self.xtol, "xtol")
        v.positive(self.residual_tol, "residual_tol")
        v.integer(self.maxiter, "maxiter")

    def __call__(self, problem: FixedPointProblem):
        return backend.solve(
            problem,
            method=self.method,
            xtol=self.xtol,
            residual_tol=self.residual_tol,
            maxiter=self.maxiter,
        )


def fixed_point(
    function: Callable[[FloatArray], ArrayLike],
    x0: ArrayLike,
    *,
    tool: NumericalTool[FixedPointProblem, ArrayResult] | None = None,
    check_convergence: bool = True,
) -> ArrayResult:
    """Solve x = function(x) from scalar or vector x0.

    The callback must preserve shape. Return ArrayResult with convergence diagnostics;
    check_convergence=True raises ConvergenceError on reported failure."""
    return run(
        FixedPointProblem(function, x0),
        tool if tool is not None else FixedPointSolver(),
        check_convergence=check_convergence,
    )


__all__ = ["FixedPointSolver", "fixed_point"]
