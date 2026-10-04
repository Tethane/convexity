from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from ..._typing import FloatArray
from ..interfaces import (
    BoundaryValueProblem,
    BoundaryValueResult,
    InitialValueProblem,
    NumericalTool,
    ODEResult,
)

if TYPE_CHECKING:
    from .. import ODEEvent

from dataclasses import dataclass

from .. import _validation as v
from .._backends import ode as backend
from ..interfaces.base import run


@dataclass(frozen=True)
class IVPSolver:
    """Integrate dy/dt = f(t,y) with adaptive numerical tolerances and optional event termination."""

    method: str = "RK45"
    atol: float = 1e-9
    rtol: float = 1e-7
    max_step: float = np.inf

    def __post_init__(self) -> None:
        v.choice(
            self.method, ("RK23", "RK45", "DOP853", "Radau", "BDF", "LSODA"), "method"
        )
        v.tolerances(self.atol, self.rtol)
        if self.atol <= 0 or self.rtol < 100 * np.finfo(float).eps:
            raise ValueError(
                "ODE atol must be positive and rtol >= 100 * machine epsilon"
            )
        if v.scalar(self.max_step, "max_step", finite=False) <= 0:
            raise ValueError("max_step must be positive")

    def __call__(self, problem: InitialValueProblem):
        return backend.solve_ivp(
            problem,
            method=self.method,
            atol=self.atol,
            rtol=self.rtol,
            max_step=self.max_step,
        )


@dataclass(frozen=True)
class BVPSolver:
    """Solve a vector boundary-value problem on an initial mesh; inspect residuals and convergence."""

    tolerance: float = 1e-6
    boundary_tolerance: float = 1e-8
    max_nodes: int = 1000

    def __post_init__(self) -> None:
        v.positive(self.tolerance, "tolerance")
        v.positive(self.boundary_tolerance, "boundary_tolerance")
        v.integer(self.max_nodes, "max_nodes")

    def __call__(self, problem: BoundaryValueProblem):
        if self.max_nodes < np.size(problem.x):
            raise ValueError("max_nodes must accommodate the initial mesh")
        return backend.solve_bvp(
            problem,
            tolerance=self.tolerance,
            boundary_tolerance=self.boundary_tolerance,
            max_nodes=self.max_nodes,
        )


def solve_ivp(
    function: Callable[[float, FloatArray], ArrayLike],
    y0: ArrayLike,
    t_span: tuple[float, float],
    *,
    t_eval: ArrayLike | None = None,
    events: Sequence[ODEEvent] = (),
    jacobian: Callable[[float, FloatArray], ArrayLike] | None = None,
    tool: NumericalTool[InitialValueProblem, ODEResult] | None = None,
    check_convergence: bool = True,
) -> ODEResult:
    """Integrate y'=function(t,y), where y is a state vector.

    Return times (t,), states (state,t), dense solution and event records.
    Event states use (event,state). t_eval follows the integration direction;
    terminal events may stop before t_span ends. Callback units must be consistent."""
    return run(
        InitialValueProblem(function, y0, t_span, t_eval, tuple(events), jacobian),
        tool if tool is not None else IVPSolver(),
        check_convergence=check_convergence,
    )


def solve_bvp(
    function: Callable[[FloatArray, FloatArray], ArrayLike],
    boundary: Callable[[FloatArray, FloatArray], ArrayLike],
    x: ArrayLike,
    y: ArrayLike,
    *,
    tool: NumericalTool[BoundaryValueProblem, BoundaryValueResult] | None = None,
    check_convergence: bool = True,
) -> BoundaryValueResult:
    """Solve y'=function(x,y), boundary(y_left,y_right)=0 on an increasing mesh.

    Callbacks use y (state,mesh), boundary returns (state,). Return an adaptive
    mesh, solution, residuals and convergence diagnostics."""
    return run(
        BoundaryValueProblem(function, boundary, x, y),
        tool if tool is not None else BVPSolver(),
        check_convergence=check_convergence,
    )


def solve_ode(
    function: Callable[[float, FloatArray], ArrayLike],
    y0: ArrayLike,
    t: ArrayLike,
    *,
    tool: NumericalTool[InitialValueProblem, ODEResult] | None = None,
    events: Sequence[ODEEvent] = (),
    jacobian: Callable[[float, FloatArray], ArrayLike] | None = None,
    check_convergence: bool = True,
) -> ODEResult:
    """Convenience IVP on a supplied increasing or decreasing time grid."""
    times = v.array(t, "t", ndim=1, real=True)
    if times.size < 2:
        raise ValueError("supply at least two output times")
    return solve_ivp(
        function,
        y0,
        (times[0], times[-1]),
        t_eval=times,
        events=events,
        jacobian=jacobian,
        tool=tool,
        check_convergence=check_convergence,
    )


__all__ = ["IVPSolver", "BVPSolver", "solve_ivp", "solve_bvp", "solve_ode"]
