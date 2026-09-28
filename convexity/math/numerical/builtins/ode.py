from dataclasses import dataclass
import numpy as np
from .. import _validation as v
from .._backends import ode as backend
from ..interfaces.base import run
from ..interfaces.ode import InitialValueProblem, BoundaryValueProblem


@dataclass(frozen=True)
class IVPSolver:
    method: str = "RK45"
    atol: float = 1e-9
    rtol: float = 1e-7
    max_step: float = np.inf

    def __post_init__(self):
        v.choice(self.method, ("RK23", "RK45", "DOP853", "Radau", "BDF", "LSODA"), "method")
        v.tolerances(self.atol, self.rtol)
        if self.atol <= 0 or self.rtol < 100*np.finfo(float).eps:
            raise ValueError("ODE atol must be positive and rtol >= 100 * machine epsilon")
        if v.scalar(self.max_step, "max_step", finite=False) <= 0:
            raise ValueError("max_step must be positive")

    def __call__(self, problem: InitialValueProblem):
        return backend.solve_ivp(problem, method=self.method, atol=self.atol, rtol=self.rtol, max_step=self.max_step)


@dataclass(frozen=True)
class BVPSolver:
    tolerance: float = 1e-6
    boundary_tolerance: float = 1e-8
    max_nodes: int = 1000

    def __post_init__(self):
        v.positive(self.tolerance, "tolerance")
        v.positive(self.boundary_tolerance, "boundary_tolerance")
        v.integer(self.max_nodes, "max_nodes")

    def __call__(self, problem: BoundaryValueProblem):
        if self.max_nodes < np.size(problem.x):
            raise ValueError("max_nodes must accommodate the initial mesh")
        return backend.solve_bvp(problem, tolerance=self.tolerance,
                                 boundary_tolerance=self.boundary_tolerance, max_nodes=self.max_nodes)


def solve_ivp(function, y0, t_span, *, t_eval=None, events=(), jacobian=None, tool=None, check_convergence=True):
    return run(InitialValueProblem(function, y0, t_span, t_eval, events, jacobian),
               tool if tool is not None else IVPSolver(), check_convergence=check_convergence)


def solve_bvp(function, boundary, x, y, *, tool=None, check_convergence=True):
    return run(BoundaryValueProblem(function, boundary, x, y),
               tool if tool is not None else BVPSolver(), check_convergence=check_convergence)


def solve_ode(function, y0, t, *, tool=None, events=(), jacobian=None, check_convergence=True):
    """Convenience IVP on a supplied increasing or decreasing time grid."""
    times = v.array(t, "t", ndim=1, real=True)
    if times.size < 2:
        raise ValueError("supply at least two output times")
    return solve_ivp(function, y0, (times[0], times[-1]), t_eval=times, events=events,
                     jacobian=jacobian, tool=tool, check_convergence=check_convergence)


__all__ = ["IVPSolver", "BVPSolver", "solve_ivp", "solve_bvp", "solve_ode"]
