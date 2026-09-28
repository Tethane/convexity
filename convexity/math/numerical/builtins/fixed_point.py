from dataclasses import dataclass
from .. import _validation as v
from .._backends import fixed_point as backend
from ..interfaces.base import run
from ..interfaces.fixed_point import FixedPointProblem


@dataclass(frozen=True)
class FixedPointSolver:
    method: str = "del2"
    xtol: float = 1e-10
    residual_tol: float = 1e-8
    maxiter: int = 500

    def __post_init__(self):
        v.choice(self.method, ("del2", "iteration"), "method")
        v.positive(self.xtol, "xtol")
        v.positive(self.residual_tol, "residual_tol")
        v.integer(self.maxiter, "maxiter")

    def __call__(self, problem: FixedPointProblem):
        return backend.solve(problem, method=self.method, xtol=self.xtol,
                             residual_tol=self.residual_tol, maxiter=self.maxiter)


def fixed_point(function, x0, *, tool=None, check_convergence=True):
    return run(FixedPointProblem(function, x0), tool if tool is not None else FixedPointSolver(),
               check_convergence=check_convergence)


__all__ = ["FixedPointSolver", "fixed_point"]
