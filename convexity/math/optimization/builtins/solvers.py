"""Built-in solvers and bounded exact enumeration."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING

from numpy.typing import ArrayLike

from ..._typing import FloatArray

if TYPE_CHECKING:
    from .. import SolveResult

if TYPE_CHECKING:
    from .. import NonlinearProblem, QuadraticProgram, Solver


from dataclasses import dataclass, field
from itertools import product
from math import prod

import numpy as np

from ... import _decision as v
from ..interfaces import (
    NonlinearProblem,
    OptimizationError,
    QuadraticProgram,
    SolveResult,
)


@dataclass(frozen=True)
class ConvexSolver:
    """Solve a declared convex problem with the selected provider solver and options."""

    solver: str | None = None
    options: Mapping[str, object] = field(default_factory=dict)
    tolerance: float = 1e-6

    def solve(self, problem):
        from .._backends.cvxpy import solve

        if not isinstance(problem, QuadraticProgram):
            raise TypeError("ConvexSolver requires QuadraticProgram")
        tol = v.scalar(self.tolerance, "tolerance", minimum=0, strict=True)
        return solve(problem, self.solver, dict(self.options), tol)


@dataclass(frozen=True)
class NonlinearSolver:
    """Numerically optimize a nonlinear objective from a supplied initial point, with explicit constraints and tolerances."""

    tolerance: float = 1e-9
    maxiter: int = 1000

    def solve(self, problem):
        from .._backends.scipy import solve

        if not isinstance(problem, NonlinearProblem):
            raise TypeError("NonlinearSolver requires NonlinearProblem")
        return solve(
            problem,
            v.scalar(self.tolerance, "tolerance", minimum=0, strict=True),
            v.integer(self.maxiter, "maxiter", 1),
        )


def solve[ProblemT](
    problem: ProblemT,
    solver: Solver[ProblemT] | None = None,
    *,
    check_success: bool = True,
) -> SolveResult:
    """Solve a QuadraticProgram or NonlinearProblem using a built-in or custom Solver.

    Return SolveResult with status, feasibility and optional primal solution.
    check_success=True raises on failed solves; a failed result may have solution=None."""
    if solver is None:
        solver = (
            NonlinearSolver()
            if isinstance(problem, NonlinearProblem)
            else ConvexSolver()
        )
    result = solver.solve(problem)
    if not isinstance(result, SolveResult):
        raise TypeError("solver must return SolveResult")
    if check_success and not result.success:
        raise OptimizationError(result)
    return result


def enumerate_minimum(
    objective: Callable[[FloatArray], float],
    domains: Sequence[ArrayLike],
    *,
    feasible: Callable[[FloatArray], bool] | None = None,
    max_candidates: int = 100000,
) -> SolveResult:
    """Exact Cartesian finite search; first candidate wins ties; bounded upfront."""
    domains = tuple(v.array(d, "domain", ndim=1) for d in domains)
    if not domains or prod(map(len, domains)) > v.integer(
        max_candidates, "max_candidates", 1
    ):
        raise ValueError("empty problem or enumeration exceeds max_candidates")
    best, value, count = None, np.inf, 0
    for row in product(*domains):
        x = v.array(row, "candidate")
        count += 1
        if feasible is not None and not v.boolean(feasible(x), "feasibility"):
            continue
        fx = v.scalar(objective(x), "objective")
        if fx < value:
            best, value = x, fx
    if best is None:
        return SolveResult(None, None, "infeasible", "enumeration", iterations=count)
    return SolveResult(
        best, float(value), "optimal", "enumeration", 0.0, gap=0.0, iterations=count
    )
