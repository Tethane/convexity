"""Built-in continuous/discrete Lyapunov and algebraic Riccati tools."""

from dataclasses import dataclass

from numpy.typing import ArrayLike

from .._backends import matrix_equations
from ..interfaces import ArrayResult, LyapunovEquation, NumericalTool, RiccatiEquation, TimeDomain, run


@dataclass(frozen=True)
class LyapunovSolver:
    def __call__(self, problem: LyapunovEquation) -> ArrayResult:
        return matrix_equations.lyapunov(problem)


@dataclass(frozen=True)
class RiccatiSolver:
    def __call__(self, problem: RiccatiEquation) -> ArrayResult:
        return matrix_equations.riccati(problem)


def solve_lyapunov(a: ArrayLike, q: ArrayLike, *, time: TimeDomain = "continuous",
                   tool: NumericalTool[LyapunovEquation, ArrayResult] | None = None,
                   check_convergence: bool = True) -> ArrayResult:
    """Solve AX+XA^H+Q=0 (continuous) or AXA^H-X+Q=0 (discrete)."""
    return run(LyapunovEquation(a, q, time), tool if tool is not None else LyapunovSolver(),
               check_convergence=check_convergence)


def solve_riccati(a: ArrayLike, b: ArrayLike, q: ArrayLike, r: ArrayLike, *,
                  time: TimeDomain = "continuous",
                  tool: NumericalTool[RiccatiEquation, ArrayResult] | None = None,
                  check_convergence: bool = True) -> ArrayResult:
    """Solve the standard CARE or DARE; return X and its equation residual norm."""
    return run(RiccatiEquation(a, b, q, r, time), tool if tool is not None else RiccatiSolver(),
               check_convergence=check_convergence)
