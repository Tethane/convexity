"""Bracketed/Newton/secant scalar roots and Powell hybrid system roots."""

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
from numpy.typing import ArrayLike

from .. import _validation as v
from .._backends import roots
from ..interfaces import ArrayResult, NumericalTool, RootSystem, ScalarRootProblem, run


@dataclass(frozen=True)
class ScalarRootSolver:
    """Find a scalar zero using a bracket or initial values, with convergence diagnostics."""

    method: Literal["brentq", "bisect", "toms748", "newton", "secant"] = "brentq"
    xtol: float = 1e-12
    rtol: float = 1e-10
    maxiter: int = 100

    def __post_init__(self) -> None:
        v.choice(
            self.method, ("brentq", "bisect", "toms748", "newton", "secant"), "method"
        )
        v.positive(self.xtol, "xtol")
        v.positive(self.rtol, "rtol", zero=True)
        if (
            self.method in ("brentq", "bisect", "toms748")
            and self.rtol < 4 * np.finfo(float).eps
        ):
            raise ValueError("bracketed root rtol must be >= 4 * machine epsilon")
        v.integer(self.maxiter, "maxiter")

    def __call__(self, problem: ScalarRootProblem) -> ArrayResult:
        return roots.scalar_root(
            problem,
            method=self.method,
            xtol=self.xtol,
            rtol=self.rtol,
            maxiter=self.maxiter,
        )


@dataclass(frozen=True)
class SystemRootSolver:
    """Powell hybrid solver; require both solver success and a small residual."""

    xtol: float = 1e-10
    residual_tol: float = 1e-8
    max_evaluations: int = 1000
    method: str = "hybr"
    maxiter: int = 100

    def __post_init__(self) -> None:
        v.positive(self.xtol, "xtol")
        v.positive(self.residual_tol, "residual_tol")
        v.integer(self.max_evaluations, "max_evaluations")
        v.integer(self.maxiter, "maxiter")
        v.choice(
            self.method,
            ("hybr", "broyden1", "broyden2", "anderson", "krylov"),
            "method",
        )

    def __call__(self, problem: RootSystem) -> ArrayResult:
        return roots.system_root(
            problem,
            xtol=self.xtol,
            residual_tol=self.residual_tol,
            max_evaluations=self.max_evaluations,
            method=self.method,
            maxiter=self.maxiter,
        )


def root_scalar(
    function: Callable[[float], float],
    *,
    bracket: tuple[float, float] | None = None,
    x0: float | None = None,
    x1: float | None = None,
    derivative: Callable[[float], float] | None = None,
    tool: NumericalTool[ScalarRootProblem, ArrayResult] | None = None,
    check_convergence: bool = True,
) -> ArrayResult:
    """Find a scalar zero; the default Brent solver requires a sign-changing bracket."""
    return run(
        ScalarRootProblem(function, bracket, x0, x1, derivative),
        tool if tool is not None else ScalarRootSolver(),
        check_convergence=check_convergence,
    )


def root(
    function: Callable[..., ArrayLike],
    x0: ArrayLike,
    *,
    jacobian: Callable[..., ArrayLike] | None = None,
    tool: NumericalTool[RootSystem, ArrayResult] | None = None,
    check_convergence: bool = True,
) -> ArrayResult:
    """Find a zero of a square nonlinear system; inspect residual_norm in diagnostics."""
    return run(
        RootSystem(function, x0, jacobian),
        tool if tool is not None else SystemRootSolver(),
        check_convergence=check_convergence,
    )
