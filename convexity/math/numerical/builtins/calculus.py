"""Canonical quadrature, finite differences, and optional automatic derivatives."""

from __future__ import annotations

from typing import TYPE_CHECKING, overload

if TYPE_CHECKING:
    from ..._provider_types import NativeTensor

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike

from ..._typing import FloatArray
from .. import _validation as v
from .._backends import autodiff, calculus
from ..interfaces import (
    ArrayResult,
    DifferentiationKind,
    DifferentiationProblem,
    NumericalTool,
    QuadratureProblem,
    run,
)


@dataclass(frozen=True)
class AdaptiveQuadrature:
    """QUADPACK adaptive real scalar integration, including infinite bounds."""

    atol: float = 1e-10
    rtol: float = 1e-10
    limit: int = 100

    def __post_init__(self) -> None:
        v.tolerances(self.atol, self.rtol)
        if self.atol == 0 and self.rtol < 50 * np.finfo(float).eps:
            raise ValueError(
                "with atol=0, quadrature rtol must be >= 50 * machine epsilon"
            )
        v.integer(self.limit, "limit")

    def __call__(self, problem: QuadratureProblem) -> ArrayResult:
        return calculus.quadrature(
            problem, atol=self.atol, rtol=self.rtol, limit=self.limit
        )


@dataclass(frozen=True)
class FiniteDifference:
    """SciPy adaptive finite differences of one-point Python/NumPy callbacks.

    direction=-1/+1 requests one-sided scalar first derivatives. Other
    operations use central differences. Error estimates are componentwise.
    """

    atol: float = 1e-8
    rtol: float = 1e-6
    order: int = 8
    maxiter: int = 10
    initial_step: float = 0.5
    direction: int = 0

    def __post_init__(self) -> None:
        v.tolerances(self.atol, self.rtol)
        v.integer(self.order, "order")
        v.integer(self.maxiter, "maxiter")
        v.positive(self.initial_step, "initial_step")
        if isinstance(self.direction, bool) or self.direction not in (-1, 0, 1):
            raise ValueError("direction must be -1, 0, or 1")

    def __call__(self, problem: DifferentiationProblem) -> ArrayResult:
        if self.direction and problem.kind != "derivative":
            raise ValueError(
                "one-sided steps are supported for scalar derivatives only"
            )
        return calculus.finite_difference(
            problem,
            atol=self.atol,
            rtol=self.rtol,
            order=self.order,
            maxiter=self.maxiter,
            initial_step=self.initial_step,
            direction=self.direction,
        )


@dataclass(frozen=True)
class AutoDiff:
    """PyTorch derivatives of tensor callbacks, evaluated in float64.

    Inputs/results use the ordinary NumPy contract; only the callback uses
    tensors. Results are detached. Constant/disconnected outputs have zero
    derivatives, following PyTorch's strict=False convention.
    """

    device: str = "cpu"

    def __post_init__(self) -> None:
        if not isinstance(self.device, str) or not self.device:
            raise ValueError("device must be a nonempty PyTorch device string")

    def __call__(self, problem: DifferentiationProblem) -> ArrayResult:
        return autodiff.differentiate(problem, device=self.device)


def quad(
    function: Callable[[float], float],
    a: float,
    b: float,
    *,
    tool: NumericalTool[QuadratureProblem, ArrayResult] | None = None,
    check_convergence: bool = True,
) -> ArrayResult:
    """Integrate a real scalar function; result includes an absolute error estimate."""
    return run(
        QuadratureProblem(function, a, b),
        tool if tool is not None else AdaptiveQuadrature(),
        check_convergence=check_convergence,
    )


def _differentiate(
    function: Callable[[FloatArray], ArrayLike]
    | Callable[[float], ArrayLike]
    | Callable[[NativeTensor], NativeTensor],
    x: ArrayLike,
    kind: DifferentiationKind,
    tool: NumericalTool[DifferentiationProblem, ArrayResult] | None,
    check_convergence: bool,
) -> ArrayResult:
    return run(
        DifferentiationProblem(function, x, kind),
        tool if tool is not None else FiniteDifference(),
        check_convergence=check_convergence,
    )


@overload
def derivative(
    function: Callable[[NativeTensor], NativeTensor],
    x: float,
    *,
    tool: AutoDiff,
    check_convergence: bool = True,
) -> ArrayResult: ...


@overload
def derivative(
    function: Callable[[float], ArrayLike],
    x: float,
    *,
    tool: NumericalTool[DifferentiationProblem, ArrayResult] | None = None,
    check_convergence: bool = True,
) -> ArrayResult: ...


def derivative(
    function: Callable[[float], ArrayLike] | Callable[[NativeTensor], NativeTensor],
    x: float,
    *,
    tool: NumericalTool[DifferentiationProblem, ArrayResult] | None = None,
    check_convergence: bool = True,
) -> ArrayResult:
    """First derivative of a scalar function at a scalar point."""
    return _differentiate(function, x, "derivative", tool, check_convergence)


@overload
def gradient(
    function: Callable[[NativeTensor], NativeTensor],
    x: ArrayLike,
    *,
    tool: AutoDiff,
    check_convergence: bool = True,
) -> ArrayResult: ...


@overload
def gradient(
    function: Callable[[FloatArray], ArrayLike],
    x: ArrayLike,
    *,
    tool: NumericalTool[DifferentiationProblem, ArrayResult] | None = None,
    check_convergence: bool = True,
) -> ArrayResult: ...


def gradient(
    function: Callable[[FloatArray], ArrayLike]
    | Callable[[NativeTensor], NativeTensor],
    x: ArrayLike,
    *,
    tool: NumericalTool[DifferentiationProblem, ArrayResult] | None = None,
    check_convergence: bool = True,
) -> ArrayResult:
    """Gradient of a scalar-valued function; x and result have shape (n,)."""
    return _differentiate(function, x, "gradient", tool, check_convergence)


@overload
def jacobian(
    function: Callable[[NativeTensor], NativeTensor],
    x: ArrayLike,
    *,
    tool: AutoDiff,
    check_convergence: bool = True,
) -> ArrayResult: ...


@overload
def jacobian(
    function: Callable[[FloatArray], ArrayLike],
    x: ArrayLike,
    *,
    tool: NumericalTool[DifferentiationProblem, ArrayResult] | None = None,
    check_convergence: bool = True,
) -> ArrayResult: ...


def jacobian(
    function: Callable[[FloatArray], ArrayLike]
    | Callable[[NativeTensor], NativeTensor],
    x: ArrayLike,
    *,
    tool: NumericalTool[DifferentiationProblem, ArrayResult] | None = None,
    check_convergence: bool = True,
) -> ArrayResult:
    """Jacobian of f: R^n -> R^m; result has shape (m, n)."""
    return _differentiate(function, x, "jacobian", tool, check_convergence)


@overload
def hessian(
    function: Callable[[NativeTensor], NativeTensor],
    x: ArrayLike,
    *,
    tool: AutoDiff,
    check_convergence: bool = True,
) -> ArrayResult: ...


@overload
def hessian(
    function: Callable[[FloatArray], ArrayLike],
    x: ArrayLike,
    *,
    tool: NumericalTool[DifferentiationProblem, ArrayResult] | None = None,
    check_convergence: bool = True,
) -> ArrayResult: ...


def hessian(
    function: Callable[[FloatArray], ArrayLike]
    | Callable[[NativeTensor], NativeTensor],
    x: ArrayLike,
    *,
    tool: NumericalTool[DifferentiationProblem, ArrayResult] | None = None,
    check_convergence: bool = True,
) -> ArrayResult:
    """Hessian of a scalar-valued function; result has shape (n, n)."""
    return _differentiate(function, x, "hessian", tool, check_convergence)
