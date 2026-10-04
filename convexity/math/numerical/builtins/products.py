from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from numpy.typing import ArrayLike

from ..._typing import FloatArray

if TYPE_CHECKING:
    from .. import ArrayResult, DerivativeCheck, DifferentiationProblem, NumericalTool


from dataclasses import dataclass

from .. import _validation as v
from .._backends import products
from ..interfaces import ArrayResult, DerivativeProductProblem, NumericalTool
from ..interfaces.base import run
from ..interfaces.calculus import DifferentiationKind, DifferentiationProblem
from .calculus import FiniteDifference


@dataclass(frozen=True)
class FiniteDifferenceProduct:
    """Approximate Jacobian/Hessian products numerically using the configured finite-difference step."""

    differentiation: FiniteDifference = FiniteDifference()

    def __call__(self, problem: DerivativeProductProblem):
        d = self.differentiation
        if d.direction:
            raise ValueError("derivative products require central differences")
        return products.finite_product(
            problem,
            options=dict(
                atol=d.atol,
                rtol=d.rtol,
                order=d.order,
                maxiter=d.maxiter,
                initial_step=d.initial_step,
                direction=0,
            ),
        )


@dataclass(frozen=True)
class AutoDiffProduct:
    """Compute derivative products through a differentiable tensor callback; NumPy conversion breaks the graph."""

    device: str = "cpu"

    def __post_init__(self) -> None:
        if not isinstance(self.device, str) or not self.device:
            raise ValueError("device must be a nonempty string")

    def __call__(self, problem: DerivativeProductProblem):
        return products.autodiff_product(problem, device=self.device)


def jvp(
    function: Callable[[FloatArray], ArrayLike],
    x: ArrayLike,
    vector: ArrayLike,
    *,
    tool: NumericalTool[DerivativeProductProblem, ArrayResult] | None = None,
    check_convergence: bool = True,
) -> ArrayResult:
    """Evaluate Jacobian(function)(x) @ vector for vector x and vector output.

    vector has input shape; the result has output shape and numerical diagnostics."""
    return run(
        DerivativeProductProblem(function, x, vector, "jvp"),
        tool if tool is not None else FiniteDifferenceProduct(),
        check_convergence=check_convergence,
    )


def vjp(
    function: Callable[[FloatArray], ArrayLike],
    x: ArrayLike,
    vector: ArrayLike,
    *,
    tool: NumericalTool[DerivativeProductProblem, ArrayResult] | None = None,
    check_convergence: bool = True,
) -> ArrayResult:
    """Evaluate vector @ Jacobian(function)(x) for vector x and vector output.

    vector has output shape; the result has input shape and numerical diagnostics."""
    return run(
        DerivativeProductProblem(function, x, vector, "vjp"),
        tool if tool is not None else FiniteDifferenceProduct(),
        check_convergence=check_convergence,
    )


def check_derivative(
    function: Callable[[FloatArray], ArrayLike],
    x: ArrayLike,
    actual: ArrayLike | ArrayResult,
    *,
    kind: DifferentiationKind = "jacobian",
    atol: float = 1e-6,
    rtol: float = 1e-5,
    reference_tool: NumericalTool[DifferentiationProblem, ArrayResult] | None = None,
) -> DerivativeCheck:
    """Compare a supplied derivative array/result against numerical differences.

    Pass a NumPy callback as function even when actual was computed with PyTorch.
    Failed reference differentiation raises rather than producing a passing check.
    """
    v.tolerances(atol, rtol)
    problem = DifferentiationProblem(function, x, kind)
    reference = run(
        problem, reference_tool if reference_tool is not None else FiniteDifference()
    )
    actual_result = actual if isinstance(actual, ArrayResult) else ArrayResult(actual)
    problem.validate_result(actual_result)
    if not actual_result.diagnostics.converged:
        raise ValueError("cannot check a nonconverged derivative")
    if actual_result.value.shape != reference.value.shape:
        raise ValueError("actual derivative and reference must have the same shape")
    return products.compare(actual_result.value, reference.value, atol=atol, rtol=rtol)


__all__ = [
    "FiniteDifferenceProduct",
    "AutoDiffProduct",
    "jvp",
    "vjp",
    "check_derivative",
]
