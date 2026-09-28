from dataclasses import dataclass
import numpy as np

from .. import _validation as v
from .._backends import products
from ..interfaces.base import ArrayResult, run
from ..interfaces.calculus import DifferentiationKind, DifferentiationProblem
from ..interfaces.products import DerivativeProductProblem
from .calculus import FiniteDifference


@dataclass(frozen=True)
class FiniteDifferenceProduct:
    differentiation: FiniteDifference = FiniteDifference()

    def __call__(self, problem: DerivativeProductProblem):
        d = self.differentiation
        if d.direction:
            raise ValueError("derivative products require central differences")
        return products.finite_product(problem, options=dict(atol=d.atol, rtol=d.rtol, order=d.order,
                                       maxiter=d.maxiter, initial_step=d.initial_step, direction=0))


@dataclass(frozen=True)
class AutoDiffProduct:
    device: str = "cpu"

    def __post_init__(self):
        if not isinstance(self.device, str) or not self.device:
            raise ValueError("device must be a nonempty string")

    def __call__(self, problem: DerivativeProductProblem):
        return products.autodiff_product(problem, device=self.device)


def jvp(function, x, vector, *, tool=None, check_convergence=True):
    return run(DerivativeProductProblem(function, x, vector, "jvp"),
               tool if tool is not None else FiniteDifferenceProduct(), check_convergence=check_convergence)


def vjp(function, x, vector, *, tool=None, check_convergence=True):
    return run(DerivativeProductProblem(function, x, vector, "vjp"),
               tool if tool is not None else FiniteDifferenceProduct(), check_convergence=check_convergence)


def check_derivative(function, x, actual, *, kind: DifferentiationKind = "jacobian", atol=1e-6, rtol=1e-5, reference_tool=None):
    """Compare a supplied derivative array/result against numerical differences.

    Pass a NumPy callback as function even when actual was computed with PyTorch.
    Failed reference differentiation raises rather than producing a passing check.
    """
    v.tolerances(atol, rtol)
    problem = DifferentiationProblem(function, x, kind)
    reference = run(problem, reference_tool if reference_tool is not None else FiniteDifference())
    actual_result = actual if isinstance(actual, ArrayResult) else ArrayResult(actual)
    problem.validate_result(actual_result)
    if not actual_result.diagnostics.converged:
        raise ValueError("cannot check a nonconverged derivative")
    if actual_result.value.shape != reference.value.shape:
        raise ValueError("actual derivative and reference must have the same shape")
    return products.compare(actual_result.value, reference.value, atol=atol, rtol=rtol)


__all__ = ["FiniteDifferenceProduct", "AutoDiffProduct", "jvp", "vjp", "check_derivative"]
