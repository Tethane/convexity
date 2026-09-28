"""Public calculus namespace; implementation adapters remain private."""
from .builtins.calculus import (
    AdaptiveQuadrature,
    FiniteDifference,
    AutoDiff,
    quad,
    derivative,
    gradient,
    jacobian,
    hessian,
)
from .builtins.products import (
    FiniteDifferenceProduct,
    AutoDiffProduct,
    jvp,
    vjp,
    check_derivative,
)

__all__ = ['AdaptiveQuadrature', 'FiniteDifference', 'AutoDiff', 'quad', 'derivative', 'gradient', 'jacobian', 'hessian', 'FiniteDifferenceProduct', 'AutoDiffProduct', 'jvp', 'vjp', 'check_derivative']
