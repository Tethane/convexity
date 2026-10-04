"""Reuse numerical derivatives (including PyTorch) and SciPy hedge solvers."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict, cast

from ..._typing import ScalarLike
from ...numerical import ArrayResult

if TYPE_CHECKING:
    from ..._provider_types import NativeTensor
from collections.abc import Callable, Sequence

from numpy.typing import ArrayLike

from ..._typing import FloatArray

if TYPE_CHECKING:
    from .. import HedgeResult
if TYPE_CHECKING:
    from ...curves import YieldCurve


import numpy as np

from ..._finance import array, positive, scalar
from ...numerical import AutoDiff, FiniteDifference, hessian, jacobian
from .._backends import scipy as _scipy


def sensitivities(
    function: Callable[[FloatArray], ArrayLike]
    | Callable[[NativeTensor], NativeTensor],
    point: ArrayLike,
    *,
    order: int = 1,
    autodiff: bool = False,
) -> ArrayResult:
    """Per-unit derivatives: first order is (output,input); second is Hessian.

    PyTorch callbacks must remain tensor-native; QuantLib is opaque to autograd.
    """
    if order not in (1, 2):
        raise ValueError("order must be 1 or 2")
    tool = AutoDiff() if autodiff else FiniteDifference()
    differentiate = cast(
        Callable[..., ArrayResult], jacobian if order == 1 else hessian
    )
    return differentiate(function, point, tool=tool)


class SensitivityReport(TypedDict):
    """Scalar value, gradient (coordinate,), and Hessian (coordinate,coordinate)."""

    value: float
    gradient: FloatArray
    hessian: FloatArray


def bump_sensitivities(
    function: Callable[[FloatArray], ScalarLike], point: ArrayLike, *, steps: ArrayLike
) -> SensitivityReport:
    """Central gradient/Hessian including cross-Greeks; absolute factor bumps."""
    x = array(point, "point", ndim=1)
    h = np.broadcast_to(array(steps, "steps"), x.shape)
    if np.any(h <= 0):
        raise ValueError("steps must be positive")
    base = scalar(function(x), "base value")
    n = len(x)
    g = np.empty(n)
    H = np.empty((n, n))
    basis = np.diag(h)
    for i in range(n):
        plus, minus = (
            scalar(function(x + basis[i]), "plus value"),
            scalar(function(x - basis[i]), "minus value"),
        )
        g[i] = (plus - minus) / (2 * h[i])
        H[i, i] = (plus - 2 * base + minus) / h[i] ** 2
        for j in range(i):
            H[i, j] = H[j, i] = (
                function(x + basis[i] + basis[j])
                - function(x + basis[i] - basis[j])
                - function(x - basis[i] + basis[j])
                + function(x - basis[i] - basis[j])
            ) / (4 * h[i] * h[j])
    return {
        "value": base,
        "gradient": array(g, "gradient"),
        "hessian": array(H, "hessian"),
    }


def hedge(
    exposures: ArrayLike,
    target: ArrayLike,
    *,
    weights: ArrayLike | None = None,
    lower: ArrayLike = -np.inf,
    upper: ArrayLike = np.inf,
    equality: ArrayLike | None = None,
    equality_target: ArrayLike | None = None,
    ridge: float = 0.0,
) -> HedgeResult:
    """Choose h to minimize ||sqrt(W)(A h + target)||² + ridge ||h||².

    A has (risk,instrument) axes; equality is E h = b. Nullspace is the
    unconstrained instrument-space exposure nullspace, independent of bounds.
    """
    return _scipy.hedge(
        exposures,
        target,
        weights=weights,
        lower=lower,
        upper=upper,
        equality=equality,
        equality_target=equality_target,
        ridge=ridge,
    )


def curve_sensitivities(
    value: Callable[[tuple[YieldCurve, ...]], ScalarLike],
    curves: Sequence[YieldCurve],
    *,
    bump: float = 0.0001,
) -> SensitivityReport:
    """Parallel curve derivatives per unit decimal rate; DV01=-derivative*1e-4."""
    h = positive(bump, "bump")
    curves = tuple(curves)

    def evaluate(shifts):
        return value(tuple((curve.shifted(s) for curve, s in zip(curves, shifts))))

    return bump_sensitivities(evaluate, np.zeros(len(curves)), steps=h)
