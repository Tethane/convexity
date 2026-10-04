"""Validated factories for scikit-learn covariance kernels."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from .. import _validation as v
from ..interfaces.kernels import Kernel


def RBF(
    length_scale: ArrayLike = 1.0, *, bounds: ArrayLike | str = (1e-5, 1e5)
) -> Kernel:
    """Squared exponential; a length-scale vector enables anisotropy."""
    from sklearn.gaussian_process.kernels import RBF as SK

    length, bounds = v.parameter(length_scale, bounds, "length_scale", vector=True)
    return Kernel(SK(length, bounds))


def Matern(
    length_scale: ArrayLike = 1.0,
    *,
    nu: float = 1.5,
    bounds: ArrayLike | str = (1e-5, 1e5),
) -> Kernel:
    """Matérn with fixed smoothness nu > 0 (or infinity for RBF)."""
    from sklearn.gaussian_process.kernels import Matern as SK

    length, bounds = v.parameter(length_scale, bounds, "length_scale", vector=True)
    if isinstance(nu, (bool, np.bool_)) or not isinstance(
        nu, (int, float, np.integer, np.floating)
    ):
        raise ValueError("nu must be a positive real number or infinity")
    if np.isnan(nu) or nu <= 0:
        raise ValueError("nu must be positive")
    return Kernel(SK(length, bounds, float(nu)))


def Constant(value: float = 1.0, *, bounds: ArrayLike | str = (1e-5, 1e5)) -> Kernel:
    """Constant covariance value, not a deterministic mean or a standard deviation."""
    from sklearn.gaussian_process.kernels import ConstantKernel

    value, bounds = v.parameter(value, bounds, "value", allow_zero=True)
    return Kernel(ConstantKernel(value, bounds))


def WhiteNoise(
    noise_level: float = 1.0, *, bounds: ArrayLike | str = (1e-5, 1e5)
) -> Kernel:
    """Independent variance on a Gram diagonal; all cross covariances are zero."""
    from sklearn.gaussian_process.kernels import WhiteKernel

    level, bounds = v.parameter(noise_level, bounds, "noise_level", allow_zero=True)
    return Kernel(WhiteKernel(level, bounds))


def DotProduct(
    sigma_0: float = 1.0, *, bounds: ArrayLike | str = (1e-5, 1e5)
) -> Kernel:
    """x @ y.T + sigma_0**2. Pure linear covariance uses sigma_0=0, bounds='fixed'."""
    from sklearn.gaussian_process.kernels import DotProduct as SK

    value, bounds = v.parameter(sigma_0, bounds, "sigma_0", allow_zero=True)
    return Kernel(SK(value, bounds))


def RationalQuadratic(
    length_scale: float = 1.0,
    alpha: float = 1.0,
    *,
    length_scale_bounds: ArrayLike | str = (1e-5, 1e5),
    alpha_bounds: ArrayLike | str = (1e-5, 1e5),
) -> Kernel:
    """Isotropic scale mixture of RBF kernels; alpha is the mixture shape."""
    from sklearn.gaussian_process.kernels import RationalQuadratic as SK

    length, lb = v.parameter(length_scale, length_scale_bounds, "length_scale")
    shape, ab = v.parameter(alpha, alpha_bounds, "alpha")
    return Kernel(SK(length, shape, lb, ab))


def Periodic(
    length_scale: float = 1.0,
    periodicity: float = 1.0,
    *,
    length_scale_bounds: ArrayLike | str = (1e-5, 1e5),
    periodicity_bounds: ArrayLike | str = (1e-5, 1e5),
) -> Kernel:
    """Exp-sine-squared periodic covariance for one-dimensional inputs.

    The one-dimensional restriction preserves the PSD guarantee; applying
    sine to arbitrary Euclidean distances in multiple dimensions need not.
    """
    from .._backends.sklearn import periodic_kernel

    length, lb = v.parameter(length_scale, length_scale_bounds, "length_scale")
    period, pb = v.parameter(periodicity, periodicity_bounds, "periodicity")
    return Kernel(periodic_kernel(length, period, lb, pb))
