"""Positive Laplace-quadrature lifts and independently evaluated L2 error."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .. import KernelError, MemoryKernel


if TYPE_CHECKING:
    from .. import ExponentialKernel


import numpy as np
from scipy.integrate import quad
from scipy.special import gamma

from .. import _validation as v
from ..interfaces.base import MemoryKernel
from ..interfaces.results import KernelError
from .kernels import ExponentialKernel, FractionalKernel


def kernel_error(
    target: MemoryKernel, approximation: MemoryKernel, horizon: float
) -> KernelError:
    """Compute integral_0^T (K-K_n)^2, including the singular near-zero tail.

    By Ito isometry this is E[(X(T)-X_n(T))**2] under a common W.
    quadrature_error is SciPy's estimate, not a rigorous bound.
    """
    horizon = v.scalar(horizon, "horizon", minimum=0, strict=True)
    energy = v.scalar(target.covariance(horizon, horizon), "target energy", minimum=0)
    if isinstance(target, FractionalKernel) and target.hurst <= 0.5:
        h = target.hurst

        # t=T*z**(1/(2H)) cancels K(t)^2's endpoint singularity.
        def integrand(z):
            t = horizon * z ** (1 / (2 * h))
            ratio = float(approximation(t)) * t ** (0.5 - h) / np.sqrt(2 * h)
            return (1 - ratio) ** 2

        value, error = quad(integrand, 0, 1, epsabs=1e-12, epsrel=2e-9, limit=400)
        value, error = energy * value, energy * error
    else:
        value, error = quad(
            lambda t: float(target(t) - approximation(t)) ** 2,
            0,
            horizon,
            epsabs=0,
            epsrel=2e-9,
            limit=400,
        )
    return KernelError(horizon, value, energy, error)


def approximate_fractional(
    hurst: float,
    *,
    factors: int = 16,
    horizon: float = 1.0,
    rate_min: float | None = None,
    rate_max: float | None = None,
    tolerance: float | None = None,
) -> tuple[ExponentialKernel, KernelError]:
    """Return (ExponentialKernel, KernelError) from geometric Laplace bins.

    Rates are bin barycenters and weights their exact Laplace mass. Increasing
    factors refines fixed bounds, but cannot remove their truncation error.
    Explicit rate bounds have units 1/time. H=.5 is exactly one zero-rate factor.
    """
    h = v.hurst(hurst, rough=True)
    factors = v.integer(factors, "factors", 1)
    horizon = v.scalar(horizon, "horizon", minimum=0, strict=True)
    lo = (
        1e-5 / horizon
        if rate_min is None
        else v.scalar(rate_min, "rate_min", minimum=0, strict=True)
    )
    hi = (
        1e8 / horizon
        if rate_max is None
        else v.scalar(rate_max, "rate_max", minimum=0, strict=True)
    )
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        raise ValueError(
            "rate bounds must be finite and satisfy 0 < rate_min < rate_max"
        )
    if h == 0.5:
        kernel = ExponentialKernel([1.0], [0.0])
    else:
        alpha = 0.5 - h
        edges = np.geomspace(lo, hi, factors + 1)
        widths = np.diff(np.log(edges))
        mass = edges[:-1] ** alpha * np.expm1(alpha * widths) / alpha
        first = edges[:-1] ** (alpha + 1) * np.expm1((alpha + 1) * widths) / (alpha + 1)
        kernel = ExponentialKernel(np.sqrt(2 * h) / gamma(alpha) * mass, first / mass)
    error = kernel_error(FractionalKernel(h), kernel, horizon)
    if tolerance is not None:
        error.require(tolerance)
    return kernel, error
