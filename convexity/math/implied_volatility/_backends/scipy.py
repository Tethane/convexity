"""Private scipy execution for implied_volatility."""

import numpy as np
from scipy.optimize import brentq, minimize
from ..._finance import array, scalar, positive
from ...pricing import forward_option


def implied_volatility(
    price,
    forward,
    strike,
    maturity,
    *,
    discount=1.0,
    kind="call",
    model="black76",
    displacement=0.0,
    tolerance=1e-12,
):
    p, t = (scalar(price, "price", minimum=0), positive(maturity, "maturity"))
    positive(tolerance, "tolerance")
    intrinsic = forward_option(
        forward,
        strike,
        0.0,
        t,
        discount=discount,
        kind=kind,
        model=model,
        displacement=displacement,
    )
    if p < intrinsic - tolerance:
        raise ValueError("price is below intrinsic value")
    if abs(p - intrinsic) <= tolerance:
        return 0.0
    if model == "black76":
        upper = discount * (
            forward + displacement if kind == "call" else strike + displacement
        )
        if p >= upper:
            raise ValueError("price is at or above the finite-volatility upper bound")

    def residual(vol):
        return (
            forward_option(
                forward,
                strike,
                vol,
                t,
                discount=discount,
                kind=kind,
                model=model,
                displacement=displacement,
            )
            - p
        )

    upper_vol = 1.0
    while residual(upper_vol) < 0 and upper_vol < 1000000000000.0:
        upper_vol *= 2
    root, report = brentq(residual, 0.0, upper_vol, xtol=tolerance, full_output=True)
    if not report.converged:
        raise RuntimeError("implied volatility inversion did not converge")
    return float(root)


def fit_svi(
    cls, k, variance, *, initial=(0.02, 0.1, -0.2, 0.0, 0.2), constraint_grid=None
):
    """Constrain positivity, Lee wing slopes, and g(k)>=0 on a declared grid."""
    k, w = (array(k, "log moneyness", ndim=1), array(variance, "variance", ndim=1))
    if k.shape != w.shape or len(k) < 5 or np.any(w <= 0):
        raise ValueError("need at least five matching positive variance observations")
    grid = (
        np.linspace(min(-3.0, k.min()), max(3.0, k.max()), 301)
        if constraint_grid is None
        else array(constraint_grid, "constraint grid", ndim=1)
    )

    def values(p, x):
        a, b, r, m, s = p
        z = x - m
        root = np.sqrt(z * z + s * s)
        v = a + b * (r * z + root)
        vp = b * (r + z / root)
        vpp = b * s * s / root**3
        return (v, vp, vpp)

    def constraints(p):
        a, b, r, m, s = p
        v, vp, vpp = values(p, grid)
        safe = np.maximum(v, 1e-12)
        g = (1 - grid * vp / (2 * safe)) ** 2 - vp**2 / 4 * (1 / safe + 0.25) + vpp / 2
        return np.r_[a + b * s * np.sqrt(1 - r * r) - 1e-10, 2 - b * (1 + abs(r)), g]

    result = minimize(
        lambda p: np.sum((values(p, k)[0] - w) ** 2),
        initial,
        method="SLSQP",
        bounds=[(None, None), (0, None), (-0.999, 0.999), (None, None), (1e-06, None)],
        constraints=[{"type": "ineq", "fun": constraints}],
        options={"ftol": 1e-14, "maxiter": 3000},
    )
    if not result.success or np.min(constraints(result.x)) < -1e-08:
        raise RuntimeError(f"SVI calibration failed: {result.message}")
    return cls(*result.x)
