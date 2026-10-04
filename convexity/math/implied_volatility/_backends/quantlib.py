"""Private quantlib execution for implied_volatility."""

import numpy as np
import QuantLib as ql
from ..._finance import array, scalar, positive, qdate, day_count, choice


def sabr_volatility(
    strike, forward, maturity, alpha, beta, nu, rho, *, model="black76", shift=0.0
):
    choice(model, ("black76", "bachelier"), "model")
    if not 0 <= scalar(beta, "beta") <= 1 or abs(scalar(rho, "rho")) >= 1:
        raise ValueError("SABR requires beta in [0,1] and |rho|<1")
    return ql.shiftedSabrVolatility(
        scalar(strike, "strike"),
        scalar(forward, "forward"),
        positive(maturity, "maturity"),
        positive(alpha, "alpha"),
        beta,
        scalar(nu, "nu", minimum=0),
        rho,
        scalar(shift, "shift", minimum=0),
        ql.ShiftedLognormal if model == "black76" else ql.Normal,
    )


def black_variance_surface(
    reference_date,
    dates,
    strikes,
    volatilities,
    *,
    convention="actual365",
    extrapolate=False,
):
    """QuantLib surface; vols shape (strike,expiry), bilinear total variance."""
    ds = [qdate(d) for d in dates]
    ks = array(strikes, "strikes", ndim=1)
    vs = array(volatilities, "volatilities", ndim=2)
    ref = qdate(reference_date)
    if (
        len(ds) < 2
        or len(ks) < 2
        or vs.shape != (len(ks), len(ds))
        or np.any(vs < 0)
        or np.any(np.diff(ks) <= 0)
        or (ks[0] <= 0)
        or (ds[0] <= ref)
        or any((b <= a for a, b in zip(ds, ds[1:])))
    ):
        raise ValueError(
            "surface requires ordered positive strikes, future expiries and matching nonnegative vols"
        )
    surface = ql.BlackVarianceSurface(
        ref,
        ql.NullCalendar(),
        ds,
        ks.tolist(),
        ql.Matrix(vs.tolist()),
        day_count(convention),
    )
    surface.enableExtrapolation(extrapolate)
    return surface


def dupire_local_volatility(surface, discount, dividend, spot, time, strike):
    if (
        surface.referenceDate() != discount.reference_date
        or dividend.reference_date != discount.reference_date
    ):
        raise ValueError("surface and curves must share reference date")
    local = ql.LocalVolSurface(
        ql.BlackVolTermStructureHandle(surface),
        discount.handle,
        dividend.handle,
        positive(spot, "spot"),
    )
    return local.localVol(positive(time, "time"), positive(strike, "strike"), False)
