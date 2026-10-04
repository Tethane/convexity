"""Private scipy execution for credit."""

import numpy as np
from scipy.optimize import brentq
from scipy.integrate import quad
from scipy.linalg import expm
from scipy.stats import norm
from ..interfaces import StructuralCreditResult
from ..._finance import scalar, positive, unit_interval, array, integer
from ..._validation import generator
from ...regimes import MarkovChain


def sample_default_times(self, draws, *, horizon, rng):
    """Default times up to horizon; +inf means survival beyond horizon."""
    generator(rng)
    n = integer(draws, "draws", 1)
    end = positive(horizon, "horizon")
    p = 1 - self.survival(end)
    u = rng.random(n)
    out = np.full(n, np.inf)
    for i in np.flatnonzero(u < p):
        out[i] = brentq(lambda t: 1 - self.survival(t) - u[i], 0.0, end)
    return array(out, "default times", finite=False)


def risky_cashflows(
    times,
    amounts,
    discount,
    default,
    *,
    recovery=0.0,
    recovery_notional=1.0,
    integration_points=None,
):
    """Recovery of par paid at default through final payment, continuous integral.

    Float times use each curve's year convention; matching conventions required.
    """
    if default.measure != "Q":
        raise ValueError("risky valuation requires Q default probabilities")
    if (
        discount.reference_date != default.reference_date
        or discount.structure.dayCounter() != default.structure.dayCounter()
    ):
        raise ValueError("credit and discount references/day counts must agree")
    t = array(times, "times", ndim=1)
    a = array(amounts, "amounts", ndim=1)
    if t.shape != a.shape or t[0] <= 0 or np.any(np.diff(t) <= 0):
        raise ValueError("positive ordered times must match cash amounts")
    promised = sum(
        (x * discount.discount(s) * default.survival(s) for s, x in zip(t, a))
    )
    rec = unit_interval(recovery, "recovery") * positive(
        recovery_notional, "recovery notional"
    )
    expected_recovery = (
        rec
        * quad(
            lambda s: discount.discount(s) * default.density(s),
            0.0,
            t[-1],
            points=integration_points,
            epsabs=1e-10,
            limit=200,
        )[0]
    )
    return promised + expected_recovery


def transition(self, time):
    return MarkovChain(expm(self.generator * scalar(time, "time", minimum=0)))


def merton_structural(
    asset_value, debt, asset_volatility, maturity, rate, *, physical_drift=None
):
    """Merton terminal default model; Q debt valuation and optional P probability.

    This is terminal insolvency, not a first-passage default-time intensity model.
    """
    v = positive(asset_value, "asset value")
    k = positive(debt, "debt")
    sigma = positive(asset_volatility, "asset volatility")
    t = positive(maturity, "maturity")
    r = scalar(rate, "rate")
    sd = sigma * np.sqrt(t)
    d2 = (np.log(v / k) + (r - 0.5 * sigma * sigma) * t) / sd
    d1 = d2 + sd
    equity = v * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d2)
    debt_value = v * norm.cdf(-d1) + k * np.exp(-r * t) * norm.cdf(d2)
    result = {
        "equity": float(equity),
        "debt": float(debt_value),
        "q_default_probability": float(norm.cdf(-d2)),
        "credit_spread": float(-np.log(debt_value / k) / t - r),
    }
    if physical_drift is not None:
        result["p_default_probability"] = float(
            norm.cdf(
                -(
                    np.log(v / k)
                    + (scalar(physical_drift, "physical drift") - 0.5 * sigma * sigma)
                    * t
                )
                / sd
            )
        )
    return StructuralCreditResult(**result)
