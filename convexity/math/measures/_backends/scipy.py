"""Private scipy execution for measures."""

import numpy as np
from scipy.special import logsumexp
from ..._finance import array


def reweight(scenarios, log_likelihood):
    """Self-normalized importance law; no IID error estimate is asserted."""
    from ...simulation import PathScenarios

    ll = array(log_likelihood, "log likelihood", ndim=1, finite=False)
    if ll.shape != scenarios.weights.shape or np.any(np.isposinf(ll)):
        raise ValueError(
            "log likelihood must align with scenario weights and not contain +inf"
        )
    with np.errstate(divide="ignore"):
        log_mass = np.log(scenarios.weights) + ll
    z = np.asarray(logsumexp(log_mass, return_sign=False), dtype=float).item()
    if not np.isfinite(z):
        raise ValueError("reweighted mass is zero")
    return PathScenarios(
        scenarios.times,
        scenarios.values,
        np.exp(log_mass - z),
        method="measure-reweighting",
        backend=scenarios.backend,
    )


def market_price_of_risk(self, time, state):
    p, q = (self.physical.model, self.pricing.model)
    sigma = np.atleast_2d(p.diffusion(time, state))
    difference = np.atleast_1d(p.drift(time, state) - q.drift(time, state))
    risk = np.linalg.lstsq(sigma, difference, rcond=None)[0]
    if np.max(abs(sigma @ risk - difference)) > 1e-10 * np.max(abs(difference)):
        raise ValueError("drift difference is outside the diffusion span")
    return array(risk, "market price of risk")
