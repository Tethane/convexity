"""Reuse scalar laws and the existing PSD Gaussian state contract."""

import numpy as np


def atom(state):
    from ...distributions import empirical

    return empirical(np.asarray(state)[None, ...])


def gaussian(mean, covariance):
    from ...distributions import normal

    if np.ndim(mean) == 0:
        return atom(mean) if covariance == 0 else normal(mean, np.sqrt(covariance))
    from ...state_space import GaussianState

    return GaussianState(mean, covariance).distribution()


def cir_law(state, dt, kappa, theta, sigma):
    from scipy.stats import ncx2
    from ...distributions import Distribution
    from ...distributions._backends.scipy import ScalarAdapter

    decay = np.exp(-kappa * dt)
    scale = sigma**2 * (-np.expm1(-kappa * dt)) / (4 * kappa)
    return Distribution(
        ScalarAdapter(
            ncx2(4 * kappa * theta / sigma**2, state * decay / scale, scale=scale)
        )
    )
