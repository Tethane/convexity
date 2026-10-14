"""Dense Gaussian algebra with scale-aware PSD validation and no jitter."""

import numpy as np
from .. import _validation as v


def root(covariance):
    raw = v.array(covariance, "covariance", ndim=2)
    if raw.shape[0] != raw.shape[1]:
        raise ValueError("covariance must be square")
    cov = raw
    if np.any(np.diag(cov) < 0):
        raise ValueError("covariance diagonal must be nonnegative")
    # Normalize each coordinate before factorization, preserving small units.
    scale = np.sqrt(np.diag(cov))
    active = scale > 0
    if np.any(cov[~active, :] != 0) or np.any(cov[:, ~active] != 0):
        raise ValueError("zero-variance coordinates must have zero covariance")
    out = np.zeros_like(cov)
    if active.any():
        block = cov[np.ix_(active, active)] / np.outer(scale[active], scale[active])
        block = v.covariance(block, (len(block),))
        eigenvalues, vectors = np.linalg.eigh(block)
        tolerance = 100 * np.finfo(float).eps * len(block) * np.max(np.abs(block))
        eigenvalues[eigenvalues < tolerance] = 0
        out[np.ix_(active, active)] = (
            scale[active, None] * vectors * np.sqrt(eigenvalues)
        )
    return out


def sample(covariance, draws, rng):
    factor = root(covariance)
    if not factor.any():
        return np.zeros((draws, len(factor)))
    return rng.standard_normal((draws, len(factor))) @ factor.T
