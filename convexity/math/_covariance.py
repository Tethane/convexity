"""Coordinate-scaled real covariance contracts, without provider dependencies."""

import numpy as np
from ._validation import array


def psd(value, name="covariance", *, definite=False):
    """Validate in dimensionless coordinates; repair only negative roundoff.

    A zero variance requires an exactly zero row. Positive eigenvalues are not
    truncated. Coordinate scaling prevents a large variance from hiding an
    invalid small block. No jitter or unit-dependent absolute floor is added.
    """
    x = array(value, name, ndim=2)
    if x.shape[0] != x.shape[1]:
        raise ValueError(f"{name} must be symmetric square")
    diagonal = np.diag(x)
    if np.any(diagonal < 0) or (definite and np.any(diagonal == 0)):
        raise ValueError(
            f"{name} must be positive {'definite' if definite else 'semidefinite'}"
        )
    active = diagonal > 0
    if np.any(x[~active] != 0) or np.any(x[:, ~active] != 0):
        raise ValueError(
            f"{name} must be positive semidefinite: zero variance needs a zero row"
        )
    if not active.any():
        return x
    units = np.sqrt(diagonal[active])
    with np.errstate(over="ignore", invalid="ignore"):
        block = x[np.ix_(active, active)] / units[:, None] / units[None, :]
    tol = 100 * np.finfo(float).eps * len(block)
    if not np.isfinite(block).all() or np.any(abs(block) > 1 + tol):
        raise ValueError(f"{name} must be positive semidefinite")
    if not np.allclose(block, block.T, atol=tol, rtol=0):
        raise ValueError(f"{name} must be symmetric")
    block = block / 2 + block.T / 2
    ev, vectors = np.linalg.eigh(block)
    if ev[0] < -tol * max(1.0, float(ev[-1])) or (definite and ev[0] <= 0):
        raise ValueError(
            f"{name} must be positive {'definite' if definite else 'semidefinite'}"
        )
    if ev[0] < 0:
        block = (vectors * np.maximum(ev, 0)) @ vectors.T
    result = np.zeros_like(x)
    result[np.ix_(active, active)] = (block * units[:, None]) * units[None, :]
    return array(result / 2 + result.T / 2, name)
