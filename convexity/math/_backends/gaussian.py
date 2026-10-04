"""Gaussian algebra preserving coordinate units and affine-support density.

    Full rank uses a Cholesky factor of correlation, never a covariance-wide
    rank cutoff. Singular rank is assessed in dimensionless coordinates. The
    density volume is measured in the original Euclidean support coordinates.
"""
import numpy as np
from scipy.linalg import solve_triangular
from .._covariance import psd


def factor(covariance):
    cov = psd(covariance)
    scales = np.sqrt(np.diag(cov))
    active = scales > 0
    units = scales[active]
    corr = cov[np.ix_(active, active)] / units[:, None] / units[None, :]
    if not len(corr):
        return np.zeros((len(cov), 0)), scales, active, np.empty((0, 0)), True
    try:
        root = np.linalg.cholesky(corr)
        triangular = True
    except np.linalg.LinAlgError:
        ev, vectors = np.linalg.eigh(corr)
        keep = ev > 100 * np.finfo(float).eps * len(corr) * max(1., ev[-1])
        root = vectors[:, keep] * np.sqrt(ev[keep])
        triangular = False
    loading = np.zeros((len(cov), root.shape[1]))
    loading[active] = units[:, None] * root
    return loading, scales, active, root, triangular


def square_root(covariance):
    """Return square L with L @ L.T = covariance, retaining coordinate scales."""
    loading, _, _, _, _ = factor(covariance)
    result = np.zeros((len(loading), len(loading)))
    result[:, :loading.shape[1]] = loading
    return result


def conditional_gain(cross_covariance, covariance):
    """Solve K C = cross on C's support in dimensionless coordinates.

    For singular C this is a generalized inverse used only for conditional
    Gaussian moments, where cross covariance annihilates C's null space.
    """
    _, scales, active, root, triangular = factor(covariance)
    cross = np.asarray(cross_covariance)
    gain = np.zeros_like(cross)
    if not active.any():
        return gain
    rhs = (cross[:, active] / scales[active]).T
    if triangular:
        solved = solve_triangular(root.T, solve_triangular(root, rhs, lower=True))
    else:
        norms2 = np.sum(root**2, axis=0)
        solved = (root / norms2**2) @ (root.T @ rhs)
    gain[:, active] = solved.T / scales[active]
    return gain


def sample(mean, covariance, size, rng):
    if isinstance(size, (int, np.integer)):
        size = (int(size),)
    loading, _, _, _, _ = factor(covariance)
    return np.asarray(mean) + rng.standard_normal(tuple(size) + (loading.shape[1],)) @ loading.T


def logpdf(values, mean, covariance):
    mean = np.asarray(mean)
    values = np.asarray(values)
    shape = values.shape[:-1]
    delta = values.reshape(-1, len(mean)) - mean
    loading, scales, active, root, triangular = factor(covariance)
    rank = root.shape[1]
    supported = np.isfinite(delta).all(axis=1) & np.all(delta[:, ~active] == 0, axis=1)
    result = np.full(len(delta), -np.inf)
    if rank == 0:
        result[supported] = 0.
        return result.reshape(shape)
    z = delta[supported][:, active] / scales[active]
    if triangular:
        white = solve_triangular(root, z.T, lower=True).T
        log_volume = np.log(scales[active]).sum() + np.log(np.diag(root)).sum()
        on_support = np.ones(len(z), dtype=bool)
    else:
        # Orthogonal columns of root in standardized coordinates.
        norms = np.linalg.norm(root, axis=0)
        axes = root / norms
        coordinates = z @ axes
        white = coordinates / norms
        error = np.max(abs(z - coordinates @ axes.T), axis=1)
        tol = 1000 * np.finfo(float).eps * len(root)
        on_support = error <= tol * np.maximum(1., np.max(abs(z), axis=1))
        # QR gives the volume of the loading's parallelepiped, including the
        # correct Hausdorff Jacobian for a singular law under diagonal scaling.
        _, qr = np.linalg.qr(loading, mode='reduced')
        log_volume = np.log(abs(np.diag(qr))).sum()
    density = -.5 * (rank * np.log(2*np.pi) + np.sum(white**2, axis=1)) - log_volume
    result[supported] = np.where(on_support, density, -np.inf)
    return result.reshape(shape)
