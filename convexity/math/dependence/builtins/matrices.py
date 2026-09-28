"""Correlation statistics, matrix constraints and explicit repairs."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from .. import _validation as v
from .._backends import covariance as backend
from ..interfaces.base import checked_convergence


def covariance_to_correlation(covariance):
    cov = v.psd(covariance)
    sd = np.sqrt(np.diag(cov))
    if np.any(sd == 0):
        raise ValueError("correlation is undefined for zero-variance coordinates")
    corr = cov / sd[:, None] / sd[None, :]
    np.fill_diagonal(corr, 1)
    return v.array(np.clip(corr, -1, 1), "correlation")


def correlation_to_covariance(correlation, standard_deviation):
    corr = v.correlation(correlation)
    sd = v.array(standard_deviation, "standard_deviation", ndim=1)
    if sd.shape != (len(corr),) or np.any(sd < 0):
        raise ValueError("standard_deviation must contain d nonnegative values")
    return v.psd(corr * sd[:, None] * sd[None, :])


def correlation(samples, *, method="pearson"):
    """Pearson, Spearman or Kendall tau-b on complete observation rows."""
    from scipy.stats import kendalltau, rankdata
    v.choice(method, ("pearson", "spearman", "kendall"), "method")
    x = v.samples(samples)
    if np.any(np.ptp(x, axis=0) == 0):
        raise ValueError("correlation requires nonconstant columns")
    if method in ("pearson", "spearman"):
        if method == "spearman":
            x = rankdata(x, axis=0, method="average")
        return covariance_to_correlation(np.atleast_2d(np.cov(x, rowvar=False)))
    result = np.eye(x.shape[1])
    for i in range(x.shape[1]):
        for j in range(i):
            statistic: Any = kendalltau(x[:, i], x[:, j])
            result[i, j] = result[j, i] = statistic.statistic
    return v.array(result, "Kendall correlation")


@dataclass(frozen=True)
class PositiveSemidefinite:
    """Eigenvalue lower bound in matrix units; numerical membership tolerance."""

    min_eigenvalue: float = 0.
    tolerance: float = 1e-10

    def __post_init__(self):
        floor = v.scalar(self.min_eigenvalue, "min_eigenvalue")
        tolerance = v.scalar(self.tolerance, "tolerance")
        if floor < 0:
            raise ValueError("min_eigenvalue must be nonnegative")
        if tolerance < 0:
            raise ValueError("tolerance must be nonnegative")
        object.__setattr__(self, "min_eigenvalue", floor)
        object.__setattr__(self, "tolerance", tolerance)

    def contains(self, matrix):
        x = v.symmetric(matrix)
        eigenvalues = np.linalg.eigvalsh(x)
        scale = max(float(np.max(np.abs(eigenvalues))), abs(self.min_eigenvalue), np.finfo(float).tiny)
        return bool(eigenvalues[0] >= self.min_eigenvalue - self.tolerance * scale)

    def project(self, matrix):
        """Frobenius projection by spectral clipping; diagonal may change."""
        return backend.project_psd(v.symmetric(matrix), self.min_eigenvalue)


def nearest_correlation(matrix, *, min_eigenvalue=1e-10, maxiter=1000, check_convergence=True):
    """statsmodels alternating projection with fixed unit diagonal."""
    x = v.symmetric(matrix)
    floor = v.scalar(min_eigenvalue, "min_eigenvalue", positive=True)
    if floor >= 1:
        raise ValueError("min_eigenvalue must lie in (0,1)")
    v.integer(maxiter, "maxiter", 1)
    v.boolean(check_convergence, "check_convergence")
    if not np.allclose(np.diag(x), 1, rtol=0, atol=1e-10):
        raise ValueError("matrix must already have unit diagonal")
    result = backend.nearest_correlation(x, floor=floor, maxiter=maxiter)
    return checked_convergence(result, check_convergence)
