"""Covariance state, factor composition and explicit matrix repairs."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .. import _validation as v
from .base import CovarianceEstimator, Diagnostics, checked_convergence, diagnostics


@dataclass(frozen=True, init=False)
class CovarianceEstimate:
    location: NDArray[Any]
    covariance: NDArray[Any]
    nobs: int | None
    effective_nobs: float | None
    shrinkage: float | None
    diagnostics: Diagnostics

    def __init__(self, location, covariance, *, nobs=None, effective_nobs=None,
                 shrinkage=None, diagnostics=None):
        loc = v.array(location, "location", ndim=1)
        cov = v.psd(covariance)
        if cov.shape != (len(loc), len(loc)):
            raise ValueError("location and covariance dimensions must agree")
        count = None if nobs is None else v.integer(nobs, "nobs", 1)
        effective = None if effective_nobs is None else v.scalar(effective_nobs, "effective_nobs", positive=True)
        if effective is not None and (effective < 1 - 1e-10 or (count is not None and effective > count + 1e-10)):
            raise ValueError("effective_nobs must lie between 1 and nobs")
        amount = None if shrinkage is None else v.scalar(shrinkage, "shrinkage")
        if amount is not None and not 0 <= amount <= 1:
            raise ValueError("shrinkage must lie in [0,1]")
        for name, value in dict(location=loc, covariance=cov, nobs=count, effective_nobs=effective,
                                shrinkage=amount, diagnostics=_report(diagnostics)).items():
            object.__setattr__(self, name, value)

    @property
    def dimension(self):
        return len(self.location)

    @property
    def correlation(self):
        from ..builtins.matrices import covariance_to_correlation
        return covariance_to_correlation(self.covariance)

    @property
    def standard_deviation(self):
        return v.array(np.sqrt(np.diag(self.covariance)), "standard deviation")

    @property
    def rank(self):
        return int(np.linalg.matrix_rank(self.covariance, hermitian=True))

    @property
    def precision(self):
        from scipy.linalg import cho_factor, cho_solve
        cov = v.psd(self.covariance, definite=True)
        return v.array(cho_solve(cho_factor(cov), np.eye(self.dimension)), "precision")

    def variance(self, weights):
        """Variance of linear combinations, weights shaped (...,d)."""
        w = v.coordinates(weights, self.dimension)
        return v.array(np.maximum(np.einsum("...i,ij,...j->...", w, self.covariance, w), 0),
                       "variance", empty=True)

    def mahalanobis(self, values):
        """Squared distances; singular covariance has no implicit pseudoinverse."""
        delta = v.coordinates(values, self.dimension) - self.location
        return v.array(np.einsum("...i,ij,...j->...", delta, self.precision, delta),
                       "squared distance", empty=True)


_report = diagnostics


@dataclass(frozen=True, init=False)
class FactorCovariance(CovarianceEstimate):
    """Cov(X) = B Cov(F) B.T + Cov(epsilon), with uncorrelated factors/noise."""

    loadings: NDArray[Any]
    factor_covariance: NDArray[Any]
    residual_covariance: NDArray[Any]

    def __init__(self, loadings, factor_covariance, residual_covariance, *, location=None,
                 nobs=None, diagnostics=None):
        b = v.array(loadings, "loadings", ndim=2)
        f = v.psd(factor_covariance, "factor_covariance")
        raw = v.array(residual_covariance, "residual_covariance")
        noise = v.psd(np.diag(raw) if raw.ndim == 1 else raw, "residual_covariance")
        if f.shape != (b.shape[1], b.shape[1]) or noise.shape != (b.shape[0], b.shape[0]):
            raise ValueError("loadings (d,k), factor covariance (k,k), residual covariance (d,d) or (d,)")
        super().__init__(np.zeros(b.shape[0]) if location is None else location,
                         b @ f @ b.T + noise, nobs=nobs, diagnostics=diagnostics)
        for name, value in dict(loadings=b, factor_covariance=f, residual_covariance=noise).items():
            object.__setattr__(self, name, value)

    @property
    def common_covariance(self):
        return v.array(self.loadings @ self.factor_covariance @ self.loadings.T, "common covariance")


@dataclass(frozen=True, init=False)
class MatrixRepair:
    matrix: NDArray[Any]
    distance: float
    min_eigenvalue_before: float
    min_eigenvalue_after: float
    diagnostics: Diagnostics

    def __init__(self, original, matrix, *, diagnostics=None):
        report = _report(diagnostics)
        before = v.symmetric(original)
        after = v.psd(matrix) if report.converged else v.symmetric(matrix)
        if before.shape != after.shape:
            raise ValueError("matrix repair must preserve dimensions")
        for name, value in dict(matrix=after, distance=float(np.linalg.norm(after - before)),
                                min_eigenvalue_before=float(np.linalg.eigvalsh(before)[0]),
                                min_eigenvalue_after=float(np.linalg.eigvalsh(after)[0]),
                                diagnostics=report).items():
            object.__setattr__(self, name, value)


def fit_covariance(samples, estimator: CovarianceEstimator, *, check_convergence=True):
    """Fit any compatible tool and check the returned mathematical state."""
    x = v.samples(samples, minimum=1)
    result = estimator.fit(x)
    if not isinstance(result, CovarianceEstimate):
        raise TypeError("estimator must return CovarianceEstimate")
    if result.dimension != x.shape[1] or (result.nobs is not None and result.nobs != len(x)):
        raise ValueError("estimate must match training dimensions and observation count")
    return checked_convergence(result, check_convergence)
