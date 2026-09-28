"""Library covariance fits and small weighted/factor composition identities."""

import warnings

import numpy as np

from ..interfaces.base import Diagnostics
from ..interfaces.covariance import CovarianceEstimate, FactorCovariance, MatrixRepair


def empirical(x, *, ddof, assume_centered):
    from sklearn.covariance import EmpiricalCovariance
    model = EmpiricalCovariance(assume_centered=assume_centered, store_precision=False).fit(x)
    return CovarianceEstimate(model.location_, model.covariance_ * (len(x) / (len(x) - ddof)),
                              nobs=len(x), effective_nobs=len(x),
                              diagnostics=Diagnostics(backend="sklearn.EmpiricalCovariance"))


def shrinkage(x, *, method, amount, assume_centered):
    from sklearn.covariance import LedoitWolf, OAS, ShrunkCovariance
    cls = {"fixed": ShrunkCovariance, "ledoit_wolf": LedoitWolf, "oas": OAS}[method]
    kwargs = {"shrinkage": amount} if method == "fixed" else {}
    model = cls(assume_centered=assume_centered, store_precision=False, **kwargs).fit(x)
    return CovarianceEstimate(model.location_, model.covariance_, nobs=len(x),
                              shrinkage=amount if method == "fixed" else model.shrinkage_,
                              diagnostics=Diagnostics(backend=f"sklearn.{cls.__name__}"))


def ewma(x, *, decay, ddof, assume_centered):
    # Normalized finite-history geometric observation weights, newest row last.
    weights = np.power(decay, np.arange(len(x) - 1, -1, -1, dtype=float))
    weights /= weights.sum()
    effective = float(1 / (weights @ weights))
    divisor = 1 - ddof / effective
    if divisor <= np.finfo(float).eps:
        raise ValueError("EWMA bias correction needs more than one effective observation")
    location = np.zeros(x.shape[1]) if assume_centered else weights @ x
    delta = x - location
    cov = (delta.T * weights) @ delta / divisor
    return CovarianceEstimate(location, cov, nobs=len(x), effective_nobs=effective,
                              diagnostics=Diagnostics(backend="numpy weighted moments"))


def observed_factors(x, factors, *, ddof, residual):
    from scipy.linalg import lstsq
    mean, factor_mean = x.mean(axis=0), factors.mean(axis=0)
    centered_factors, centered_x = factors - factor_mean, x - mean
    coefficients, _, rank, _ = lstsq(centered_factors, centered_x)
    if rank < factors.shape[1]:
        raise ValueError("centered factors must have full column rank")
    errors = centered_x - centered_factors @ coefficients
    divisor = len(x) - ddof
    noise = errors.T @ errors / divisor
    if residual == "diagonal":
        noise = np.diag(noise)
    return FactorCovariance(coefficients.T, centered_factors.T @ centered_factors / divisor,
                            noise, location=mean, nobs=len(x),
                            diagnostics=Diagnostics(backend="scipy.linalg.lstsq"))


def project_psd(matrix, floor):
    from scipy.linalg import eigh
    eigenvalues, axes = eigh(matrix)
    result = (axes * np.maximum(eigenvalues, floor)) @ axes.T
    return MatrixRepair(matrix, result, diagnostics=Diagnostics(backend="scipy.linalg.eigh"))


def nearest_correlation(matrix, *, floor, maxiter):
    from statsmodels.stats.correlation_tools import corr_nearest
    from statsmodels.tools.sm_exceptions import IterationLimitWarning
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", IterationLimitWarning)
        result = corr_nearest(matrix, threshold=floor, n_fact=np.nextafter(maxiter / len(matrix), np.inf))
    failures = [w for w in caught if issubclass(w.category, IterationLimitWarning)]
    for warning in caught:
        if not issubclass(warning.category, IterationLimitWarning):
            warnings.warn(str(warning.message), warning.category, stacklevel=2)
    converged = not failures
    message = ""
    if failures:
        # corr_nearest stops only when no spectral clipping is necessary;
        # an active PSD constraint can therefore exhaust its budget even at a
        # fixed point. Check the last two library iterates and feasibility.
        if maxiter > 1:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", IterationLimitWarning)
                previous = corr_nearest(matrix, threshold=floor,
                                        n_fact=np.nextafter((maxiter - 1) / len(matrix), np.inf))
            scale = max(float(np.linalg.norm(result)), 1.)
            converged = (np.linalg.norm(result - previous) <= 1e-10 * scale
                         and np.linalg.eigvalsh(result)[0] >= floor - 1e-10)
        message = ("Backend iteration budget reached; last-iterate change and feasibility verified."
                   if converged else "Correlation projection did not converge within maxiter.")
    return MatrixRepair(matrix, result, diagnostics=Diagnostics(
        backend="statsmodels.corr_nearest", converged=bool(converged), message=message))
