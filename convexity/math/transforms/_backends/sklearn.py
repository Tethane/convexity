"""scikit-learn fits, exported as independent mathematical state."""

import warnings

import numpy as np

from ..interfaces import Diagnostics, FactorAnalysisResult, FittedScaling, PCAResult


def standardize(x, *, center, scale, ddof):
    from sklearn.preprocessing import StandardScaler

    model = StandardScaler(with_mean=center, with_std=scale).fit(x)
    location = model.mean_ if center else np.zeros(x.shape[1])
    scales = np.array(model.scale_, dtype=float) if scale else np.ones(x.shape[1])
    if scale and ddof:
        # Preserve the backend's unit scale for constant/numerically constant columns.
        variance, mean = np.asarray(model.var_), np.asarray(model.mean_)
        active = variance > (len(x) * np.finfo(float).eps * variance
                             + (len(x) * mean * np.finfo(float).eps)**2)
        scales[active] *= np.sqrt(len(x) / (len(x) - ddof))
    return FittedScaling(location, scales, diagnostics=Diagnostics(backend="sklearn.StandardScaler"))


def robust_scale(x, *, center, scale, quantile_range, unit_variance):
    from sklearn.preprocessing import RobustScaler

    model = RobustScaler(with_centering=center, with_scaling=scale,
                         quantile_range=quantile_range, unit_variance=unit_variance).fit(x)
    return FittedScaling(model.center_ if center else np.zeros(x.shape[1]),
                         model.scale_ if scale else np.ones(x.shape[1]),
                         diagnostics=Diagnostics(backend="sklearn.RobustScaler"))


def minmax_scale(x, *, lower, upper):
    from sklearn.preprocessing import MinMaxScaler

    model = MinMaxScaler(feature_range=(lower, upper), clip=False).fit(x)
    return FittedScaling(-model.min_ / model.scale_, 1 / model.scale_,
                         diagnostics=Diagnostics(backend="sklearn.MinMaxScaler"))


def pca(x, *, n_components, whiten, ddof):
    from sklearn.decomposition import PCA

    if not np.any(x != x[0]):
        raise ValueError("PCA needs nonzero training variation")
    model = PCA(n_components=n_components, svd_solver="full", whiten=False).fit(x)
    variance = model.singular_values_**2 / (len(x) - ddof)
    total = float(np.sum((x - model.mean_)**2) / (len(x) - ddof))
    return PCAResult(model.mean_, np.asarray(model.components_).T, variance, total,
                     nobs=len(x), ddof=ddof, whiten=whiten,
                     diagnostics=Diagnostics(backend="sklearn.PCA(full)"))


def factor_analysis(x, *, n_components, rotation, tolerance, maxiter):
    from sklearn.decomposition import FactorAnalysis
    from sklearn.exceptions import ConvergenceWarning

    model = FactorAnalysis(n_components=n_components, rotation=rotation,
                           svd_method="lapack", tol=tolerance, max_iter=maxiter)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x)
    failures = [w for w in caught if issubclass(w.category, ConvergenceWarning)]
    for warning in caught:
        if not issubclass(warning.category, ConvergenceWarning):
            warnings.warn(str(warning.message), warning.category, stacklevel=2)
    return FactorAnalysisResult(
        model.mean_, model.components_.T, model.noise_variance_, nobs=len(x),
        log_likelihood=model.loglike_, diagnostics=Diagnostics(
            backend="sklearn.FactorAnalysis(lapack)", converged=not failures,
            message="; ".join(str(w.message) for w in failures), iterations=model.n_iter_))
