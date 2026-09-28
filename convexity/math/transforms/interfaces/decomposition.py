"""Scaling, principal coordinates and Gaussian latent factor models."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .. import _validation as v
from .affine import AffineTransform
from .base import Diagnostics


@dataclass(frozen=True, init=False)
class FittedScaling(AffineTransform):
    scale: NDArray[Any]
    diagnostics: Diagnostics

    def __init__(self, location, scale, *, diagnostics=None):
        scales = v.array(scale, "scale", ndim=1)
        if np.any(scales <= 0):
            raise ValueError("scale must be strictly positive")
        super().__init__(location, np.diag(1 / scales), np.diag(scales))
        object.__setattr__(self, "scale", scales)
        object.__setattr__(self, "diagnostics", _diagnostics(diagnostics))


@dataclass(frozen=True, init=False)
class PCAResult(AffineTransform):
    """Principal axes in columns; variances use the explicit n-ddof divisor."""

    components: NDArray[Any]
    explained_variance: NDArray[Any]
    explained_variance_ratio: NDArray[Any]
    singular_values: NDArray[Any]
    nobs: int
    ddof: int
    whiten: bool
    diagnostics: Diagnostics

    def __init__(self, location, components, explained_variance, total_variance,
                 *, nobs, ddof=1, whiten=False, diagnostics=None):
        basis = v.array(components, "components", ndim=2)
        variance = v.array(explained_variance, "explained_variance", ndim=1)
        k = basis.shape[1]
        if (variance.shape != (k,) or np.any(variance < 0)
                or not np.allclose(basis.T @ basis, np.eye(k), rtol=1e-8, atol=1e-10)):
            raise ValueError("components must be orthonormal columns with k nonnegative variances")
        n = v.integer(nobs, "nobs", 2)
        divisor = v.integer(ddof, "ddof")
        v.choice(divisor, (0, 1), "ddof")
        if k > min(n, basis.shape[0]):
            raise ValueError("component count cannot exceed min(nobs, dimension)")
        total = v.scalar(total_variance, "total_variance", positive=True)
        if variance.sum() > total * (1 + 1e-8):
            raise ValueError("explained variance cannot exceed total variance")
        v.boolean(whiten, "whiten")
        if whiten and np.any(variance <= np.finfo(float).eps * max(n, basis.shape[0]) * total):
            raise ValueError("cannot whiten numerically zero-variance components; reduce n_components")
        scales = np.sqrt(variance) if whiten else np.ones(k)
        super().__init__(location, basis / scales, scales[:, None] * basis.T)
        values = dict(components=basis, explained_variance=variance,
                      explained_variance_ratio=v.array(variance / total, "variance ratio"),
                      singular_values=v.array(np.sqrt(variance * (n - divisor)), "singular values"),
                      nobs=n, ddof=divisor, whiten=bool(whiten), diagnostics=_diagnostics(diagnostics))
        for name, value in values.items():
            object.__setattr__(self, name, value)


@dataclass(frozen=True, init=False)
class GaussianFactorModel:
    """x = location + loadings @ f + epsilon, f~N(0,I), epsilon~N(0,diag(noise)).

    Factors and noise are independent. transform returns E[f|x], not an
    orthogonal projection or an exact inverse of the noisy generative model.
    """

    location: NDArray[Any]
    loadings: NDArray[Any]
    noise_variance: NDArray[Any]

    def __init__(self, location, loadings, noise_variance):
        loc = v.array(location, "location", ndim=1)
        basis = v.array(loadings, "loadings", ndim=2)
        noise = v.array(noise_variance, "noise_variance", ndim=1)
        if basis.shape[0] != len(loc) or noise.shape != loc.shape or basis.shape[1] > len(loc):
            raise ValueError("loadings must be (d,k), k<=d; location and noise_variance (d,)")
        if np.any(noise <= 0):
            raise ValueError("noise_variance must be strictly positive")
        for name, value in dict(location=loc, loadings=basis, noise_variance=noise).items():
            object.__setattr__(self, name, value)

    @property
    def input_dimension(self):
        return len(self.location)

    @property
    def output_dimension(self):
        return self.loadings.shape[1]

    @property
    def covariance(self):
        return v.array(self.loadings @ self.loadings.T + np.diag(self.noise_variance), "covariance")

    @property
    def posterior_covariance(self):
        from .._backends.scipy import factor_posterior
        return v.array(factor_posterior(self)[1], "posterior covariance")

    def transform(self, values):
        from .._backends.scipy import factor_posterior
        x = v.coordinates(values, self.input_dimension, "values")
        encoder, _ = factor_posterior(self)
        return v.array((x - self.location) @ encoder, "posterior factor means")

    def reconstruct(self, factors):
        z = v.coordinates(factors, self.output_dimension, "factors")
        return v.array(z @ self.loadings.T + self.location, "common component")

    def residuals(self, values):
        x = v.coordinates(values, self.input_dimension, "values")
        return v.array(x - self.reconstruct(self.transform(x)), "residuals")


@dataclass(frozen=True, init=False)
class FactorAnalysisResult(GaussianFactorModel):
    diagnostics: Diagnostics
    log_likelihood: NDArray[Any]
    nobs: int

    def __init__(self, location, loadings, noise_variance, *, diagnostics, log_likelihood, nobs):
        super().__init__(location, loadings, noise_variance)
        object.__setattr__(self, "diagnostics", _diagnostics(diagnostics))
        object.__setattr__(self, "log_likelihood", v.array(log_likelihood, "log_likelihood", ndim=1))
        object.__setattr__(self, "nobs", v.integer(nobs, "nobs", 2))


def _diagnostics(value):
    if value is not None and not isinstance(value, Diagnostics):
        raise TypeError("diagnostics must be Diagnostics")
    return Diagnostics() if value is None else value
