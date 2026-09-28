"""Joint Gaussian predictions, including singular interpolation laws."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from .. import _validation as v


@dataclass(frozen=True)
class GaussianPrediction:
    """One joint law in query-row order; mean (m,), covariance (m,m).

    Roundoff-scale negative eigenvalues are clipped to zero. Materially
    indefinite matrices are rejected. No positive jitter is added.
    """

    mean: Any
    covariance: Any

    def __post_init__(self):
        mean = v.array(self.mean, "mean", ndim=1)
        cov = v.array(self.covariance, "covariance", ndim=2)
        if cov.shape != (len(mean), len(mean)):
            raise ValueError("covariance must align with mean")
        tolerance = 100 * np.finfo(float).eps * len(mean) * max(1., float(np.max(np.abs(cov))))
        if not np.allclose(cov, cov.T, atol=tolerance, rtol=0):
            raise ValueError("covariance must be symmetric")
        cov = (cov + cov.T) / 2
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        if eigenvalues[0] < -tolerance:
            raise ValueError("covariance must be positive semidefinite")
        if eigenvalues[0] < 0:
            cov = (eigenvectors * np.maximum(eigenvalues, 0)) @ eigenvectors.T
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "covariance", v.array(cov, "covariance"))

    @property
    def event_shape(self):
        return self.mean.shape

    @property
    def variance(self):
        return v.array(np.maximum(np.diag(self.covariance), 0), "variance")

    @property
    def std(self):
        return v.array(np.sqrt(self.variance), "std")

    def sample(self, size=(), *, rng):
        """Correlated draws with shape size + (m,), using only the supplied RNG."""
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator")
        size = v.sample_shape(size)
        result = rng.multivariate_normal(self.mean, self.covariance, size=size, method="eigh")
        return v.array(result, "samples", empty=True)

    def interval(self, probability=.95):
        """Pointwise equal-tail intervals (m,2); these are not a simultaneous band."""
        from scipy.special import ndtri

        p = v.scalar(probability, "probability")
        if not 0 < p < 1:
            raise ValueError("probability must lie strictly inside (0,1)")
        radius = -ndtri((1 - p) / 2) * self.std
        return v.array(np.column_stack((self.mean - radius, self.mean + radius)), "interval")

    def log_density(self, values):
        """SciPy Gaussian log density; singular laws use density on their support.

        A deterministic law has log density 0 at its mean and -inf elsewhere.
        """
        from scipy.stats import multivariate_normal

        x = v.array(values, "values", empty=True)
        if x.ndim < 1 or x.shape[-1:] != self.event_shape:
            raise ValueError(f"values must have trailing event shape {self.event_shape}")
        if not np.any(self.covariance):
            result = np.where(np.all(x == self.mean, axis=-1), 0., -np.inf)
        else:
            result = multivariate_normal.logpdf(x, mean=self.mean, cov=self.covariance, allow_singular=True)
        result = np.array(result, dtype=float).reshape(x.shape[:-1])
        if np.any(np.isnan(result)):
            raise ValueError("Gaussian log density returned NaN")
        result.setflags(write=False)
        return result

    def marginal(self, indices):
        raw = np.asarray(indices)
        if raw.ndim != 1 or not raw.size or raw.dtype.kind not in "iu":
            raise ValueError("indices must be a nonempty integer vector")
        if np.any(raw < 0) or np.any(raw >= len(self.mean)) or len(np.unique(raw)) != len(raw):
            raise ValueError("indices must be unique and in range")
        return GaussianPrediction(self.mean[raw], self.covariance[np.ix_(raw, raw)])
