"""Joint Gaussian predictions, including singular interpolation laws."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ..._typing import (
    FloatArray,
    SampleSize,
)
from .. import _validation as v


@dataclass(frozen=True)
class GaussianPrediction:
    """One joint law in query-row order; mean (m,), covariance (m,m).

    Roundoff-scale negative eigenvalues are clipped to zero. Materially
    indefinite matrices are rejected. No positive jitter is added.
    """

    mean: FloatArray
    covariance: FloatArray

    def __post_init__(self) -> None:
        mean = v.array(self.mean, "mean", ndim=1)
        cov = v.array(self.covariance, "covariance", ndim=2)
        if cov.shape != (len(mean), len(mean)):
            raise ValueError("covariance must align with mean")
        from ..._covariance import psd

        cov = psd(cov)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "covariance", v.array(cov, "covariance"))

    @property
    def event_shape(self):
        return self.mean.shape

    @property
    def variance(self) -> FloatArray:
        return v.array(np.maximum(np.diag(self.covariance), 0), "variance")

    @property
    def std(self) -> FloatArray:
        return v.array(np.sqrt(self.variance), "std")

    def sample(self, size: SampleSize = (), *, rng: np.random.Generator) -> FloatArray:
        """Correlated draws with shape size + (m,), using only the supplied RNG."""
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator")
        size = v.sample_shape(size)
        from ..._backends.gaussian import sample

        result = sample(self.mean, self.covariance, size, rng)
        return v.array(result, "samples", empty=True)

    def interval(self, probability: float = 0.95) -> FloatArray:
        """Pointwise equal-tail intervals (m,2); these are not a simultaneous band."""
        from scipy.special import ndtri

        p = v.scalar(probability, "probability")
        if not 0 < p < 1:
            raise ValueError("probability must lie strictly inside (0,1)")
        radius = -ndtri((1 - p) / 2) * self.std
        return v.array(
            np.column_stack((self.mean - radius, self.mean + radius)), "interval"
        )

    def log_density(self, values: ArrayLike):
        """SciPy Gaussian log density; singular laws use density on their support.

        A deterministic law has log density 0 at its mean and -inf elsewhere.
        """
        from ..._backends.gaussian import logpdf

        x = v.array(values, "values", empty=True)
        if x.ndim < 1 or x.shape[-1:] != self.event_shape:
            raise ValueError(
                f"values must have trailing event shape {self.event_shape}"
            )
        result = logpdf(x, self.mean, self.covariance)
        result = np.array(result, dtype=float).reshape(x.shape[:-1])
        if np.any(np.isnan(result)):
            raise ValueError("Gaussian log density returned NaN")
        result.setflags(write=False)
        return result

    def marginal(self, indices) -> GaussianPrediction:
        raw = np.asarray(indices)
        if raw.ndim != 1 or not raw.size or raw.dtype.kind not in "iu":
            raise ValueError("indices must be a nonempty integer vector")
        if (
            np.any(raw < 0)
            or np.any(raw >= len(self.mean))
            or len(np.unique(raw)) != len(raw)
        ):
            raise ValueError("indices must be unique and in range")
        return GaussianPrediction(self.mean[raw], self.covariance[np.ix_(raw, raw)])

    def __init__(
        self,
        mean: ArrayLike,
        covariance: ArrayLike,
    ) -> None:
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "covariance", covariance)
        self.__post_init__()
