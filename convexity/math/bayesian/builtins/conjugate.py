"""Exact conjugate posterior updates, with explicit variance/rate conventions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy.typing import ArrayLike

from ..._typing import FloatArray

if TYPE_CHECKING:
    from ...distributions import Distribution


from dataclasses import dataclass, replace

import numpy as np

from .. import _validation as v


def _law(name, *args, discrete=False, **kwargs):
    from scipy import stats

    from ...distributions import Distribution
    from ...distributions._backends.scipy import ScalarAdapter

    return Distribution(
        ScalarAdapter(
            getattr(stats, name)(*args, **kwargs),
            "discrete" if discrete else "continuous",
        )
    )


def beta(alpha: float = 1.0, beta: float = 1.0) -> Distribution:
    """Beta law on a probability; alpha and beta are positive shapes."""
    return _law("beta", v.positive(alpha, "alpha"), v.positive(beta, "beta"))


def gamma(shape: float, rate: float = 1.0) -> Distribution:
    """Gamma law parameterized by rate, not scale."""
    return _law("gamma", v.positive(shape, "shape"), scale=1 / v.positive(rate, "rate"))


def inverse_gamma(shape: float, scale: float = 1.0) -> Distribution:
    """Return the scalar inverse-gamma law with positive shape and scale.

    Density is proportional to x**(-shape-1)*exp(-scale/x) for x>0.
    Moments exist only above their corresponding shape thresholds."""
    return _law(
        "invgamma", v.positive(shape, "shape"), scale=v.positive(scale, "scale")
    )


@dataclass(frozen=True)
class BetaBinomial:
    """Beta prior on a success probability; update by binomial counts and predict future counts."""

    alpha: float = 1.0
    beta: float = 1.0

    def __post_init__(self) -> None:
        v.positive(self.alpha, "alpha")
        v.positive(self.beta, "beta")

    def update(self, successes, trials: ArrayLike = 1):
        """Independent binomial rows; scalar trials broadcasts to each row."""
        y = v.counts(successes, "successes")
        n = v.array(trials, "trials")
        if n.ndim > 1 or (n.ndim == 1 and n.shape != y.shape):
            raise ValueError("trials must be scalar or align with successes")
        if np.any(n < 0) or np.any(n != np.floor(n)) or np.any(y > n):
            raise ValueError("require integer trials >= successes >= 0")
        return replace(
            self,
            alpha=self.alpha + float(y.sum()),
            beta=self.beta + float(np.sum(n - y)),
        )

    @property
    def posterior(self):
        return beta(self.alpha, self.beta)

    def predictive(self, trials: int = 1):
        """Beta-binomial count law, integrating probability uncertainty."""
        return _law(
            "betabinom",
            v.integer(trials, "trials"),
            self.alpha,
            self.beta,
            discrete=True,
        )


@dataclass(frozen=True)
class GammaPoisson:
    """Gamma prior on a Poisson rate with explicit exposure; update counts and predict future observations."""

    shape: float = 1.0
    rate: float = 1.0

    def __post_init__(self) -> None:
        v.positive(self.shape, "shape")
        v.positive(self.rate, "rate")

    def update(self, counts, exposure: ArrayLike = 1.0):
        y = v.counts(counts)
        t = v.array(exposure, "exposure")
        if t.ndim > 1 or (t.ndim == 1 and t.shape != y.shape) or np.any(t <= 0):
            raise ValueError("exposure must be positive, scalar or aligned with counts")
        return replace(
            self,
            shape=self.shape + float(y.sum()),
            rate=self.rate + float(np.sum(np.broadcast_to(t, y.shape))),
        )

    @property
    def posterior(self):
        return gamma(self.shape, self.rate)

    def predictive(self, exposure: float = 1.0):
        exposure = v.positive(exposure, "exposure")
        return _law(
            "nbinom", self.shape, self.rate / (self.rate + exposure), discrete=True
        )


@dataclass(frozen=True)
class NormalMean:
    """Unknown Gaussian mean, with known independent observation variance."""

    mean: float = 0.0
    variance: float = 1.0
    observation_variance: float = 1.0

    def __post_init__(self) -> None:
        v.scalar(self.mean, "mean")
        v.positive(self.variance, "variance")
        v.positive(self.observation_variance, "observation_variance")

    def update(self, observations: ArrayLike):
        y = v.array(observations, "observations", ndim=1, empty=True)
        variance = 1 / (1 / self.variance + len(y) / self.observation_variance)
        return replace(
            self,
            mean=variance
            * (self.mean / self.variance + float(y.sum()) / self.observation_variance),
            variance=variance,
        )

    @property
    def posterior(self):
        from ...distributions import normal

        return normal(self.mean, np.sqrt(self.variance))

    def predictive(self):
        from ...distributions import normal

        return normal(self.mean, np.sqrt(self.variance + self.observation_variance))


@dataclass(frozen=True)
class NormalInverseGamma:
    """sigma² ~ IG(shape,scale); mu | sigma² ~ N(mean,sigma²/precision)."""

    mean: float = 0.0
    precision: float = 1.0
    shape: float = 2.0
    scale: float = 1.0

    def __post_init__(self) -> None:
        v.scalar(self.mean, "mean")
        for name in ("precision", "shape", "scale"):
            v.positive(getattr(self, name), name)

    def update(self, observations: ArrayLike):
        y = v.array(observations, "observations", ndim=1, empty=True)
        if not len(y):
            return self
        n, average = len(y), float(y.mean())
        precision = self.precision + n
        return replace(
            self,
            mean=(self.precision * self.mean + n * average) / precision,
            precision=precision,
            shape=self.shape + n / 2,
            scale=self.scale
            + float(np.sum((y - average) ** 2)) / 2
            + self.precision * n * (average - self.mean) ** 2 / (2 * precision),
        )

    @property
    def posterior(self):
        """Joint vector law in (mean, variance) order; parameters are dependent."""
        from ...distributions import Distribution
        from .._backends.conjugate import NormalInverseGammaAdapter

        return Distribution(NormalInverseGammaAdapter(self))

    @property
    def mean_posterior(self):
        from ...distributions import student_t

        return student_t(
            2 * self.shape,
            self.mean,
            np.sqrt(self.scale / (self.shape * self.precision)),
        )

    @property
    def variance_posterior(self):
        return inverse_gamma(self.shape, self.scale)

    def predictive(self):
        from ...distributions import student_t

        return student_t(
            2 * self.shape,
            self.mean,
            np.sqrt(self.scale * (self.precision + 1) / (self.shape * self.precision)),
        )


@dataclass(frozen=True)
class GaussianLinearRegression:
    """Gaussian coefficient prior, known scalar noise variance, explicit design.

    No intercept is inserted. predictive returns a joint law sharing the same
    coefficient uncertainty across all query rows.
    """

    mean: FloatArray
    covariance: FloatArray
    observation_variance: float = 1.0

    def __post_init__(self) -> None:
        mean = v.array(self.mean, "mean", ndim=1)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "covariance", v.covariance(self.covariance, len(mean)))
        v.positive(self.observation_variance, "observation_variance")

    def _features(self, features):
        x = v.array(features, "features", ndim=2)
        if x.shape[1] != len(self.mean):
            raise ValueError("features must have one column per coefficient")
        return x

    def update(self, features: ArrayLike, response: ArrayLike):
        from scipy.linalg import cho_factor, cho_solve

        x = self._features(features)
        y = v.array(response, "response", ndim=1)
        if y.shape != (len(x),):
            raise ValueError("response must align with feature rows")
        precision = cho_solve(cho_factor(self.covariance), np.eye(len(self.mean)))
        factor = cho_factor(precision + x.T @ x / self.observation_variance)
        cov = cho_solve(factor, np.eye(len(self.mean)))
        mean = cho_solve(
            factor, precision @ self.mean + x.T @ y / self.observation_variance
        )
        return replace(self, mean=mean, covariance=(cov + cov.T) / 2)

    @property
    def posterior(self):
        from ...distributions import multivariate_normal

        return multivariate_normal(self.mean, self.covariance)

    def predictive(self, features: ArrayLike, *, include_noise: bool = True):
        from ...state_space import GaussianState

        v.boolean(include_noise, "include_noise")
        x = self._features(features)
        cov = x @ self.covariance @ x.T
        if include_noise:
            cov = cov + self.observation_variance * np.eye(len(x))
        # Latent predictions can be singular, e.g. repeated design rows.
        return GaussianState(x @ self.mean, cov).distribution()

    def __init__(
        self,
        mean: ArrayLike,
        covariance: ArrayLike,
        observation_variance: float = 1.0,
    ) -> None:
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "observation_variance", observation_variance)
        self.__post_init__()
