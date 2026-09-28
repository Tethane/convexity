"""Reusable covariance estimation specifications."""

from dataclasses import dataclass

from .. import _validation as v
from .._backends import covariance as backend


def _centering(ddof, assume_centered):
    v.integer(ddof, "ddof")
    v.choice(ddof, (0, 1), "ddof")
    v.boolean(assume_centered, "assume_centered")


@dataclass(frozen=True)
class SampleCovariance:
    ddof: int = 1
    assume_centered: bool = False

    def __post_init__(self):
        _centering(self.ddof, self.assume_centered)

    def fit(self, samples):
        x = v.samples(samples, minimum=max(2, self.ddof + 1))
        return backend.empirical(x, ddof=self.ddof, assume_centered=self.assume_centered)


@dataclass(frozen=True)
class EWMACovariance:
    decay: float = .94
    ddof: int = 0
    assume_centered: bool = False

    def __post_init__(self):
        _centering(self.ddof, self.assume_centered)
        decay = v.scalar(self.decay, "decay")
        if not 0 < decay <= 1:
            raise ValueError("decay must lie in (0,1]")
        object.__setattr__(self, "decay", decay)

    def fit(self, samples):
        x = v.samples(samples, minimum=1 + self.ddof)
        return backend.ewma(x, decay=self.decay, ddof=self.ddof, assume_centered=self.assume_centered)


@dataclass(frozen=True)
class ShrunkCovariance:
    shrinkage: float = .1
    assume_centered: bool = False

    def __post_init__(self):
        v.boolean(self.assume_centered, "assume_centered")
        amount = v.scalar(self.shrinkage, "shrinkage")
        if not 0 <= amount <= 1:
            raise ValueError("shrinkage must lie in [0,1]")
        object.__setattr__(self, "shrinkage", amount)

    def fit(self, samples):
        return backend.shrinkage(v.samples(samples), method="fixed", amount=self.shrinkage,
                                 assume_centered=self.assume_centered)


@dataclass(frozen=True)
class LedoitWolf:
    assume_centered: bool = False

    def __post_init__(self):
        v.boolean(self.assume_centered, "assume_centered")

    def fit(self, samples):
        return backend.shrinkage(v.samples(samples), method="ledoit_wolf", amount=None,
                                 assume_centered=self.assume_centered)


@dataclass(frozen=True)
class OAS:
    assume_centered: bool = False

    def __post_init__(self):
        v.boolean(self.assume_centered, "assume_centered")

    def fit(self, samples):
        return backend.shrinkage(v.samples(samples), method="oas", amount=None,
                                 assume_centered=self.assume_centered)


def sample_covariance(samples, *, ddof=1, assume_centered=False):
    return SampleCovariance(ddof, assume_centered).fit(samples)


def ewma_covariance(samples, *, decay=.94, ddof=0, assume_centered=False):
    return EWMACovariance(decay, ddof, assume_centered).fit(samples)


def fit_factor_covariance(samples, factors, *, ddof=1, residual="diagonal"):
    """OLS exposures to supplied factors; factor/noise moments use n-ddof.

    Full residual covariance reproduces sample covariance. Diagonal residuals
    impose a model assumption. Neither estimates regression standard errors.
    """
    _centering(ddof, False)
    v.choice(residual, ("diagonal", "full"), "residual")
    x, f = v.samples(samples), v.samples(factors)
    if len(x) != len(f) or f.shape[1] >= len(f) - 1:
        raise ValueError("factors must align with observations and leave residual degrees of freedom")
    return backend.observed_factors(x, f, ddof=ddof, residual=residual)
