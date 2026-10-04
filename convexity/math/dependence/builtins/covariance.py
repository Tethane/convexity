"""Reusable covariance estimation specifications."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from .. import CovarianceEstimate, FactorCovariance


from dataclasses import dataclass

from .. import _validation as v
from .._backends import covariance as backend


def _centering(ddof, assume_centered):
    v.integer(ddof, "ddof")
    v.choice(ddof, (0, 1), "ddof")
    v.boolean(assume_centered, "assume_centered")


@dataclass(frozen=True)
class SampleCovariance:
    """Empirical covariance from observation rows; ddof and assumed centering are explicit."""

    ddof: int = 1
    assume_centered: bool = False

    def __post_init__(self) -> None:
        _centering(self.ddof, self.assume_centered)

    def fit(self, samples: ArrayLike) -> CovarianceEstimate:
        x = v.samples(samples, minimum=max(2, self.ddof + 1))
        return backend.empirical(
            x, ddof=self.ddof, assume_centered=self.assume_centered
        )


@dataclass(frozen=True)
class EWMACovariance:
    """Exponentially weighted covariance specification with explicit decay and centering settings."""

    decay: float = 0.94
    ddof: int = 0
    assume_centered: bool = False

    def __post_init__(self) -> None:
        _centering(self.ddof, self.assume_centered)
        decay = v.scalar(self.decay, "decay")
        if not 0 < decay <= 1:
            raise ValueError("decay must lie in (0,1]")
        object.__setattr__(self, "decay", decay)

    def fit(self, samples: ArrayLike) -> CovarianceEstimate:
        x = v.samples(samples, minimum=1 + self.ddof)
        return backend.ewma(
            x, decay=self.decay, ddof=self.ddof, assume_centered=self.assume_centered
        )


@dataclass(frozen=True)
class ShrunkCovariance:
    """Blend empirical covariance with a scaled identity target using an explicit shrinkage strength."""

    shrinkage: float = 0.1
    assume_centered: bool = False

    def __post_init__(self) -> None:
        v.boolean(self.assume_centered, "assume_centered")
        amount = v.scalar(self.shrinkage, "shrinkage")
        if not 0 <= amount <= 1:
            raise ValueError("shrinkage must lie in [0,1]")
        object.__setattr__(self, "shrinkage", amount)

    def fit(self, samples: ArrayLike) -> CovarianceEstimate:
        return backend.shrinkage(
            v.samples(samples),
            method="fixed",
            amount=self.shrinkage,
            assume_centered=self.assume_centered,
        )


@dataclass(frozen=True)
class LedoitWolf:
    """Estimate covariance using data-driven shrinkage toward a scaled identity target."""

    assume_centered: bool = False

    def __post_init__(self) -> None:
        v.boolean(self.assume_centered, "assume_centered")

    def fit(self, samples: ArrayLike) -> CovarianceEstimate:
        return backend.shrinkage(
            v.samples(samples),
            method="ledoit_wolf",
            amount=None,
            assume_centered=self.assume_centered,
        )


@dataclass(frozen=True)
class OAS:
    """Estimate covariance using oracle-approximating shrinkage toward a scaled identity target."""

    assume_centered: bool = False

    def __post_init__(self) -> None:
        v.boolean(self.assume_centered, "assume_centered")

    def fit(self, samples: ArrayLike) -> CovarianceEstimate:
        return backend.shrinkage(
            v.samples(samples),
            method="oas",
            amount=None,
            assume_centered=self.assume_centered,
        )


def sample_covariance(
    samples: ArrayLike, *, ddof: int = 1, assume_centered: bool = False
) -> CovarianceEstimate:
    """Estimate covariance from samples (observation,coordinate).

    ddof selects n-ddof normalization; assume_centered=True uses zero rather than
    an estimated mean. Return an owned CovarianceEstimate."""
    return SampleCovariance(ddof, assume_centered).fit(samples)


def ewma_covariance(
    samples: ArrayLike,
    *,
    decay: float = 0.94,
    ddof: int = 0,
    assume_centered: bool = False,
) -> CovarianceEstimate:
    """Fit exponentially weighted covariance to time-ordered sample rows.

    Decay in (0,1) discounts older rows; ddof controls the weight-based correction.
    Return CovarianceEstimate in squared feature units."""
    return EWMACovariance(decay, ddof, assume_centered).fit(samples)


def fit_factor_covariance(
    samples: ArrayLike, factors: ArrayLike, *, ddof: int = 1, residual: str = "diagonal"
) -> FactorCovariance:
    """OLS exposures to supplied factors; factor/noise moments use n-ddof.

    Full residual covariance reproduces sample covariance. Diagonal residuals
    impose a model assumption. Neither estimates regression standard errors.
    """
    _centering(ddof, False)
    v.choice(residual, ("diagonal", "full"), "residual")
    x, f = v.samples(samples), v.samples(factors)
    if len(x) != len(f) or f.shape[1] >= len(f) - 1:
        raise ValueError(
            "factors must align with observations and leave residual degrees of freedom"
        )
    return backend.observed_factors(x, f, ddof=ddof, residual=residual)
