"""Generalized linear conditional means with explicit link and count semantics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict, Unpack

import numpy as np
from numpy.typing import ArrayLike

from .. import _validation as v
from .._backends import statsmodels
from ..interfaces.base import FittedModel, checked_convergence


class GeneralizedLinearModelOptions(TypedDict, total=False):
    """Optional keyword settings for generalized_linear_model."""

    family: str
    fit_intercept: bool
    tolerance: float
    maxiter: int
    check_convergence: bool


@dataclass(frozen=True)
class GeneralizedLinearModel:
    """Gaussian/identity, Poisson/log, Gamma/log, or Binomial/logit.

    Binomial response is a success count when trials is supplied, else 0/1.
    Its predicted mean is always the probability per trial, not a count.
    """

    family: str = "gaussian"
    fit_intercept: bool = True
    tolerance: float = 1e-8
    maxiter: int = 100
    check_convergence: bool = True

    def __post_init__(self) -> None:
        v.choice(self.family, ("gaussian", "poisson", "gamma", "binomial"), "family")
        v.boolean(self.fit_intercept, "fit_intercept")
        object.__setattr__(
            self,
            "tolerance",
            v.scalar(self.tolerance, "tolerance", minimum=0, strict=True),
        )
        v.integer(self.maxiter, "maxiter", 1)
        v.boolean(self.check_convergence, "check_convergence")

    def fit(
        self,
        features: ArrayLike,
        response: ArrayLike,
        *,
        frequency_weights: ArrayLike | None = None,
        trials: ArrayLike | None = None,
        offset: ArrayLike | None = None,
    ) -> FittedModel:
        x = v.features(features)
        y = v.response(response, len(x), multi=False)
        w = v.weights(frequency_weights, len(x))
        if w is not None and np.any(w != np.floor(w)):
            raise ValueError(
                "frequency_weights must be nonnegative integer replication counts"
            )
        if trials is not None:
            if self.family != "binomial":
                raise ValueError("trials is only supported for binomial models")
            trials = v.array(trials, "trials", ndim=1)
            if (
                trials.shape != y.shape
                or np.any(trials < 1)
                or np.any(trials != np.floor(trials))
            ):
                raise ValueError(
                    "trials must be positive integers aligned with response"
                )
        if self.family in ("poisson", "binomial"):
            if np.any(y < 0) or np.any(y != np.floor(y)):
                raise ValueError("count responses must be nonnegative integers")
        if self.family == "binomial" and np.any(y > (1 if trials is None else trials)):
            raise ValueError("success counts cannot exceed trials (one by default)")
        if self.family == "gamma" and np.any(y <= 0):
            raise ValueError("Gamma responses must be strictly positive")
        if offset is not None:
            offset = v.array(offset, "offset", ndim=1)
            if offset.shape != y.shape:
                raise ValueError("offset must align with training rows")
        if w is not None:
            keep = w > 0
            x, y, w = x[keep], y[keep], w[keep]
            trials = None if trials is None else trials[keep]
            offset = None if offset is None else offset[keep]
        result = statsmodels.glm(self, x, y, w, trials, offset)
        return checked_convergence(result, self.check_convergence)


def generalized_linear_model(
    features: ArrayLike,
    response: ArrayLike,
    *,
    frequency_weights: ArrayLike | None = None,
    trials: ArrayLike | None = None,
    offset: ArrayLike | None = None,
    **kwargs: Unpack[GeneralizedLinearModelOptions],
) -> FittedModel:
    """Fit GeneralizedLinearModel to features (observation,feature) and aligned response.

    Gaussian/identity, Poisson/log, Gamma/log, or Binomial/logit.
    Return the checked fitted predictor; no feature scaling is implicit.
    Optional settings: family, fit_intercept, tolerance, maxiter, check_convergence."""
    return GeneralizedLinearModel(**kwargs).fit(
        features,
        response,
        frequency_weights=frequency_weights,
        trials=trials,
        offset=offset,
    )
