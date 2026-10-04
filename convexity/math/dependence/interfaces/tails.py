"""Inspectable Student-t and extreme-value fits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ..._typing import FloatArray
from .. import _validation as v
from .base import Diagnostics, diagnostics


@dataclass(frozen=True, init=False)
class StudentTResult:
    """Fitted multivariate Student-t parameters and diagnostics; scale and covariance are distinct quantities."""

    location: FloatArray
    shape: FloatArray
    df: float
    nobs: int
    log_likelihood: float
    diagnostics: Diagnostics

    def __init__(
        self,
        location: ArrayLike,
        shape: ArrayLike,
        df: float,
        *,
        nobs: int,
        log_likelihood: float,
        diagnostics: Diagnostics | None = None,
    ) -> None:
        loc, matrix = v.array(location, "location", ndim=1), v.elliptical_shape(shape)
        if matrix.shape != (len(loc), len(loc)):
            raise ValueError("shape and location dimensions must agree")
        values = dict(
            location=loc,
            shape=matrix,
            df=v.scalar(df, "df", positive=True),
            nobs=v.integer(nobs, "nobs", 1),
            log_likelihood=v.scalar(log_likelihood, "log_likelihood"),
            diagnostics=_report(diagnostics),
        )
        for name, value in values.items():
            object.__setattr__(self, name, value)

    @property
    def covariance(self) -> FloatArray:
        if self.df <= 2:
            raise ValueError("Student-t has no finite covariance for df <= 2")
        return v.array(self.shape * (self.df / (self.df - 2)), "covariance")

    @property
    def distribution(self):
        from ...distributions import multivariate_student_t

        return multivariate_student_t(self.df, self.location, self.shape)


_report = diagnostics


@dataclass(frozen=True)
class TailFit:
    """xi uses the extreme-value convention (SciPy genextreme shape is -xi)."""

    family: str
    xi: float
    location: float
    scale: float
    nobs: int
    log_likelihood: float
    diagnostics: Diagnostics = Diagnostics()

    def __post_init__(self) -> None:
        v.choice(self.family, ("gpd", "gev"), "family")
        object.__setattr__(self, "xi", v.scalar(self.xi, "xi"))
        object.__setattr__(self, "location", v.scalar(self.location, "location"))
        object.__setattr__(self, "scale", v.scalar(self.scale, "scale", positive=True))
        v.integer(self.nobs, "nobs", 1)
        object.__setattr__(
            self, "log_likelihood", v.scalar(self.log_likelihood, "log_likelihood")
        )
        diagnostics(self.diagnostics)
        if self.family == "gpd" and self.location != 0:
            raise ValueError(
                "GPD fits describe nonnegative excesses with location zero"
            )

    @property
    def distribution(self):
        from ..builtins.tails import generalized_extreme_value, generalized_pareto

        if self.family == "gpd":
            return generalized_pareto(self.xi, scale=self.scale)
        return generalized_extreme_value(self.xi, loc=self.location, scale=self.scale)

    def return_level(self, period: ArrayLike) -> FloatArray:
        """GEV return period measured in supplied blocks; GPD in excess events."""
        from .._backends.tails import frozen_tail

        t = v.array(period, "period")
        if np.any(t <= 1):
            raise ValueError("return period must exceed one")
        return v.array(
            frozen_tail(self.family, self.xi, self.location, self.scale).isf(1 / t),
            "return level",
            finite=False,
        )


@dataclass(frozen=True)
class PeaksOverThresholdResult:
    """Threshold, fitted excess law and sample count needed to interpret tail exceedance probabilities."""

    threshold: float
    tail: TailFit
    nobs: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "threshold", v.scalar(self.threshold, "threshold"))
        v.integer(self.nobs, "nobs", 1)
        if (
            not isinstance(self.tail, TailFit)
            or self.tail.family != "gpd"
            or self.tail.nobs > self.nobs
        ):
            raise ValueError(
                "tail must be a GPD fit with no more excesses than observations"
            )

    @property
    def exceedance_rate(self):
        return self.tail.nobs / self.nobs

    @property
    def diagnostics(self):
        return self.tail.diagnostics

    def survival(self, values: ArrayLike) -> FloatArray:
        x = v.array(values, "values", finite=False, empty=True)
        if np.any(x < self.threshold):
            raise ValueError("POT only defines the upper tail at or above threshold")
        return v.array(
            self.exceedance_rate * self.tail.distribution.survival(x - self.threshold),
            "tail probability",
            empty=True,
        )

    def quantile(self, probability: ArrayLike) -> FloatArray:
        from .._backends.tails import frozen_tail

        p = v.probabilities(probability)
        if np.any(p < 1 - self.exceedance_rate):
            raise ValueError("probability falls below the fitted tail")
        law = frozen_tail("gpd", self.tail.xi, 0.0, self.tail.scale)
        return v.array(
            self.threshold + law.isf(np.minimum((1 - p) / self.exceedance_rate, 1.0)),
            "tail quantile",
            finite=False,
            empty=True,
        )

    def return_level(self, period: ArrayLike) -> FloatArray:
        """Expected exceedance interval in original observation units."""
        from .._backends.tails import frozen_tail

        t = v.array(period, "period")
        if np.any(t < 1 / self.exceedance_rate):
            raise ValueError("return period must be at least 1 / exceedance_rate")
        law = frozen_tail("gpd", self.tail.xi, 0.0, self.tail.scale)
        return v.array(
            self.threshold + law.isf((1 / t) / self.exceedance_rate),
            "return level",
            finite=False,
        )
