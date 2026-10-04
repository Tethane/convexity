"""Copula constructors and explicitly rank-based parameter estimation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy.typing import ArrayLike

from ..._typing import FloatArray

if TYPE_CHECKING:
    from .. import CopulaFit


from dataclasses import dataclass
from typing import cast

import numpy as np

from .. import _validation as v
from .._backends.copulas import Adapter, fit_parameter
from ..interfaces.base import Diagnostics
from ..interfaces.copulas import Copula, CopulaFit
from .matrices import nearest_correlation


@dataclass(frozen=True, init=False)
class IndependenceCopula(Copula):
    """Product copula of the declared dimension; supplies no cross-coordinate dependence."""

    def __init__(self, dimension: int = 2) -> None:
        super().__init__(Adapter("independence", v.integer(dimension, "dimension", 2)))


@dataclass(frozen=True, init=False)
class GaussianCopula(Copula):
    """Gaussian dependence parameterized by a correlation matrix; marginal distributions are supplied separately."""

    def __init__(self, correlation: ArrayLike) -> None:
        corr = v.correlation(correlation, definite=True)
        super().__init__(Adapter("gaussian", len(corr), correlation=corr))

    @property
    def correlation(self):
        return cast(Adapter, self.adapter).correlation


@dataclass(frozen=True, init=False)
class StudentTCopula(Copula):
    """Student-t dependence with a correlation matrix and degrees of freedom; margins remain separate."""

    def __init__(self, correlation: ArrayLike, df: float) -> None:
        corr = v.correlation(correlation, definite=True)
        super().__init__(
            Adapter(
                "student_t",
                len(corr),
                correlation=corr,
                df=v.scalar(df, "df", positive=True),
            )
        )

    @property
    def correlation(self):
        return cast(Adapter, self.adapter).correlation

    @property
    def df(self):
        return cast(Adapter, self.adapter).df


@dataclass(frozen=True, init=False)
class ClaytonCopula(Copula):
    """Archimedean Clayton dependence for unit-interval margins; parameter restrictions depend on dimension."""

    def __init__(self, theta: float) -> None:
        super().__init__(
            Adapter("clayton", 2, theta=v.scalar(theta, "theta", positive=True))
        )

    @property
    def theta(self):
        return cast(Adapter, self.adapter).theta


@dataclass(frozen=True, init=False)
class GumbelCopula(Copula):
    """Archimedean Gumbel dependence for unit-interval margins; theta controls upper-tail dependence."""

    def __init__(self, theta: float) -> None:
        amount = v.scalar(theta, "theta")
        if amount < 1:
            raise ValueError("Gumbel theta must be >= 1")
        super().__init__(Adapter("gumbel", 2, theta=amount))

    @property
    def theta(self):
        return cast(Adapter, self.adapter).theta


def pseudo_observations(samples: ArrayLike) -> FloatArray:
    """Columnwise average ranks divided by n+1; ties retain their average rank."""
    from scipy.stats import rankdata

    x = v.samples(samples)
    return v.array(
        rankdata(x, axis=0, method="average") / (len(x) + 1), "pseudo observations"
    )


@dataclass(frozen=True)
class KendallCopula:
    """Estimate copula dependence from Kendall rank correlations, with explicit family and repair choices."""

    family: str = "gaussian"
    df: float | None = None
    repair_correlation: bool = False

    def __post_init__(self) -> None:
        v.choice(self.family, ("gaussian", "student_t", "clayton", "gumbel"), "family")
        v.boolean(self.repair_correlation, "repair_correlation")
        if self.family == "student_t":
            object.__setattr__(self, "df", v.scalar(self.df, "df", positive=True))
        elif self.df is not None:
            raise ValueError("df is only used by the Student-t copula")
        if self.family in ("clayton", "gumbel") and self.repair_correlation:
            raise ValueError("correlation repair is only for elliptical copulas")

    def fit(self, samples: ArrayLike) -> CopulaFit:
        x = v.samples(samples, minimum=3)
        if x.shape[1] < 2 or np.any(np.ptp(x, axis=0) == 0):
            raise ValueError("copula fitting needs at least two nonconstant columns")
        if self.family in ("clayton", "gumbel") and x.shape[1] != 2:
            raise ValueError("Archimedean adapters are bivariate")
        parameter = fit_parameter(self.family, x, self.df)
        distance = 0.0
        if self.family in ("gaussian", "student_t"):
            if self.repair_correlation:
                repair = nearest_correlation(parameter, min_eigenvalue=1e-8)
                parameter, distance = repair.matrix, repair.distance
            if self.family == "gaussian":
                copula = GaussianCopula(parameter)
            else:
                assert self.df is not None
                copula = StudentTCopula(parameter, self.df)
        else:
            copula = (
                ClaytonCopula(float(parameter))
                if self.family == "clayton"
                else GumbelCopula(float(parameter))
            )
        likelihood = float(copula.log_density(pseudo_observations(x)).sum())
        return CopulaFit(
            copula,
            len(x),
            likelihood,
            distance,
            Diagnostics(
                backend="statsmodels Kendall tau inversion",
                message="Rank fit; log_likelihood is a pseudo-likelihood. Student-t df is supplied, not fitted.",
            ),
        )


def fit_copula(
    samples: ArrayLike,
    *,
    family: str = "gaussian",
    df: float | None = None,
    repair_correlation: bool = False,
) -> CopulaFit:
    """Fit a copula by rank dependence from samples (sample,coordinate).

    family selects the bounded supported catalog. Correlation repair is explicit;
    Student-t degrees of freedom are supplied separately from marginal laws."""
    return KendallCopula(family, df, repair_correlation).fit(samples)
