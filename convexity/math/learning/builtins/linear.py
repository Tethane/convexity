"""Linear mean, probability and quantile learning specifications."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict, Unpack

from numpy.typing import ArrayLike

from .. import _validation as v
from .._backends import sklearn
from ..interfaces.base import FittedModel, checked_convergence


class LinearRegressionOptions(TypedDict, total=False):
    """Optional keyword settings for linear_regression."""

    penalty: str
    alpha: float
    l1_ratio: float
    fit_intercept: bool
    tolerance: float
    maxiter: int
    check_convergence: bool


class RidgeOptions(TypedDict, total=False):
    """Optional keyword settings for ridge."""

    l1_ratio: float
    fit_intercept: bool
    tolerance: float
    maxiter: int
    check_convergence: bool


class LassoOptions(TypedDict, total=False):
    """Optional keyword settings for lasso."""

    l1_ratio: float
    fit_intercept: bool
    tolerance: float
    maxiter: int
    check_convergence: bool


class ElasticNetOptions(TypedDict, total=False):
    """Optional keyword settings for elastic_net."""

    fit_intercept: bool
    tolerance: float
    maxiter: int
    check_convergence: bool


class LogisticRegressionOptions(TypedDict, total=False):
    """Optional keyword settings for logistic_regression."""

    C: float
    fit_intercept: bool
    tolerance: float
    maxiter: int
    check_convergence: bool


class QuantileRegressionOptions(TypedDict, total=False):
    """Optional keyword settings for quantile_regression."""

    alpha: float
    fit_intercept: bool
    check_convergence: bool


@dataclass(frozen=True)
class LinearRegression:
    """Weighted mean squared loss / 2 plus an unscaled coefficient penalty.

    The intercept is optional and unpenalized. No feature scaling is implicit.
    Each target in a matrix response is fit with its own coefficient penalty.
    """

    penalty: str = "none"
    alpha: float = 1.0
    l1_ratio: float = 0.5
    fit_intercept: bool = True
    tolerance: float = 1e-8
    maxiter: int = 10000
    check_convergence: bool = True

    def __post_init__(self) -> None:
        v.choice(self.penalty, ("none", "ridge", "lasso", "elastic_net"), "penalty")
        object.__setattr__(
            self, "alpha", v.scalar(self.alpha, "alpha", minimum=0, strict=True)
        )
        ratio = v.scalar(self.l1_ratio, "l1_ratio")
        if not 0 <= ratio <= 1:
            raise ValueError("l1_ratio must lie in [0,1]")
        object.__setattr__(self, "l1_ratio", ratio)
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
        sample_weight: ArrayLike | None = None,
    ) -> FittedModel:
        x, y, w = v.training(features, response, sample_weight)
        return checked_convergence(
            sklearn.linear(self, x, y, w), self.check_convergence
        )


@dataclass(frozen=True)
class Ridge(LinearRegression):
    """Linear regression with an L2 coefficient penalty; no feature scaling is implicit."""

    penalty: str = field(default="ridge", init=False)


@dataclass(frozen=True)
class Lasso(LinearRegression):
    """Linear regression with an L1 coefficient penalty; no feature scaling is implicit."""

    penalty: str = field(default="lasso", init=False)


@dataclass(frozen=True)
class ElasticNet(LinearRegression):
    """Linear regression with combined L1/L2 coefficient penalty; the intercept is unpenalized."""

    penalty: str = field(default="elastic_net", init=False)


@dataclass(frozen=True)
class LogisticRegression:
    """Binary/multinomial L2 logistic learning; C uses scikit-learn's convention."""

    C: float = 1.0
    fit_intercept: bool = True
    tolerance: float = 1e-8
    maxiter: int = 1000
    check_convergence: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "C", v.scalar(self.C, "C", minimum=0, strict=True))
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
        sample_weight: ArrayLike | None = None,
    ) -> FittedModel:
        x, y, w = v.training(features, response, sample_weight, classification=True)
        return checked_convergence(
            sklearn.logistic(self, x, y, w), self.check_convergence
        )


@dataclass(frozen=True)
class QuantileRegression:
    """Independent linear quantiles: weighted pinball loss + alpha*||beta||_1."""

    quantiles: tuple[float, ...] = (0.5,)
    alpha: float = 0.0
    fit_intercept: bool = True
    check_convergence: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "quantiles", tuple(v.levels(self.quantiles)))
        object.__setattr__(self, "alpha", v.scalar(self.alpha, "alpha", minimum=0))
        v.boolean(self.fit_intercept, "fit_intercept")
        v.boolean(self.check_convergence, "check_convergence")

    def fit(
        self,
        features: ArrayLike,
        response: ArrayLike,
        *,
        sample_weight: ArrayLike | None = None,
    ) -> FittedModel:
        x, y, w = v.training(features, response, sample_weight, multi=False)
        return checked_convergence(
            sklearn.quantile(self, x, y, w), self.check_convergence
        )


def linear_regression(
    features: ArrayLike,
    response: ArrayLike,
    *,
    sample_weight: ArrayLike | None = None,
    **kwargs: Unpack[LinearRegressionOptions],
) -> FittedModel:
    """Fit LinearRegression to features (observation,feature) and aligned response.

    Weighted mean squared loss / 2 plus an unscaled coefficient penalty.
    Return the checked fitted predictor; no feature scaling is implicit.
    Optional settings: penalty, alpha, l1_ratio, fit_intercept, tolerance, maxiter, check_convergence."""
    return LinearRegression(**kwargs).fit(
        features, response, sample_weight=sample_weight
    )


def ridge(
    features: ArrayLike,
    response: ArrayLike,
    *,
    alpha: float = 1.0,
    sample_weight: ArrayLike | None = None,
    **kwargs: Unpack[RidgeOptions],
) -> FittedModel:
    """Fit Ridge to features (observation,feature) and aligned response.

    Ridge(alpha: 'float' = 1.0, l1_ratio: 'float' = 0.5, fit_intercept: 'bool' = True, tolerance: 'float' = 1e-08, maxiter: 'int' = 10000, check_convergence: 'bool' = True)
    Return the checked fitted predictor; no feature scaling is implicit.
    Optional settings: l1_ratio, fit_intercept, tolerance, maxiter, check_convergence."""
    return Ridge(alpha=alpha, **kwargs).fit(
        features, response, sample_weight=sample_weight
    )


def lasso(
    features: ArrayLike,
    response: ArrayLike,
    *,
    alpha: float = 1.0,
    sample_weight: ArrayLike | None = None,
    **kwargs: Unpack[LassoOptions],
) -> FittedModel:
    """Fit Lasso to features (observation,feature) and aligned response.

    Lasso(alpha: 'float' = 1.0, l1_ratio: 'float' = 0.5, fit_intercept: 'bool' = True, tolerance: 'float' = 1e-08, maxiter: 'int' = 10000, check_convergence: 'bool' = True)
    Return the checked fitted predictor; no feature scaling is implicit.
    Optional settings: l1_ratio, fit_intercept, tolerance, maxiter, check_convergence."""
    return Lasso(alpha=alpha, **kwargs).fit(
        features, response, sample_weight=sample_weight
    )


def elastic_net(
    features: ArrayLike,
    response: ArrayLike,
    *,
    alpha: float = 1.0,
    l1_ratio: float = 0.5,
    sample_weight: ArrayLike | None = None,
    **kwargs: Unpack[ElasticNetOptions],
) -> FittedModel:
    """Fit ElasticNet to features (observation,feature) and aligned response.

    ElasticNet(alpha: 'float' = 1.0, l1_ratio: 'float' = 0.5, fit_intercept: 'bool' = True, tolerance: 'float' = 1e-08, maxiter: 'int' = 10000, check_convergence: 'bool' = True)
    Return the checked fitted predictor; no feature scaling is implicit.
    Optional settings: fit_intercept, tolerance, maxiter, check_convergence."""
    return ElasticNet(alpha=alpha, l1_ratio=l1_ratio, **kwargs).fit(
        features, response, sample_weight=sample_weight
    )


def logistic_regression(
    features: ArrayLike,
    response: ArrayLike,
    *,
    sample_weight: ArrayLike | None = None,
    **kwargs: Unpack[LogisticRegressionOptions],
) -> FittedModel:
    """Fit LogisticRegression to features (observation,feature) and aligned response.

    Binary/multinomial L2 logistic learning; C uses scikit-learn's convention.
    Return the checked fitted predictor; no feature scaling is implicit.
    Optional settings: C, fit_intercept, tolerance, maxiter, check_convergence."""
    return LogisticRegression(**kwargs).fit(
        features, response, sample_weight=sample_weight
    )


def quantile_regression(
    features: ArrayLike,
    response: ArrayLike,
    *,
    quantiles: tuple[float, ...] = (0.5,),
    sample_weight: ArrayLike | None = None,
    **kwargs: Unpack[QuantileRegressionOptions],
) -> FittedModel:
    """Fit QuantileRegression to features (observation,feature) and aligned response.

    Independent linear quantiles: weighted pinball loss + alpha*||beta||_1.
    Return the checked fitted predictor; no feature scaling is implicit.
    Optional settings: alpha, fit_intercept, check_convergence."""
    return QuantileRegression(quantiles=quantiles, **kwargs).fit(
        features, response, sample_weight=sample_weight
    )
