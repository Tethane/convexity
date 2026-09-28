"""Linear mean, probability and quantile learning specifications."""

from dataclasses import dataclass, field

from .. import _validation as v
from .._backends import sklearn
from ..interfaces.base import checked_convergence


@dataclass(frozen=True)
class LinearRegression:
    """Weighted mean squared loss / 2 plus an unscaled coefficient penalty.

    The intercept is optional and unpenalized. No feature scaling is implicit.
    Each target in a matrix response is fit with its own coefficient penalty.
    """

    penalty: str = "none"
    alpha: float = 1.
    l1_ratio: float = .5
    fit_intercept: bool = True
    tolerance: float = 1e-8
    maxiter: int = 10000
    check_convergence: bool = True

    def __post_init__(self):
        v.choice(self.penalty, ("none", "ridge", "lasso", "elastic_net"), "penalty")
        object.__setattr__(self, "alpha", v.scalar(self.alpha, "alpha", minimum=0, strict=True))
        ratio = v.scalar(self.l1_ratio, "l1_ratio")
        if not 0 <= ratio <= 1:
            raise ValueError("l1_ratio must lie in [0,1]")
        object.__setattr__(self, "l1_ratio", ratio)
        v.boolean(self.fit_intercept, "fit_intercept")
        object.__setattr__(self, "tolerance", v.scalar(self.tolerance, "tolerance", minimum=0, strict=True))
        v.integer(self.maxiter, "maxiter", 1)
        v.boolean(self.check_convergence, "check_convergence")

    def fit(self, features, response, *, sample_weight=None):
        x, y, w = v.training(features, response, sample_weight)
        return checked_convergence(sklearn.linear(self, x, y, w), self.check_convergence)


@dataclass(frozen=True)
class Ridge(LinearRegression):
    penalty: str = field(default="ridge", init=False)


@dataclass(frozen=True)
class Lasso(LinearRegression):
    penalty: str = field(default="lasso", init=False)


@dataclass(frozen=True)
class ElasticNet(LinearRegression):
    penalty: str = field(default="elastic_net", init=False)


@dataclass(frozen=True)
class LogisticRegression:
    """Binary/multinomial L2 logistic learning; C uses scikit-learn's convention."""

    C: float = 1.
    fit_intercept: bool = True
    tolerance: float = 1e-8
    maxiter: int = 1000
    check_convergence: bool = True

    def __post_init__(self):
        object.__setattr__(self, "C", v.scalar(self.C, "C", minimum=0, strict=True))
        v.boolean(self.fit_intercept, "fit_intercept")
        object.__setattr__(self, "tolerance", v.scalar(self.tolerance, "tolerance", minimum=0, strict=True))
        v.integer(self.maxiter, "maxiter", 1)
        v.boolean(self.check_convergence, "check_convergence")

    def fit(self, features, response, *, sample_weight=None):
        x, y, w = v.training(features, response, sample_weight, classification=True)
        return checked_convergence(sklearn.logistic(self, x, y, w), self.check_convergence)


@dataclass(frozen=True)
class QuantileRegression:
    """Independent linear quantiles: weighted pinball loss + alpha*||beta||_1."""

    quantiles: tuple[float, ...] = (.5,)
    alpha: float = 0.
    fit_intercept: bool = True
    check_convergence: bool = True

    def __post_init__(self):
        object.__setattr__(self, "quantiles", tuple(v.levels(self.quantiles)))
        object.__setattr__(self, "alpha", v.scalar(self.alpha, "alpha", minimum=0))
        v.boolean(self.fit_intercept, "fit_intercept")
        v.boolean(self.check_convergence, "check_convergence")

    def fit(self, features, response, *, sample_weight=None):
        x, y, w = v.training(features, response, sample_weight, multi=False)
        return checked_convergence(sklearn.quantile(self, x, y, w), self.check_convergence)


def linear_regression(features, response, **kwargs):
    sample_weight = kwargs.pop("sample_weight", None)
    return LinearRegression(**kwargs).fit(features, response, sample_weight=sample_weight)


def ridge(features, response, *, alpha=1., sample_weight=None, **kwargs):
    return Ridge(alpha=alpha, **kwargs).fit(features, response, sample_weight=sample_weight)


def lasso(features, response, *, alpha=1., sample_weight=None, **kwargs):
    return Lasso(alpha=alpha, **kwargs).fit(features, response, sample_weight=sample_weight)


def elastic_net(features, response, *, alpha=1., l1_ratio=.5, sample_weight=None, **kwargs):
    return ElasticNet(alpha=alpha, l1_ratio=l1_ratio, **kwargs).fit(features, response, sample_weight=sample_weight)


def logistic_regression(features, response, *, sample_weight=None, **kwargs):
    return LogisticRegression(**kwargs).fit(features, response, sample_weight=sample_weight)


def quantile_regression(features, response, *, quantiles=(.5,), sample_weight=None, **kwargs):
    return QuantileRegression(quantiles=quantiles, **kwargs).fit(features, response, sample_weight=sample_weight)
