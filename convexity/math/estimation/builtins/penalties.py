"""Callable penalties and independent log priors in explicit coordinates."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from .. import _validation as v
from .._backends.priors import independent_log_density


@dataclass(frozen=True)
class ElasticNetPenalty:
    """alpha * sum weights * [ratio*|theta-center| + (1-ratio)*(theta-center)²/2]."""

    alpha: float = 1.
    l1_ratio: float = .5
    center: ArrayLike = 0.
    weights: ArrayLike = 1.

    def __post_init__(self):
        v.scalar(self.alpha, "alpha", nonnegative=True)
        ratio = v.scalar(self.l1_ratio, "l1_ratio")
        if not 0 <= ratio <= 1:
            raise ValueError("l1_ratio must lie in [0,1]")
        for name in ("center", "weights"):
            value = v.array(getattr(self, name), name)
            if value.ndim > 1 or (name == "weights" and np.any(value < 0)):
                raise ValueError("penalty center/weights must be scalars or vectors; weights must be nonnegative")
            object.__setattr__(self, name, value)

    def __call__(self, theta):
        theta = v.array(theta, "parameters", ndim=1)
        delta = theta - v.broadcast(self.center, len(theta), "center")
        weights = v.broadcast(self.weights, len(theta), "weights")
        return float(self.alpha * np.sum(weights * (
            self.l1_ratio * np.abs(delta) + (1 - self.l1_ratio) * delta**2 / 2)))


def l1_penalty(alpha=1., *, center: ArrayLike = 0., weights: ArrayLike = 1.):
    return ElasticNetPenalty(alpha, 1., center, weights)


def l2_penalty(alpha=1., *, center: ArrayLike = 0., weights: ArrayLike = 1.):
    return ElasticNetPenalty(alpha, 0., center, weights)


def elastic_net_penalty(alpha=1., l1_ratio=.5, *, center: ArrayLike = 0., weights: ArrayLike = 1.):
    return ElasticNetPenalty(alpha, l1_ratio, center, weights)


@dataclass(frozen=True)
class IndependentPrior:
    family: str = "normal"
    loc: ArrayLike = 0.
    scale: ArrayLike = 1.

    def __post_init__(self):
        v.choice(self.family, ("normal", "laplace"), "prior family")
        for name in ("loc", "scale"):
            value = v.array(getattr(self, name), name)
            if value.ndim > 1 or (name == "scale" and np.any(value <= 0)):
                raise ValueError("prior loc/scale must be scalars or vectors; scale must be positive")
            object.__setattr__(self, name, value)

    def __call__(self, theta):
        theta = v.array(theta, "parameters", ndim=1)
        loc = v.broadcast(self.loc, len(theta), "loc")
        scale = v.broadcast(self.scale, len(theta), "scale")
        return independent_log_density(self.family, theta, loc, scale)


def gaussian_prior(loc: ArrayLike = 0., scale: ArrayLike = 1.):
    """Independent Gaussian log prior; scale is a standard deviation."""
    return IndependentPrior("normal", loc, scale)


def laplace_prior(loc: ArrayLike = 0., scale: ArrayLike = 1.):
    """Independent Laplace log prior (SD=sqrt(2)*scale); nonsmooth at loc.

    MAP at a cusp does not have a Laplace curvature approximation: pass
    uncertainty=False and choose a suitable nonsmooth solver explicitly.
    """
    return IndependentPrior("laplace", loc, scale)
