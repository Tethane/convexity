"""Reusable scaling specifications; fitting always returns independent state."""

from dataclasses import dataclass

from .. import _validation as v
from .._backends import sklearn


@dataclass(frozen=True)
class Standardize:
    center: bool = True
    scale: bool = True
    ddof: int = 0

    def __post_init__(self):
        v.boolean(self.center, "center")
        v.boolean(self.scale, "scale")
        v.integer(self.ddof, "ddof")
        v.choice(self.ddof, (0, 1), "ddof")

    def fit(self, samples):
        x = v.samples(samples, minimum=self.ddof + 1)
        return sklearn.standardize(x, center=self.center, scale=self.scale, ddof=self.ddof)


@dataclass(frozen=True)
class RobustScale:
    center: bool = True
    scale: bool = True
    quantile_range: tuple[float, float] = (25., 75.)
    unit_variance: bool = False

    def __post_init__(self):
        for name in ("center", "scale", "unit_variance"):
            v.boolean(getattr(self, name), name)
        q = v.array(self.quantile_range, "quantile_range", ndim=1)
        if q.shape != (2,) or not 0 < q[0] < q[1] < 100:
            raise ValueError("quantile_range must satisfy 0 < lower < upper < 100")
        if self.unit_variance and not self.scale:
            raise ValueError("unit_variance requires scale=True")
        object.__setattr__(self, "quantile_range", tuple(map(float, q)))

    def fit(self, samples):
        return sklearn.robust_scale(v.samples(samples), center=self.center, scale=self.scale,
                                     quantile_range=self.quantile_range, unit_variance=self.unit_variance)


@dataclass(frozen=True)
class MinMaxScale:
    lower: float = 0.
    upper: float = 1.

    def __post_init__(self):
        lo = v.scalar(self.lower, "lower")
        hi = v.scalar(self.upper, "upper")
        if lo >= hi:
            raise ValueError("lower must be less than upper")

    def fit(self, samples):
        return sklearn.minmax_scale(v.samples(samples), lower=self.lower, upper=self.upper)


def standardize(samples, *, center=True, scale=True, ddof=0):
    return Standardize(center, scale, ddof).fit(samples)


def robust_scale(samples, *, center=True, scale=True, quantile_range=(25., 75.), unit_variance=False):
    return RobustScale(center, scale, quantile_range, unit_variance).fit(samples)


def minmax_scale(samples, *, lower=0., upper=1.):
    return MinMaxScale(lower, upper).fit(samples)
