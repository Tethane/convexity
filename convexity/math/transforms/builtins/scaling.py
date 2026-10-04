"""Reusable scaling specifications; fitting always returns independent state."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from .. import FittedScaling


from dataclasses import dataclass

from .. import _validation as v
from .._backends import sklearn


@dataclass(frozen=True)
class Standardize:
    """Fit per-feature mean and standard-deviation scaling; ddof selects the variance denominator."""

    center: bool = True
    scale: bool = True
    ddof: int = 0

    def __post_init__(self) -> None:
        v.boolean(self.center, "center")
        v.boolean(self.scale, "scale")
        v.integer(self.ddof, "ddof")
        v.choice(self.ddof, (0, 1), "ddof")

    def fit(self, samples: ArrayLike) -> FittedScaling:
        x = v.samples(samples, minimum=self.ddof + 1)
        return sklearn.standardize(
            x, center=self.center, scale=self.scale, ddof=self.ddof
        )


@dataclass(frozen=True)
class RobustScale:
    """Fit per-feature centering/scaling using medians and an explicit percentile range."""

    center: bool = True
    scale: bool = True
    quantile_range: tuple[float, float] = (25.0, 75.0)
    unit_variance: bool = False

    def __post_init__(self) -> None:
        for name in ("center", "scale", "unit_variance"):
            v.boolean(getattr(self, name), name)
        q = v.array(self.quantile_range, "quantile_range", ndim=1)
        if q.shape != (2,) or not 0 < q[0] < q[1] < 100:
            raise ValueError("quantile_range must satisfy 0 < lower < upper < 100")
        if self.unit_variance and not self.scale:
            raise ValueError("unit_variance requires scale=True")
        object.__setattr__(self, "quantile_range", tuple(map(float, q)))

    def fit(self, samples: ArrayLike) -> FittedScaling:
        return sklearn.robust_scale(
            v.samples(samples),
            center=self.center,
            scale=self.scale,
            quantile_range=self.quantile_range,
            unit_variance=self.unit_variance,
        )


@dataclass(frozen=True)
class MinMaxScale:
    """Fit an affine per-feature map to a requested output range using training extrema."""

    lower: float = 0.0
    upper: float = 1.0

    def __post_init__(self) -> None:
        lo = v.scalar(self.lower, "lower")
        hi = v.scalar(self.upper, "upper")
        if lo >= hi:
            raise ValueError("lower must be less than upper")

    def fit(self, samples: ArrayLike) -> FittedScaling:
        return sklearn.minmax_scale(
            v.samples(samples), lower=self.lower, upper=self.upper
        )


def standardize(
    samples: ArrayLike, *, center: bool = True, scale: bool = True, ddof: int = 0
) -> FittedScaling:
    """Fit coordinatewise mean/standard-deviation scaling to sample rows.

    ddof determines the variance divisor; center/scale independently select operations."""
    return Standardize(center, scale, ddof).fit(samples)


def robust_scale(
    samples: ArrayLike,
    *,
    center: bool = True,
    scale: bool = True,
    quantile_range: tuple[float, float] = (25.0, 75.0),
    unit_variance: bool = False,
) -> FittedScaling:
    """Fit coordinatewise median/quantile-range scaling to sample rows.

    quantile_range uses percentages; unit_variance applies the normal-reference correction."""
    return RobustScale(center, scale, quantile_range, unit_variance).fit(samples)


def minmax_scale(
    samples: ArrayLike, *, lower: float = 0.0, upper: float = 1.0
) -> FittedScaling:
    """Fit coordinatewise affine scaling of samples (observation,feature).

    Training minima/maxima map to lower/upper; later values may fall outside that range."""
    return MinMaxScale(lower, upper).fit(samples)
