"""Multivariate heavy-tail fits and univariate extreme-value models."""

from dataclasses import dataclass

import numpy as np

from .. import _validation as v
from .._backends import tails as backend
from ..interfaces.base import checked_convergence
from ..interfaces.tails import PeaksOverThresholdResult


def _optimizer(maxiter, tolerance, check_convergence):
    v.integer(maxiter, "maxiter", 1)
    v.scalar(tolerance, "tolerance", positive=True)
    v.boolean(check_convergence, "check_convergence")


@dataclass(frozen=True)
class StudentT:
    """Elliptical Student-t likelihood fit, optionally estimating df within bounds."""

    df: float = 5.
    estimate_df: bool = False
    df_bounds: tuple[float, float] = (.5, 200.)
    maxiter: int = 1000
    tolerance: float = 1e-10
    check_convergence: bool = True

    def __post_init__(self):
        object.__setattr__(self, "df", v.scalar(self.df, "df", positive=True))
        v.boolean(self.estimate_df, "estimate_df")
        limits = v.array(self.df_bounds, "df_bounds", ndim=1)
        if limits.shape != (2,) or not 0 < limits[0] < limits[1]:
            raise ValueError("df_bounds must contain two ordered positive values")
        if self.estimate_df and not limits[0] <= self.df <= limits[1]:
            raise ValueError("initial df must lie inside df_bounds")
        object.__setattr__(self, "df_bounds", tuple(map(float, limits)))
        _optimizer(self.maxiter, self.tolerance, self.check_convergence)
        object.__setattr__(self, "tolerance", float(self.tolerance))

    def fit(self, samples):
        x = v.samples(samples)
        if len(x) <= x.shape[1]:
            raise ValueError("Student-t fitting requires n > dimension")
        result = backend.fit_student_t(x, df=self.df, estimate_df=self.estimate_df,
                                      df_bounds=self.df_bounds, maxiter=self.maxiter, tolerance=self.tolerance)
        return checked_convergence(result, self.check_convergence)


def multivariate_student_t(df, loc, shape):
    """Reuse the completed family-2 law, including moments and conditioning."""
    from ...distributions import multivariate_student_t as construct
    return construct(df, loc, shape)


def generalized_pareto(xi, *, scale=1., threshold=0.):
    """Law of threshold + GPD excess; xi>0 is heavy-tailed, xi<0 bounded."""
    from ...distributions import Distribution
    return Distribution(backend.TailAdapter("gpd", v.scalar(xi, "xi"), v.scalar(threshold, "threshold"),
                                           v.scalar(scale, "scale", positive=True)))


def generalized_extreme_value(xi, *, loc=0., scale=1.):
    """GEV block-maximum law; conventional xi is the negative of SciPy's c."""
    from ...distributions import Distribution
    return Distribution(backend.TailAdapter("gev", v.scalar(xi, "xi"), v.scalar(loc, "loc"),
                                           v.scalar(scale, "scale", positive=True)))


@dataclass(frozen=True)
class GeneralizedPareto:
    maxiter: int = 3000
    tolerance: float = 1e-8
    check_convergence: bool = True

    def __post_init__(self):
        _optimizer(self.maxiter, self.tolerance, self.check_convergence)
        object.__setattr__(self, "tolerance", float(self.tolerance))

    def fit(self, excesses):
        x = _tail_samples(excesses, minimum=3)
        if np.any(x < 0):
            raise ValueError("GPD observations must be nonnegative excesses")
        result = backend.fit_tail(x, family="gpd", maxiter=self.maxiter, tolerance=self.tolerance)
        return checked_convergence(result, self.check_convergence)


@dataclass(frozen=True)
class GeneralizedExtremeValue:
    maxiter: int = 3000
    tolerance: float = 1e-8
    check_convergence: bool = True

    def __post_init__(self):
        _optimizer(self.maxiter, self.tolerance, self.check_convergence)
        object.__setattr__(self, "tolerance", float(self.tolerance))

    def fit(self, maxima):
        x = _tail_samples(maxima, minimum=4)
        result = backend.fit_tail(x, family="gev", maxiter=self.maxiter, tolerance=self.tolerance)
        return checked_convergence(result, self.check_convergence)


def _tail_samples(values, minimum):
    x = v.array(values, "tail observations", ndim=1)
    if len(x) < minimum or np.ptp(x) == 0:
        raise ValueError(f"tail fitting needs at least {minimum} nonconstant observations")
    return x


def fit_student_t(samples, *, df=5., estimate_df=False, df_bounds=(.5, 200.), maxiter=1000,
                  tolerance=1e-10, check_convergence=True):
    return StudentT(df, estimate_df, df_bounds, maxiter, tolerance, check_convergence).fit(samples)


def fit_gpd(excesses, *, maxiter=3000, tolerance=1e-8, check_convergence=True):
    return GeneralizedPareto(maxiter, tolerance, check_convergence).fit(excesses)


def fit_gev(maxima, *, maxiter=3000, tolerance=1e-8, check_convergence=True):
    return GeneralizedExtremeValue(maxiter, tolerance, check_convergence).fit(maxima)


def peaks_over_threshold(samples, threshold, *, maxiter=3000, tolerance=1e-8, check_convergence=True):
    x = v.array(samples, "samples", ndim=1)
    level = v.scalar(threshold, "threshold")
    tail = fit_gpd(x[x > level] - level, maxiter=maxiter, tolerance=tolerance, check_convergence=check_convergence)
    return PeaksOverThresholdResult(level, tail, len(x))


def block_maxima(samples, block_size, *, drop_incomplete=False):
    """Nonoverlapping positional blocks. Discard a trailing partial block only explicitly."""
    x = v.array(samples, "samples", ndim=1)
    size = v.integer(block_size, "block_size", 1)
    v.boolean(drop_incomplete, "drop_incomplete")
    n, remainder = divmod(len(x), size)
    if not n or (remainder and not drop_incomplete):
        raise ValueError("samples must contain complete blocks; opt in to dropping an incomplete final block")
    return v.array(x[:n * size].reshape(n, size).max(axis=1), "block maxima")
