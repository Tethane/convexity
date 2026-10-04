"""Multivariate heavy-tail fits and univariate extreme-value models."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy.typing import ArrayLike

from ..._typing import FloatArray

if TYPE_CHECKING:
    from .. import PeaksOverThresholdResult


if TYPE_CHECKING:
    from ...distributions import Distribution

if TYPE_CHECKING:
    from .. import TailFit


import numpy as np

if TYPE_CHECKING:
    from .. import StudentTResult


from dataclasses import dataclass

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

    df: float = 5.0
    estimate_df: bool = False
    df_bounds: tuple[float, float] = (0.5, 200.0)
    maxiter: int = 1000
    tolerance: float = 1e-10
    check_convergence: bool = True

    def __post_init__(self) -> None:
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

    def fit(self, samples: ArrayLike) -> StudentTResult:
        x = v.samples(samples)
        if len(x) <= x.shape[1]:
            raise ValueError("Student-t fitting requires n > dimension")
        result = backend.fit_student_t(
            x,
            df=self.df,
            estimate_df=self.estimate_df,
            df_bounds=self.df_bounds,
            maxiter=self.maxiter,
            tolerance=self.tolerance,
        )
        return checked_convergence(result, self.check_convergence)


def multivariate_student_t(df: float, loc: ArrayLike, shape: ArrayLike) -> Distribution:
    """Reuse the completed family-2 law, including moments and conditioning."""
    from ...distributions import multivariate_student_t as construct

    return construct(df, loc, shape)


def generalized_pareto(
    xi: float, *, scale: float = 1.0, threshold: float = 0.0
) -> Distribution:
    """Law of threshold + GPD excess; xi>0 is heavy-tailed, xi<0 bounded."""
    from ...distributions import Distribution

    return Distribution(
        backend.TailAdapter(
            "gpd",
            v.scalar(xi, "xi"),
            v.scalar(threshold, "threshold"),
            v.scalar(scale, "scale", positive=True),
        )
    )


def generalized_extreme_value(
    xi: float, *, loc: float = 0.0, scale: float = 1.0
) -> Distribution:
    """GEV block-maximum law; conventional xi is the negative of SciPy's c."""
    from ...distributions import Distribution

    return Distribution(
        backend.TailAdapter(
            "gev",
            v.scalar(xi, "xi"),
            v.scalar(loc, "loc"),
            v.scalar(scale, "scale", positive=True),
        )
    )


@dataclass(frozen=True)
class GeneralizedPareto:
    """Fit a generalized Pareto law to nonnegative threshold excesses, with checked numerical convergence."""

    maxiter: int = 3000
    tolerance: float = 1e-8
    check_convergence: bool = True

    def __post_init__(self) -> None:
        _optimizer(self.maxiter, self.tolerance, self.check_convergence)
        object.__setattr__(self, "tolerance", float(self.tolerance))

    def fit(self, excesses: ArrayLike) -> TailFit:
        x = _tail_samples(excesses, minimum=3)
        if np.any(x < 0):
            raise ValueError("GPD observations must be nonnegative excesses")
        result = backend.fit_tail(
            x, family="gpd", maxiter=self.maxiter, tolerance=self.tolerance
        )
        return checked_convergence(result, self.check_convergence)


@dataclass(frozen=True)
class GeneralizedExtremeValue:
    """Fit a generalized extreme-value law to block maxima, with checked numerical convergence."""

    maxiter: int = 3000
    tolerance: float = 1e-8
    check_convergence: bool = True

    def __post_init__(self) -> None:
        _optimizer(self.maxiter, self.tolerance, self.check_convergence)
        object.__setattr__(self, "tolerance", float(self.tolerance))

    def fit(self, maxima: ArrayLike) -> TailFit:
        x = _tail_samples(maxima, minimum=4)
        result = backend.fit_tail(
            x, family="gev", maxiter=self.maxiter, tolerance=self.tolerance
        )
        return checked_convergence(result, self.check_convergence)


def _tail_samples(values, minimum):
    x = v.array(values, "tail observations", ndim=1)
    if len(x) < minimum or np.ptp(x) == 0:
        raise ValueError(
            f"tail fitting needs at least {minimum} nonconstant observations"
        )
    return x


def fit_student_t(
    samples: ArrayLike,
    *,
    df: float = 5.0,
    estimate_df: bool = False,
    df_bounds: tuple[float, float] = (0.5, 200.0),
    maxiter: int = 1000,
    tolerance: float = 1e-10,
    check_convergence: bool = True,
) -> StudentTResult:
    """Fit a multivariate Student-t model to sample rows.

    Scale is the t dispersion matrix, not covariance; covariance exists only for
    df>2. Optional df estimation respects df_bounds and reports convergence."""
    return StudentT(
        df, estimate_df, df_bounds, maxiter, tolerance, check_convergence
    ).fit(samples)


def fit_gpd(
    excesses: ArrayLike,
    *,
    maxiter: int = 3000,
    tolerance: float = 1e-8,
    check_convergence: bool = True,
) -> TailFit:
    """Fit a generalized Pareto law to nonnegative threshold excesses.

    Inputs are excess amounts, not raw observations. Return a tail fit with explicit
    shape/scale semantics and convergence diagnostics."""
    return GeneralizedPareto(maxiter, tolerance, check_convergence).fit(excesses)


def fit_gev(
    maxima: ArrayLike,
    *,
    maxiter: int = 3000,
    tolerance: float = 1e-8,
    check_convergence: bool = True,
) -> TailFit:
    """Fit a generalized extreme-value law to scalar block maxima.

    Blocks and their common horizon are supplied by the caller. Return a tail-fit
    result with convergence diagnostics; this does not establish block independence."""
    return GeneralizedExtremeValue(maxiter, tolerance, check_convergence).fit(maxima)


def peaks_over_threshold(
    samples: ArrayLike,
    threshold: float,
    *,
    maxiter: int = 3000,
    tolerance: float = 1e-8,
    check_convergence: bool = True,
) -> PeaksOverThresholdResult:
    """Fit the exceedance law above threshold from raw scalar samples.

    Return threshold, exceedance frequency and GPD tail fit. Tail probabilities
    combine observed exceedance mass with the conditional excess law."""
    x = v.array(samples, "samples", ndim=1)
    level = v.scalar(threshold, "threshold")
    tail = fit_gpd(
        x[x > level] - level,
        maxiter=maxiter,
        tolerance=tolerance,
        check_convergence=check_convergence,
    )
    return PeaksOverThresholdResult(level, tail, len(x))


def block_maxima(
    samples: ArrayLike, block_size: int, *, drop_incomplete: bool = False
) -> FloatArray:
    """Nonoverlapping positional blocks. Discard a trailing partial block only explicitly."""
    x = v.array(samples, "samples", ndim=1)
    size = v.integer(block_size, "block_size", 1)
    v.boolean(drop_incomplete, "drop_incomplete")
    n, remainder = divmod(len(x), size)
    if not n or (remainder and not drop_incomplete):
        raise ValueError(
            "samples must contain complete blocks; opt in to dropping an incomplete final block"
        )
    return v.array(x[: n * size].reshape(n, size).max(axis=1), "block maxima")
