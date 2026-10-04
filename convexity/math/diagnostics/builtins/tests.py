"""Classical tests and dependence-aware uncertainty with explicit lags."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from numpy.typing import ArrayLike

from ..._typing import FloatArray

if TYPE_CHECKING:
    from .. import TestResult


import numpy as np

from ... import _decision as v
from .._backends import statistical
from ..interfaces import BootstrapResult


def _series(x):
    x = v.array(x, "series", ndim=1)
    if len(x) < 4 or np.ptp(x) == 0:
        raise ValueError("test requires at least four nonconstant observations")
    return x


def residual_test(
    residuals: ArrayLike, *, method: str = "ljung_box", lags: Sequence[int] = (1, 5, 10)
) -> TestResult:
    """Test a scalar residual series for the selected dependence/normality diagnostic.

    Return statistics and p-values; lags are observation offsets, not elapsed time."""
    x = _series(residuals)
    v.choice(method, ("ljung_box", "jarque_bera"), "method")
    lags = tuple(v.integer(k, "lag", 1) for k in lags)
    if method == "ljung_box" and (not lags or max(lags) >= len(x)):
        raise ValueError("lags must be nonempty and less than sample length")
    return statistical.residual_test(x, method, lags)


def stationarity_test(
    series: ArrayLike,
    *,
    method: str = "adf",
    regression: str = "c",
    lags: int | None = None,
) -> TestResult:
    """Run ADF or KPSS on a scalar series with explicit trend/lag specification.

    ADF tests a unit-root null; KPSS tests a stationarity null. Interpret p-values
    against the selected null, not as posterior model probabilities."""
    x = _series(series)
    v.choice(method, ("adf", "kpss"), "method")
    v.choice(regression, ("c", "ct"), "regression")
    if lags is not None:
        v.integer(lags, "lags")
        if lags >= len(x):
            raise ValueError("lags must be less than sample length")
    return statistical.stationarity(x, method, regression, lags)


def cointegration_test(
    dependent: ArrayLike,
    regressors: ArrayLike,
    *,
    trend: str = "c",
    lags: int | None = None,
) -> TestResult:
    """Test a scalar dependent series against aligned regressor columns.

    Return the Engle-Granger statistic/p-value; the null is no cointegration."""
    y, x = _series(dependent), v.array(regressors, "regressors")
    if x.ndim not in (1, 2) or len(x) != len(y):
        raise ValueError("regressors must have aligned observation rows")
    v.choice(trend, ("c", "ct", "ctt", "n"), "trend")
    if lags is not None:
        v.integer(lags, "lags")
        if lags >= len(y):
            raise ValueError("lags must be less than sample length")
    return statistical.cointegration(y, x, trend, lags)


def hac_mean_covariance(observations: ArrayLike, *, lags: int) -> FloatArray:
    """Bartlett HAC covariance of the sample mean; no degrees-of-freedom correction."""
    x = v.array(observations, "observations")
    if x.ndim == 1:
        x = x[:, None]
    lag = v.integer(lags, "lags")
    if x.ndim != 2 or len(x) < 2 or lag >= len(x):
        raise ValueError("observations must be (time,variable), lags < time")
    return v.array(statistical.hac(x, lag), "HAC mean covariance")


def compare_scores(first: ArrayLike, second: ArrayLike, *, lags: int) -> TestResult:
    """Paired asymptotic normal comparison; negative difference favors first."""
    a, b = v.array(first, "first", ndim=1), v.array(second, "second", ndim=1)
    lag = v.integer(lags, "lags")
    if a.shape != b.shape or len(a) < 3 or lag >= len(a):
        raise ValueError("scores must align and lags < observations")
    return statistical.compare(a - b, lag)


def adjust_pvalues(pvalues: ArrayLike, *, method: str = "holm") -> FloatArray:
    """Adjust a vector of valid p-values for the selected multiple-testing procedure.

    Holm controls familywise error; FDR procedures use their stated dependence assumptions."""
    p = v.array(pvalues, "pvalues", ndim=1)
    if np.any(p < 0) or np.any(p > 1):
        raise ValueError("pvalues must lie in [0,1]")
    v.choice(method, ("holm", "bh"), "method")
    return v.array(statistical.adjust(p, method), "adjusted pvalues")


def block_bootstrap(
    statistic: Callable[[FloatArray], ArrayLike],
    observations: ArrayLike,
    *,
    block_length: int,
    draws: int = 999,
    level: float = 0.95,
    rng: np.random.Generator,
) -> BootstrapResult:
    """Circular fixed-length blocks of rows; percentile intervals, paired columns.

    Block length 1 gives IID resampling. Choose blocks to cover dependence and
    horizon overlap; intervals do not correct bias or nonstationarity.
    """
    x = v.array(observations, "observations")
    block, draws = (
        v.integer(block_length, "block length", 1),
        v.integer(draws, "draws", 2),
    )
    level, rng = v.probability(level, "level", open_interval=True), v.generator(rng)
    if x.ndim < 1 or len(x) < 2 or block > len(x):
        raise ValueError("block length must not exceed observation count >= 2")
    estimate = v.array(statistic(x), "estimate")
    samples = []
    for _ in range(draws):
        starts = rng.integers(len(x), size=(len(x) + block - 1) // block)
        indices = ((starts[:, None] + np.arange(block)) % len(x)).ravel()[: len(x)]
        value = v.array(
            statistic(v.array(x[indices], "resample")), "bootstrap statistic"
        )
        if value.shape != estimate.shape:
            raise ValueError("bootstrap statistic changed shape")
        samples.append(value)
    samples = np.stack(samples)
    lo, hi = np.quantile(samples, [(1 - level) / 2, (1 + level) / 2], axis=0)
    return BootstrapResult(estimate, samples, lo, hi, level, block)
