"""Positional differences, lag designs and deterministic periodic features."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from .. import FeatureMatrix


if TYPE_CHECKING:
    from .. import DifferenceResult


from dataclasses import dataclass

from .. import _validation as v
from .._backends import sequence


@dataclass(frozen=True)
class Difference:
    """Apply repeated lag differences; retained boundary data supports reconstruction."""

    lag: int = 1
    order: int = 1

    def __post_init__(self) -> None:
        v.integer(self.lag, "lag", 1)
        v.integer(self.order, "order", 1)

    def __call__(self, values):
        x = v.series(values)
        if len(x) <= self.lag * self.order:
            raise ValueError("series length must exceed lag * order")
        return sequence.difference(x, lag=self.lag, order=self.order)


@dataclass(frozen=True)
class LagFeatures:
    """Nonnegative lags in caller order, with lag-major, then feature-major columns."""

    lags: tuple[int, ...] = (1,)

    def __post_init__(self) -> None:
        lags = tuple(v.integer(x, "lag") for x in self.lags)
        if not lags or len(set(lags)) != len(lags):
            raise ValueError("lags must be nonempty and unique")
        object.__setattr__(self, "lags", lags)

    def __call__(self, values):
        x = v.series(values)
        if max(self.lags) >= len(x):
            raise ValueError("largest lag must be less than series length")
        return sequence.lags(x, self.lags)


@dataclass(frozen=True)
class FourierFeatures:
    """sin/cos pairs on integer t, with period measured in row steps.

    Require order < period/2: this avoids an identically zero sine column at
    the Nyquist frequency. No intercept, rank pruning or phase reset is implicit.
    """

    period: float
    order: int = 1

    def __post_init__(self) -> None:
        period = v.scalar(self.period, "period", positive=True)
        v.integer(self.order, "order", 1)
        if 2 * self.order >= period:
            raise ValueError(
                "require order < period/2 to avoid aliased/Nyquist columns"
            )

    def __call__(self, n: int, *, start: int = 0):
        v.integer(n, "n", 1)
        v.integer(start, "start")
        return sequence.fourier(n, start=start, period=self.period, order=self.order)


@dataclass(frozen=True)
class SeasonalFeatures:
    """Encode a supplied time coordinate using periodic features with explicit periods and harmonics."""

    period: int
    drop_first: bool = False

    def __post_init__(self) -> None:
        v.integer(self.period, "period", 2)
        v.boolean(self.drop_first, "drop_first")

    def __call__(self, n: int, *, start: int = 0):
        v.integer(n, "n", 1)
        v.integer(start, "start")
        return sequence.seasonal(
            n, start=start, period=self.period, drop_first=self.drop_first
        )


def difference(values: ArrayLike, *, lag: int = 1, order: int = 1) -> DifferenceResult:
    """Difference a time-major array using positive row lag and order.

    Return DifferenceResult with differenced values and reconstruction boundary data.
    The first lag*order rows are removed; this does not rescale by elapsed time."""
    return Difference(lag, order)(values)


def lag_features(values: ArrayLike, lags: Sequence[int] = (1,)) -> FeatureMatrix:
    """Return lagged feature columns and their retained source-row indices.

    Input is time-major; lags are nonnegative row offsets. Leading unavailable rows
    are discarded, preventing future observations from entering a lag feature."""
    return LagFeatures(tuple(lags))(values)


def fourier_features(
    n: int, period: float, *, order: int = 1, start: int = 0
) -> FeatureMatrix:
    """Return periodic sine/cosine features for n row indices starting at start.

    Period is measured in observations; order sets harmonic count."""
    return FourierFeatures(period, order)(n, start=start)


def seasonal_features(
    n: int, period: int, *, start: int = 0, drop_first: bool = False
) -> FeatureMatrix:
    """Return periodic indicator features for n observation indices.

    Period counts rows; drop_first removes the reference category for full-rank regression."""
    return SeasonalFeatures(period, drop_first)(n, start=start)
