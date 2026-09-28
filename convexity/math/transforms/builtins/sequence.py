"""Positional differences, lag designs and deterministic periodic features."""

from dataclasses import dataclass

from .. import _validation as v
from .._backends import sequence


@dataclass(frozen=True)
class Difference:
    lag: int = 1
    order: int = 1

    def __post_init__(self):
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

    def __post_init__(self):
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

    def __post_init__(self):
        period = v.scalar(self.period, "period", positive=True)
        v.integer(self.order, "order", 1)
        if 2 * self.order >= period:
            raise ValueError("require order < period/2 to avoid aliased/Nyquist columns")

    def __call__(self, n, *, start=0):
        v.integer(n, "n", 1)
        v.integer(start, "start")
        return sequence.fourier(n, start=start, period=self.period, order=self.order)


@dataclass(frozen=True)
class SeasonalFeatures:
    period: int
    drop_first: bool = False

    def __post_init__(self):
        v.integer(self.period, "period", 2)
        v.boolean(self.drop_first, "drop_first")

    def __call__(self, n, *, start=0):
        v.integer(n, "n", 1)
        v.integer(start, "start")
        return sequence.seasonal(n, start=start, period=self.period, drop_first=self.drop_first)


def difference(values, *, lag=1, order=1):
    return Difference(lag, order)(values)


def lag_features(values, lags=(1,)):
    return LagFeatures(lags)(values)


def fourier_features(n, period, *, order=1, start=0):
    return FourierFeatures(period, order)(n, start=start)


def seasonal_features(n, period, *, start=0, drop_first=False):
    return SeasonalFeatures(period, drop_first)(n, start=start)
