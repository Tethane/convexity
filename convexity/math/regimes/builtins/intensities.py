"""Deterministic Poisson intensities and exponential self-excitation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ..._typing import (
    FloatArray,
)
from .. import _validation as v
from ..interfaces.base import Intensity


@dataclass(frozen=True)
class ConstantIntensity:
    """Homogeneous event intensity per unit of the supplied time coordinate."""

    rate: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "rate", v.scalar(self.rate, "rate", minimum=0))

    def intensity(self, time: float):
        v.scalar(time, "time")
        return self.rate

    def integrated(self, start: float, end: float) -> float:
        start, end = v.interval(start, end)
        return v.scalar(self.rate * (end - start), "integrated intensity", minimum=0)


@dataclass(frozen=True)
class PiecewiseConstantIntensity:
    """rates[i] applies between breaks[i-1] and breaks[i], extending to ±inf.

    At a breakpoint the right-hand rate is used. Rates have len(breaks)+1 rows.
    """

    breaks: FloatArray
    rates: FloatArray

    def __post_init__(self) -> None:
        breaks = v.times(self.breaks)
        rates = v.array(self.rates, "rates", ndim=1)
        if rates.shape != (len(breaks) + 1,) or np.any(rates < 0):
            raise ValueError("rates must be nonnegative with len(breaks)+1 entries")
        object.__setattr__(self, "breaks", breaks)
        object.__setattr__(self, "rates", rates)

    def intensity(self, time: float) -> float:
        return float(
            self.rates[
                np.searchsorted(self.breaks, v.scalar(time, "time"), side="right")
            ]
        )

    def integrated(self, start: float, end: float) -> float:
        start, end = v.interval(start, end)
        edges = np.r_[
            start, self.breaks[(self.breaks > start) & (self.breaks < end)], end
        ]
        rates = self.rates[np.searchsorted(self.breaks, edges[:-1], side="right")]
        return v.scalar(rates @ np.diff(edges), "integrated intensity", minimum=0)

    def __init__(
        self,
        breaks: ArrayLike,
        rates: ArrayLike,
    ) -> None:
        object.__setattr__(self, "breaks", breaks)
        object.__setattr__(self, "rates", rates)
        self.__post_init__()


@dataclass(frozen=True)
class PoissonProcess:
    """Independent increments with a deterministic Intensity implementation."""

    rate: Intensity

    def __post_init__(self) -> None:
        if not callable(getattr(self.rate, "intensity", None)) or not callable(
            getattr(self.rate, "integrated", None)
        ):
            raise TypeError(
                "rate must implement intensity(time) and integrated(start,end)"
            )

    def integrated(self, start: float, end: float) -> float:
        start, end = v.interval(start, end)
        return v.scalar(
            self.rate.integrated(start, end), "integrated intensity", minimum=0
        )

    def count_distribution(self, start: float, end: float):
        from ...distributions import poisson

        return poisson(self.integrated(start, end))

    def event_probability(self, start: float, end: float) -> float:
        return float(-np.expm1(-self.integrated(start, end)))

    def log_likelihood(self, times: ArrayLike, start: float, end: float) -> float:
        start, end = v.interval(start, end)
        times = v.times(times)
        if np.any(times <= start) or np.any(times > end):
            raise ValueError("times must lie in (start,end]")
        rates = np.array(
            [v.scalar(self.rate.intensity(t), "intensity", minimum=0) for t in times]
        )
        with np.errstate(divide="ignore"):
            return float(np.log(rates).sum() - self.integrated(start, end))

    def sample(self, start: float, end: float, *, rng: np.random.Generator):
        """Poisson count then inverse integrated-intensity ordered times."""
        from scipy.optimize import brentq

        start, end = v.interval(start, end)
        v.generator(rng)
        mass = self.integrated(start, end)
        count = int(rng.poisson(mass))
        if count == 0:
            return v.times([])
        # The built-ins permit exact segment sampling, including zero-rate gaps.
        if isinstance(self.rate, (ConstantIntensity, PiecewiseConstantIntensity)):
            breaks = (
                self.rate.breaks
                if isinstance(self.rate, PiecewiseConstantIntensity)
                else np.array([])
            )
            edges = np.r_[start, breaks[(breaks > start) & (breaks < end)], end]
            rates = np.array([self.rate.intensity(t) for t in edges[:-1]])
            positive = rates > 0
            left, right, rates = (
                edges[:-1][positive],
                edges[1:][positive],
                rates[positive],
            )
            masses = (right - left) * rates
            bins = rng.choice(len(left), size=count, p=masses / masses.sum())
            times = left[bins] + (right[bins] - left[bins]) * (1 - rng.random(count))
        else:
            targets = mass * (1 - rng.random(count))
            times = np.array(
                [
                    brentq(
                        lambda t: self.integrated(start, t) - u,
                        start,
                        end,
                        xtol=np.nextafter(0.0, 1.0),
                        rtol=4 * np.finfo(float).eps,
                    )
                    for u in targets
                ]
            )
        times = np.sort(times)
        if np.any(times <= start) or np.any(np.diff(times) <= 0):
            raise ValueError("time resolution cannot represent distinct sampled events")
        return v.times(times)

    @classmethod
    def fit(cls, times: ArrayLike, start: float, end: float):
        """Homogeneous Poisson MLE, including windows with no events."""
        start, end = v.interval(start, end)
        times = v.times(times)
        if end == start or np.any(times <= start) or np.any(times > end):
            raise ValueError(
                "fit needs positive exposure and event times in (start,end]"
            )
        return cls(ConstantIntensity(len(times) / (end - start)))


@dataclass(frozen=True)
class ExponentialHawkes:
    """λ(t|H) = baseline + excitation * sum(exp(-decay*(t-ti))), ti < t.

    excitation/decay < 1 is required. Supplied history is conditioned on;
    omitted history means an empty prehistory, not stationary initialization.
    """

    baseline: float
    excitation: float
    decay: float

    def __post_init__(self) -> None:
        for name in ("baseline", "excitation", "decay"):
            value = v.scalar(
                getattr(self, name), name, minimum=0, strict=name == "decay"
            )
            object.__setattr__(self, name, value)
        if self.excitation >= self.decay:
            raise ValueError("stationary branching ratio excitation/decay must be < 1")

    @property
    def stationary_rate(self):
        return self.baseline / (1 - self.excitation / self.decay)

    def intensity(self, time: float, history: ArrayLike = ()) -> float:
        time, history = v.scalar(time, "time"), v.times(history)
        return float(
            self.baseline
            + self.excitation
            * np.exp(-self.decay * (time - history[history < time])).sum()
        )

    def integrated(self, start: float, end: float, history: ArrayLike = ()) -> float:
        """Compensator along a supplied complete history, not an expected count."""
        start, end = v.interval(start, end)
        history = v.times(history)
        events = history[history < end]
        left = np.maximum(start, events)
        terms = np.exp(-self.decay * (left - events)) * (
            -np.expm1(-self.decay * (end - left))
        )
        return float(
            self.baseline * (end - start) + self.excitation / self.decay * terms.sum()
        )

    def log_likelihood(
        self, times: ArrayLike, start: float, end: float, *, history: ArrayLike = ()
    ):
        start, end = v.interval(start, end)
        times, history = v.times(times), v.times(history)
        if np.any(history > start) or np.any(times <= start) or np.any(times > end):
            raise ValueError(
                "history must be <= start and observed times in (start,end]"
            )
        # Recursive excitation gives O(n + history) likelihood evaluation.
        excitation = self.excitation * np.exp(-self.decay * (start - history)).sum()
        previous, total = start, 0.0
        for t in times:
            excitation *= np.exp(-self.decay * (t - previous))
            rate = self.baseline + excitation
            if rate == 0:
                return -np.inf
            total += np.log(rate)
            excitation += self.excitation
            previous = t
        return float(total - self.integrated(start, end, np.r_[history, times]))

    def sample(
        self,
        start: float,
        end: float,
        *,
        rng: np.random.Generator,
        history: ArrayLike = (),
        max_events: int = 1_000_000,
    ):
        """Exact thinning with a decaying intensity upper bound."""
        start, end = v.interval(start, end)
        v.generator(rng)
        history = v.times(history)
        limit = v.integer(max_events, "max_events", 1)
        if np.any(history > start):
            raise ValueError("history must end at or before start")
        excitation = self.excitation * np.exp(-self.decay * (start - history)).sum()
        t, out = start, []
        while t < end:
            bound = self.baseline + excitation
            if bound == 0:
                break
            wait = rng.exponential(1 / bound)
            candidate = t + wait
            if candidate > end:
                break
            if candidate <= t:
                raise ValueError("time resolution cannot represent the next event")
            excitation *= np.exp(-self.decay * wait)
            t = candidate
            if rng.random() * bound < self.baseline + excitation:
                if len(out) == limit:
                    raise RuntimeError(
                        "max_events exceeded; no truncated realization returned"
                    )
                out.append(t)
                excitation += self.excitation
        return v.times(out)
