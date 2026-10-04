"""Scheduled surprises, conditional impacts, and marked jump processes."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ..._typing import FloatArray

if TYPE_CHECKING:
    from ...distributions import Distribution
from .. import _validation as v
from ..interfaces import MarkedEvents, sample_events
from ..interfaces.base import EventProcess
from .intensities import PoissonProcess


@dataclass(frozen=True)
class ScheduledEvent:
    """Known event time, surprise law, and optional conditional impact law.

    impact(surprise) returns a Distribution; without it the mark is the surprise.
    Surprise values are supplied deviations from a caller's reference expectation.
    """

    time: float
    surprise: Distribution
    impact: Callable[[FloatArray], Distribution] | None = None
    name: str = "event"

    def __post_init__(self) -> None:
        object.__setattr__(self, "time", v.scalar(self.time, "time"))
        v.law(self.surprise)
        if self.impact is not None and not callable(self.impact):
            raise TypeError("impact must be callable")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a nonempty string")

    def sample(self, *, draws: int = 1, rng: np.random.Generator):
        """Return (surprises, impacts), each with a leading draw axis."""
        draws = v.integer(draws, "draws", 1)
        surprises = self.surprise.sample(draws, rng=rng)
        if self.impact is None:
            return surprises, surprises
        marks, shape = [], None
        for surprise in surprises:
            law = v.law(self.impact(surprise), shape)
            shape = law.event_shape
            marks.append(law.sample(rng=rng))
        return surprises, v.array(marks, "impacts")

    def impact_distribution(self, *, draws: int = 1000, rng: np.random.Generator):
        """Exact surprise law for identity impact; otherwise an empirical mixture."""
        from ...distributions import empirical

        v.generator(rng)
        v.integer(draws, "draws", 1)
        return (
            self.surprise
            if self.impact is None
            else empirical(self.sample(draws=draws, rng=rng)[1])
        )


@dataclass(frozen=True)
class EventSchedule:
    """Independent scheduled events, with additive scalar/vector marks.

    Simultaneous events are allowed. Dependence within a surprise/impact vector
    is preserved; dependence across separate events must be modeled explicitly.
    """

    events: Sequence[ScheduledEvent]
    event_shape: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        events = tuple(self.events)
        if any(not isinstance(x, ScheduledEvent) for x in events):
            raise TypeError("events must be ScheduledEvent objects")
        if not isinstance(self.event_shape, tuple) or len(self.event_shape) > 1:
            raise ValueError("event_shape must be () or (dimension,)")
        if self.event_shape:
            v.integer(self.event_shape[0], "dimension", 1)
        object.__setattr__(self, "events", tuple(sorted(events, key=lambda x: x.time)))

    def sample(
        self, start: float, end: float, *, rng: np.random.Generator
    ) -> MarkedEvents:
        start, end = v.interval(start, end)
        v.generator(rng)
        events = [x for x in self.events if start < x.time <= end]
        marks = []
        for event in events:
            value = event.sample(draws=1, rng=rng)[1][0]
            if np.shape(value) != self.event_shape:
                raise ValueError("event impact does not match schedule event_shape")
            marks.append(value)
        values = np.array(marks).reshape((len(events),) + self.event_shape)
        return MarkedEvents([x.time for x in events], values, start, end)


@dataclass(frozen=True)
class MarkedPointProcess:
    """Event times from any EventProcess, IID independent jump-size law.

    Totals are returned as samples or an empirical law. A compound Poisson sum
    generally has an atom at zero plus a continuous part, so it is not disguised
    as a continuous Distribution with a PDF.
    """

    process: EventProcess
    marks: Distribution

    def __post_init__(self) -> None:
        if not callable(getattr(self.process, "sample", None)):
            raise TypeError("process must implement sample")
        v.law(self.marks)

    def sample(
        self, start: float, end: float, *, rng: np.random.Generator
    ) -> MarkedEvents:
        times = sample_events(self.process, start, end, rng=rng)
        return MarkedEvents(times, self.marks.sample(len(times), rng=rng), start, end)

    def aggregate(
        self, start: float, end: float, *, draws: int = 1000, rng: np.random.Generator
    ):
        from ...distributions import empirical

        v.interval(start, end)
        v.generator(rng)
        draws = v.integer(draws, "draws", 1)
        return empirical([self.sample(start, end, rng=rng).total for _ in range(draws)])

    def moments(self, start: float, end: float):
        """Exact compound-Poisson (mean,covariance); marks must have moments."""
        if not isinstance(self.process, PoissonProcess):
            raise NotImplementedError(
                "analytic aggregate moments require a PoissonProcess"
            )
        mass = self.process.integrated(start, end)
        shape = self.marks.event_shape
        if mass == 0:
            return v.array(np.zeros(shape), "mean"), v.array(
                np.zeros(shape * 2), "covariance"
            )
        mean, covariance = self.marks.mean(), self.marks.covariance()
        second = covariance + (np.outer(mean, mean) if shape else mean**2)
        return v.moment(mass * mean), v.moment(mass * second)
