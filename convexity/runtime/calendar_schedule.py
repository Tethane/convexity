"""Declarative recurring exchange schedules for continuously managed deployments."""

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Literal


@dataclass(frozen=True)
class SessionSchedule:
    """Repeat on real exchange sessions starting on an explicit session date.

    Full calendar years resolve first/last monthly sessions correctly. The runner
    adds each new year's durable namespace automatically, retaining prior years'
    delivery history. ``start`` is part of the stable deployment specification;
    do not replace it with today's date when restarting an existing deployment.
    """

    namespace: str
    name: str
    start: date
    frequency: Literal["daily", "monthly"] = "daily"
    calendar: str = "XNYS"
    at: Literal["open", "close"] = "close"
    offset: timedelta = timedelta()
    day: Literal["first", "last"] = "first"
    policy: Literal["skip", "catch_up"] = "skip"
    end: date | None = None

    def __post_init__(self):
        if not self.namespace or not self.name or not self.calendar:
            raise ValueError(
                "Supply schedule namespace, timer name, and exchange calendar."
            )
        if type(self.start) is not date or (
            self.end is not None
            and (type(self.end) is not date or self.end < self.start)
        ):
            raise ValueError("Use session dates with start <= end.")
        if (
            self.frequency not in ("daily", "monthly")
            or self.at not in ("open", "close")
            or self.day not in ("first", "last")
            or self.policy not in ("skip", "catch_up")
            or not isinstance(self.offset, timedelta)
        ):
            raise ValueError("Invalid recurring session schedule.")
        if abs(self.offset) >= timedelta(days=1):
            raise ValueError(
                "Session offsets must be less than one day; use an explicit ScheduleSpec for other dates."
            )

    def materialize(self, year: int):
        from convexity.backtest import SessionCalendar
        from .runner import ScheduleSpec

        if type(year) is not int or not 1 <= year <= 9999:
            raise ValueError("Supply a calendar year.")
        calendar = SessionCalendar.from_exchange(
            self.calendar, date(year, 1, 1), date(year, 12, 31)
        )
        sessions = calendar.sessions
        if self.frequency == "monthly":
            groups = {}
            for session in sessions:
                groups.setdefault(session.label.month, []).append(session)
            sessions = tuple(
                group[0 if self.day == "first" else -1] for group in groups.values()
            )
        selected = SessionCalendar(
            tuple(
                session
                for session in sessions
                if self.start <= session.label
                and (self.end is None or session.label <= self.end)
            ),
            calendar.name,
        )
        return ScheduleSpec(
            f"session:{self.namespace}:{year}",
            selected.daily(self.name, at=self.at, offset=self.offset),
            self.policy,
        )
