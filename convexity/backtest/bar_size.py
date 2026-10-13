"""Duration parsing and explicit session/calendar OHLCV aggregation rules."""
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
import re

from .data import Session


@dataclass(frozen=True)
class BarSize:
    count: int
    unit: str

    @property
    def label(self) -> str:
        if self.unit == "second":
            if self.count % 3600 == 0:
                n, unit = self.count // 3600, "hour"
            elif self.count % 60 == 0:
                n, unit = self.count // 60, "min"
            else:
                n, unit = self.count, "sec"
        else:
            n, unit = self.count, self.unit
        return f"{n} {unit}{'s' if n != 1 else ''}"

    @property
    def source(self) -> str:
        if self.unit != "second":
            return "1 day"
        if self.count % 60 == 0:
            return "1 min"
        for seconds in (30, 15, 10, 5, 1):
            if self.count % seconds == 0:
                return f"{seconds} secs"  # IBKR spells even 1 second as '1 secs'.
        raise AssertionError("Unreachable")

    @property
    def multiday(self) -> bool:
        return self.unit != "second" and (self.unit != "day" or self.count > 1)


def parse_bar_size(value: str) -> BarSize:
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([a-zA-Z]+)\s*", value) if isinstance(value, str) else None
    if not match:
        raise ValueError("bar_size must be a positive duration, e.g. '7 mins', '90 secs', '2 hours', or '1 week'.")
    number, unit = Decimal(match[1]), match[2].lower()
    seconds = {"s": 1, "sec": 1, "secs": 1, "second": 1, "seconds": 1,
               "m": 60, "min": 60, "mins": 60, "minute": 60, "minutes": 60,
               "h": 3600, "hr": 3600, "hrs": 3600, "hour": 3600, "hours": 3600}
    periods = {"d": "day", "day": "day", "days": "day",
               "w": "week", "wk": "week", "wks": "week", "week": "week", "weeks": "week",
               "mo": "month", "month": "month", "months": "month",
               "y": "year", "yr": "year", "year": "year", "years": "year"}
    if unit in seconds:
        number *= seconds[unit]
        unit = "second"
    elif unit in periods:
        unit = periods[unit]
    else:
        raise ValueError("Use seconds, minutes, hours, days, weeks, months, or years for bar_size.")
    if number <= 0 or number != number.to_integral_value():
        raise ValueError("Bar sizes must be positive, with whole-second intraday resolution or whole calendar/session periods.")
    return BarSize(int(number), unit)


def intervals(sessions: tuple[Session, ...], bar_size: str):
    """Intraday: session open. Days: N sessions. Weeks: Monday. Months: calendar."""
    spec = parse_bar_size(bar_size)
    if spec.unit == "second":
        for session in sessions:
            start = session.open
            # Limit before constructing timedelta, so even very large sizes clip safely.
            step = timedelta(seconds=min(spec.count, int((session.close - start).total_seconds())))
            while start < session.close:
                end = min(start + step, session.close)
                yield start, end
                start = end
        return
    groups: dict[int, list[Session]] = {}
    for index, session in enumerate(sessions):
        label = session.label
        if spec.unit == "day":
            key = index // spec.count
        elif spec.unit == "week":
            key = ((label - date(1970, 1, 5)).days // 7) // spec.count
        elif spec.unit == "month":
            key = ((label.year - 1970) * 12 + label.month - 1) // spec.count
        else:
            key = (label.year - 1970) // spec.count
        groups.setdefault(key, []).append(session)
    for group in groups.values():
        yield group[0].open, group[-1].close
