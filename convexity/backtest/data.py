"""Local, explicitly timestamped stock bars and supplied exchange-session tables."""
from dataclasses import dataclass
from datetime import date, datetime, timedelta
import csv
import hashlib
import json
from pathlib import Path

from convexity.runtime import Bar, Bars, Timer
from convexity.runtime.types import aware, decimal


@dataclass(frozen=True)
class HistoricalData:
    bars: tuple[Bar, ...]
    source: str = "in-memory"
    price_basis: str = "unadjusted_no_corporate_actions"

    def __post_init__(self):
        if self.price_basis not in ("unadjusted_no_corporate_actions", "split_adjusted_price_return"):
            raise ValueError("Use unadjusted_no_corporate_actions or split_adjusted_price_return; dividend/total-return accounting is unsupported.")
        if not self.bars:
            raise ValueError("Historical data is empty.")
        ordered = tuple(sorted(self.bars, key=lambda b: (b.end, b.symbol)))
        object.__setattr__(self, "bars", ordered)
        previous: dict[str, Bar] = {}
        symbols = {b.symbol for b in ordered}
        groups: dict[datetime, list[Bar]] = {}
        for bar in ordered:
            last = previous.get(bar.symbol)
            if last is not None and bar.start < last.end:
                raise ValueError(f"Duplicate/overlapping bars for {bar.symbol}.")
            previous[bar.symbol] = bar
            groups.setdefault(bar.end, []).append(bar)
        previous_end = None
        for end, panel in groups.items():
            if {b.symbol for b in panel} != symbols or len(panel) != len(symbols):
                raise ValueError(f"Incomplete bar panel at {end}; every instrument needs a bar.")
            if len({b.start for b in panel}) != 1:
                raise ValueError("All instruments must use aligned bar intervals.")
            if previous_end is not None and panel[0].start < previous_end:
                raise ValueError("Bar panels overlap.")
            previous_end = end

    @property
    def identity(self) -> str:
        payload = [(b.symbol, b.start.isoformat(), b.end.isoformat(), str(b.open), str(b.high),
                    str(b.low), str(b.close), b.volume) for b in self.bars]
        return hashlib.sha256(json.dumps([self.price_basis, payload], separators=(",", ":")).encode()).hexdigest()

    def panels(self) -> tuple[Bars, ...]:
        grouped = {}
        for bar in self.bars:
            grouped.setdefault(bar.end, {})[bar.symbol] = bar
        return tuple(Bars(t, bars) for t, bars in grouped.items())

    def resample(self, bar_size: str, calendar: "SessionCalendar", *,
                 boundaries: tuple[datetime, ...] = ()) -> "HistoricalData":
        """Aggregate OHLCV; boundaries split warmup/training/trading periods.

        Boundaries must fall between sessions. Source bars must cover each output
        interval exactly; resampling never fills gaps or splits a source bar.
        """
        from .history import HistoryError, _aggregate
        from .bar_size import parse_bar_size
        parse_bar_size(bar_size)
        for boundary in boundaries:
            aware(boundary)
            if any(session.open < boundary < session.close for session in calendar.sessions):
                raise ValueError("Resampling boundaries must fall between sessions.")
        groups: dict[int, list[Session]] = {}
        for session in calendar.sessions:
            groups.setdefault(sum(boundary <= session.open for boundary in boundaries), []).append(session)
        result = []
        used = 0
        for sessions in groups.values():
            for symbol in {bar.symbol for bar in self.bars}:
                source = [bar for bar in self.bars if bar.symbol == symbol
                          and sessions[0].open <= bar.start and bar.end <= sessions[-1].close]
                result.extend(_aggregate(source, tuple(sessions), bar_size))
                used += len(source)
        if used != len(self.bars):
            raise HistoryError("Source bars cross a resampling boundary or lie outside its calendar.")
        return HistoricalData(tuple(result), self.source, self.price_basis)

    @classmethod
    def from_csv(cls, path: str | Path, *, price_basis: str = "unadjusted_no_corporate_actions"):
        bars = []
        with Path(path).open(newline="") as stream:
            for row in csv.DictReader(stream):
                if decimal(row.get("split_factor") or "1") != 1 or decimal(row.get("dividend") or "0") != 0:
                    raise ValueError("Corporate actions require preprocessing/accounting not supported in this MVP.")
                bars.append(Bar(row["symbol"], datetime.fromisoformat(row["start"]), datetime.fromisoformat(row["end"]),
                                decimal(row["open"]), decimal(row["high"]), decimal(row["low"]), decimal(row["close"]), int(row["volume"])))
        return cls(tuple(bars), str(path), price_basis)


@dataclass(frozen=True)
class Session:
    label: date
    open: datetime
    close: datetime

    def __post_init__(self):
        aware(self.open)
        aware(self.close)
        if self.open >= self.close:
            raise ValueError("Session open must precede close.")


@dataclass(frozen=True)
class SessionCalendar:
    """An explicit session table: holidays are omitted, early closes supplied.

    Import a trusted exchange calendar as CSV or supply Session objects. There is
    no weekday-only fallback pretending to be a real exchange calendar.
    """
    sessions: tuple[Session, ...]
    name: str = "supplied"

    def __post_init__(self):
        sessions = tuple(sorted(self.sessions, key=lambda s: s.open))
        if len({s.label for s in sessions}) != len(sessions):
            raise ValueError("Duplicate session labels.")
        if any(a.close > b.open for a, b in zip(sessions, sessions[1:])):
            raise ValueError("Overlapping sessions.")
        object.__setattr__(self, "sessions", sessions)

    @classmethod
    def from_csv(cls, path: str | Path):
        with Path(path).open(newline="") as stream:
            return cls(tuple(Session(date.fromisoformat(r["session"]), datetime.fromisoformat(r["open"]),
                                     datetime.fromisoformat(r["close"])) for r in csv.DictReader(stream)), str(path))

    @classmethod
    def from_exchange(cls, name: str, start: date | str, end: date | str) -> "SessionCalendar":
        """Resolve an inclusive exchange session table, including holidays/DST.

        Requires the optional calendars extra. The resolved table and provider
        version are retained in exported backtest configuration. Supply complete
        months when constructing production monthly schedules. Calendars with
        intraday breaks are unsupported by the continuous-session stock runtime.
        """
        if not isinstance(name, str) or not name.strip():
            raise ValueError("An exchange calendar name such as XNYS is required.")
        first = date.fromisoformat(start) if isinstance(start, str) else start
        last = date.fromisoformat(end) if isinstance(end, str) else end
        if type(first) is not date or type(last) is not date or first > last:
            raise ValueError("Supply ISO dates/date objects with start <= end.")
        try:
            import exchange_calendars
        except ImportError as exc:
            raise ImportError("Install exchange calendars with: python -m pip install -e '.[calendars]'.") from exc
        # Padding permits one-day/holiday-only requests; the provider itself
        # expects a range containing at least one actual trading session.
        calendar = exchange_calendars.get_calendar(name, start=first - timedelta(days=7), end=last + timedelta(days=7))
        table = calendar.schedule.loc[first.isoformat():last.isoformat()]
        if table["break_start"].notna().any() or table["break_end"].notna().any():
            raise ValueError("Calendars with intraday breaks require an unsupported split-session model.")
        sessions = tuple(Session(label.date(), row["open"].to_pydatetime(), row["close"].to_pydatetime())
                         for label, row in table.iterrows())
        return cls(sessions, f"{name}:exchange_calendars-{exchange_calendars.__version__}")

    def daily(self, name: str, *, at: str = "close", offset: timedelta = timedelta()) -> tuple[Timer, ...]:
        if at not in ("open", "close"):
            raise ValueError("at must be open or close.")
        return tuple(Timer(getattr(s, at) + offset, name) for s in self.sessions)

    def monthly(self, name: str, *, day: str = "first", at: str = "close", offset: timedelta = timedelta()) -> tuple[Timer, ...]:
        if day not in ("first", "last") or at not in ("open", "close"):
            raise ValueError("Use first/last session and open/close.")
        groups = {}
        for session in self.sessions:
            groups.setdefault((session.label.year, session.label.month), []).append(session)
        return tuple(Timer(getattr(sessions[0 if day == "first" else -1], at) + offset, name)
                     for sessions in groups.values())
