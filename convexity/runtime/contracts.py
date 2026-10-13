"""Strategy-owned declarations shared by simulated and broker runtimes."""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo
from typing import Any

from .calendar_schedule import SessionSchedule
from .types import (Bars, BracketOrder, BrokerOrderUpdate, CancelOrder, COMMAND_TYPES, Context, FillEvent, OCOOrders, SubscriptionUpdate,
                    Instrument, OptionQuotes, OrderUpdate, Rejection, StockQuotes,
                    SubmitOrder, Timer, VerticalOrder)


class ContractViolation(ValueError):
    """A strategy attempted an action outside its declared trading contract."""


@dataclass(frozen=True)
class StrategyMetadata:
    name: str
    version: str = "1"

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip() or not isinstance(self.version, str) or not self.version.strip():
            raise ValueError("Strategy metadata requires a nonempty name and version.")


@dataclass(frozen=True)
class BarSubscription:
    symbols: tuple[str, ...]
    bar_size: str = "1 day"
    calendar: str = "XNYS"

    def __post_init__(self):
        if isinstance(self.symbols, str):
            raise TypeError("BarSubscription.symbols must be a tuple of symbols.")
        object.__setattr__(self, "symbols", tuple(self.symbols))
        if not self.symbols or any(not isinstance(s, str) or not s.strip() for s in self.symbols) or len(set(self.symbols)) != len(self.symbols):
            raise ValueError("Bar subscriptions require unique nonempty symbols.")
        if not self.bar_size or not self.calendar:
            raise ValueError("Bar subscriptions require a bar size and calendar.")


@dataclass(frozen=True)
class ClockSchedule:
    """A typed local time resolved once per exchange session into a UTC Timer.

    Use as EventSubscription(Timer, schedule=ClockSchedule(...)). Times outside
    a session (including early closes) are skipped; DST follows the named zone.
    """
    name: str
    at: time = time(15, 50)
    timezone: str = "America/New_York"
    calendar: str = "XNYS"

    def __post_init__(self):
        if not self.name or not self.calendar:
            raise ValueError("Clock schedules require a name and calendar.")
        if type(self.at) is not time:
            raise TypeError("ClockSchedule.at requires datetime.time, e.g. time(15, 50).")
        if self.at.tzinfo is not None:
            raise ValueError("Use a local wall-clock time and the timezone field.")
        ZoneInfo(self.timezone)

    @property
    def namespace(self) -> str:
        return f"clock:{self.name}:{self.at}:{self.timezone}:{self.calendar}"

    def materialize(self, year: int):
        from convexity.backtest.data import SessionCalendar
        from .runner import ScheduleSpec
        calendar = SessionCalendar.from_exchange(self.calendar, date(year, 1, 1), date(year, 12, 31))
        zone = ZoneInfo(self.timezone)
        timers = []
        for session in calendar.sessions:
            local_date = session.open.astimezone(zone).date()
            last_date = session.close.astimezone(zone).date()
            while local_date <= last_date:
                due = datetime.combine(local_date, self.at, zone).astimezone(timezone.utc)
                if session.open <= due <= session.close:
                    timers.append(Timer(due, self.name))
                local_date += timedelta(days=1)
        return ScheduleSpec(f"{self.namespace}:{year}", tuple(timers))


class ClockSubscription(ClockSchedule):
    """Compatibility spelling; prefer EventSubscription(Timer, schedule=ClockSchedule(...))."""

    def __init__(self, name: str, at: time | str = time(15, 50),
                 timezone: str = "America/New_York", calendar: str = "XNYS"):
        super().__init__(name, time.fromisoformat(at) if isinstance(at, str) else at, timezone, calendar)


@dataclass(frozen=True)
class EventSubscription:
    """Opt in to an event class. Timer subscriptions also declare their schedule."""
    event_type: type
    schedule: ClockSchedule | SessionSchedule | None = None

    def __post_init__(self):
        if not isinstance(self.event_type, type):
            raise TypeError("EventSubscription requires an event class.")
        if self.event_type is Bars:
            raise ValueError("Use BarSubscription for market bars.")
        if self.event_type is Timer:
            if not isinstance(self.schedule, (ClockSchedule, SessionSchedule)):
                raise ValueError("Timer subscriptions require a ClockSchedule or SessionSchedule.")
        elif self.schedule is not None:
            raise ValueError("Only Timer events accept a schedule.")


def event_schedule(subscription) -> ClockSchedule | SessionSchedule | None:
    """Resolve the schedule for a Timer event, including legacy declarations."""
    if isinstance(subscription, EventSubscription):
        return subscription.schedule
    if isinstance(subscription, (ClockSubscription, SessionSchedule)):
        return subscription
    return None


@dataclass(frozen=True)
class StrategyContract:
    instruments: tuple[str | Instrument, ...]
    subscriptions: tuple[BarSubscription | ClockSubscription | EventSubscription | SessionSchedule, ...] = ()
    warmup_sessions: int = 0
    available_subscriptions: tuple[BarSubscription | ClockSubscription | EventSubscription | SessionSchedule, ...] = ()

    def __post_init__(self):
        if isinstance(self.instruments, str):
            raise TypeError("Contract instruments must be a tuple.")
        object.__setattr__(self, "instruments", tuple(self.instruments))
        object.__setattr__(self, "subscriptions", tuple(self.subscriptions))
        object.__setattr__(self, "available_subscriptions", tuple(self.available_subscriptions))
        if any(not isinstance(i, (str, Instrument)) or (isinstance(i, str) and not i.strip()) for i in self.instruments):
            raise ValueError("Contract instruments must be symbols or Instrument objects.")
        if len(set(self.symbols)) != len(self.symbols):
            raise ValueError("Contract instruments must be unique.")
        if any(not isinstance(s, (BarSubscription, ClockSubscription, EventSubscription, SessionSchedule)) for s in self.all_subscriptions):
            raise TypeError("Unsupported strategy subscription.")
        if type(self.warmup_sessions) is not int or self.warmup_sessions < 0:
            raise ValueError("warmup_sessions must be a nonnegative integer.")
        clocks = [schedule.name for s in self.subscriptions if (schedule := event_schedule(s)) is not None]
        if len(set(clocks)) != len(clocks):
            raise ValueError("Clock names must be unique within a strategy contract.")

    @property
    def all_subscriptions(self) -> tuple:
        return tuple(dict.fromkeys((*self.subscriptions, *self.available_subscriptions)))

    @property
    def available_observed_symbols(self) -> frozenset[str]:
        return frozenset(symbol for sub in self.all_subscriptions if isinstance(sub, BarSubscription) for symbol in sub.symbols)

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(i if isinstance(i, str) else i.symbol for i in self.instruments)

    @property
    def resolved_instruments(self) -> tuple[Instrument, ...]:
        return tuple(Instrument(i) if isinstance(i, str) else i for i in self.instruments)

    @property
    def observed_symbols(self) -> frozenset[str]:
        return frozenset(s for sub in self.subscriptions if isinstance(sub, BarSubscription) for s in sub.symbols)


def strategy_contract(strategy) -> StrategyContract | None:
    contract = getattr(strategy, "contract", None)
    if contract is not None and not isinstance(contract, StrategyContract):
        raise TypeError("Strategy.contract must be a StrategyContract.")
    return contract


def validate_commands(strategy, commands) -> None:
    """Validate an entire callback batch before any command can reach a venue."""
    contract = strategy_contract(strategy)
    for command in commands:
        if not isinstance(command, COMMAND_TYPES):
            raise TypeError("Strategy callbacks must return runtime commands.")
        if isinstance(command, (BracketOrder, OCOOrders)):
            from .orders import expand_orders
            validate_commands(strategy, expand_orders(command))
        if contract is not None and isinstance(command, (SubmitOrder, VerticalOrder)) and command.symbol not in contract.symbols:
            raise ContractViolation(f"Instrument {command.symbol!r} is outside the strategy contract; declared trading instruments: {', '.join(contract.symbols) or '(none)' }.")


def select_event(strategy, event, *, subscriptions=None):
    """Return a subscription-scoped event or None if the owner did not subscribe."""
    contract = strategy_contract(strategy)
    if contract is None:
        return event
    subscriptions = contract.subscriptions if subscriptions is None else subscriptions
    observed = frozenset(symbol for sub in subscriptions if isinstance(sub, BarSubscription) for symbol in sub.symbols)
    if isinstance(event, Bars):
        bars = {s: b for s, b in event.bars.items() if s in observed}
        return Bars(event.time, bars) if bars else None
    if isinstance(event, Timer):
        return event if any((schedule := event_schedule(s)) is not None and schedule.name == event.name for s in subscriptions) else None
    if not any(isinstance(s, EventSubscription) and isinstance(event, s.event_type) for s in subscriptions):
        return None
    visible = set(contract.symbols) | observed
    if isinstance(event, StockQuotes):
        quotes = {s: q for s, q in event.quotes.items() if s in visible}
        return StockQuotes(event.time, quotes) if quotes else None
    if isinstance(event, OptionQuotes):
        quotes = tuple(q for q in event.quotes if q.contract.symbol in visible)
        return OptionQuotes(event.time, quotes) if quotes else None
    return event


def dispatch_event(strategy, event, context) -> Any:
    """Specific callbacks take precedence; on_event is the fallback, once only."""
    names = []
    if isinstance(event, SubscriptionUpdate):
        names = ["on_subscription"]
    elif isinstance(event, Bars):
        names = ["on_bar"]
    elif isinstance(event, Timer):
        names = ["on_timer"]
    elif isinstance(event, FillEvent):
        names = ["on_fill"]
    elif isinstance(event, OrderUpdate):
        names = (["on_partial"] if event.order.status == "partially_filled" else []) + ["on_order"]
    elif isinstance(event, BrokerOrderUpdate):
        names = (["on_partial"] if 0 < event.record.filled_quantity < event.record.request.quantity else []) + ["on_order"]
    elif isinstance(event, Rejection):
        names = ["on_rejection"]
    for name in names:
        callback = getattr(strategy, name, None)
        if callable(callback):
            return callback(event, context)
    return strategy.on_event(event, context)


def filter_context(strategy, context: Context, *, subscriptions=None) -> Context:
    contract = strategy_contract(strategy)
    if contract is None:
        return context
    subscriptions = contract.subscriptions if subscriptions is None else subscriptions
    observed = frozenset(symbol for sub in subscriptions if isinstance(sub, BarSubscription) for symbol in sub.symbols)
    visible = set(contract.symbols) | observed
    return replace(context,
        positions={s: p for s, p in context.positions.items() if s in visible},
        account_positions={s: p for s, p in context.account_positions.items() if s in visible},
        prices={s: p for s, p in context.prices.items() if s in visible},
        ask_prices={s: p for s, p in context.ask_prices.items() if s in visible},
        bid_prices={s: p for s, p in context.bid_prices.items() if s in visible},
        instrument_currencies={s: c for s, c in context.instrument_currencies.items() if s in visible},
        marks={s: p for s, p in context.marks.items() if s in visible},
        mark_times={s: p for s, p in context.mark_times.items() if s in visible},
        history={s: bars for s, bars in context.history.items() if s in observed})


def assert_contract_unchanged(strategy, registered: StrategyContract | None) -> None:
    if strategy_contract(strategy) != registered:
        raise ContractViolation("Strategy contract changed after registration; create a fresh runtime for a different contract.")
