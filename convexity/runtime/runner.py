"""Managed paper trading: data, durable timers, broker events, and reconnection."""
from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import math
import json
from dataclasses import asdict, replace
from pathlib import Path
from functools import wraps
from typing import Any, Literal, Protocol

from convexity.backtest.data import HistoricalData, SessionCalendar
from convexity.brokers.ibkr import PaperAccountConfig, PaperAccountError, Quote
from convexity.brokers.ibkr_options import OptionQuoteRequest
from convexity.brokers.ibkr_trading import PaperBroker
from convexity.brokers.ibkr_verticals import VerticalBroker
from convexity.trading import OrderJournal
from .capabilities import PAPER_CAPABILITIES
from .contracts import BarSubscription, ClockSchedule, StrategyMetadata, event_schedule, strategy_contract
from .paper import PaperRuntime
from .schedules import DurableSchedule
from .calendar_schedule import SessionSchedule
from .status import RuntimeStatus
from .state import DeliveryError, DeliveryResult
from .types import (AccountUpdate, Bar, Bars, ConnectionHealth, Instrument, OptionQuotes,
                    StockQuotes, Strategy, Timer, aware)


class DataFeedError(RuntimeError):
    """The declared feed is incomplete or stale; trading must not continue."""


@dataclass(frozen=True)
class StockSubscription:
    """Completed one-minute regular-session bars and an explicit exchange calendar.

    ``calendar`` may be a supplied session table or an exchange-calendars name.
    The named calendar is renewed daily; install convexity[calendars] to use it.
    """
    history_seconds: int = 1800
    calendar: SessionCalendar | str = "XNYS"
    max_bars: int = 10000
    warmup_sessions: int = 0

    def __post_init__(self):
        if type(self.warmup_sessions) is not int or self.warmup_sessions < 0:
            raise ValueError("warmup_sessions must be a nonnegative integer.")
        if type(self.history_seconds) is not int or not 180 <= self.history_seconds <= 86400:
            raise ValueError("history_seconds must be between 180 and 86400.")
        if not isinstance(self.calendar, SessionCalendar) and (not isinstance(self.calendar, str) or not self.calendar):
            raise ValueError("Supply a session calendar or an exchange calendar name.")
        if type(self.max_bars) is not int or self.max_bars < self.history_seconds // 60:
            raise ValueError("max_bars must retain at least the requested one-minute history window.")


@dataclass(frozen=True)
class ScheduleSpec:
    """A durable, immutable schedule; use catch_up only for overdue-safe decisions."""
    namespace: str
    timers: tuple[Timer, ...]
    policy: Literal["skip", "catch_up"] = "skip"

    def __post_init__(self):
        object.__setattr__(self, "timers", tuple(self.timers))
        if not self.namespace or self.policy not in ("skip", "catch_up"):
            raise ValueError("Supply a schedule namespace and skip/catch_up policy.")
        if any(not isinstance(timer, Timer) for timer in self.timers):
            raise TypeError("Schedules contain Timer objects.")


class PaperDataSource(Protocol):
    """Injectable observations; the runner owns validation and callback delivery."""
    async def history(self, broker: PaperBroker, symbol: str, subscription: StockSubscription) -> HistoricalData: ...
    async def option_quotes(self, broker: PaperBroker, requests: Sequence[OptionQuoteRequest]) -> tuple[Quote, ...]: ...


class IBKRDataSource:
    async def history(self, broker, symbol, subscription):
        from convexity.brokers.ibkr_data import read_stock_history
        seconds = subscription.history_seconds
        if subscription.warmup_sessions:
            calendar = _warmup_calendar(subscription, _now())
            completed = [session for session in calendar.sessions if session.close <= _now()]
            opening = completed[-subscription.warmup_sessions].open
            seconds = max(seconds, math.ceil((_now() - opening).total_seconds()) + 60)
        return await read_stock_history(broker, symbol, seconds=seconds)

    async def option_quotes(self, broker, requests):
        from convexity.brokers.ibkr_data import read_session_option_quotes
        return await read_session_option_quotes(broker, requests)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _warmup_calendar(subscription: StockSubscription, now: datetime) -> SessionCalendar:
    value = subscription.calendar
    if isinstance(value, SessionCalendar):
        calendar = value
    else:
        days = max(14, subscription.warmup_sessions * 3 + 14)
        calendar = SessionCalendar.from_exchange(value, now.date() - timedelta(days=days), now.date() + timedelta(days=10))
    completed = [session for session in calendar.sessions if session.close <= now]
    if len(completed) < subscription.warmup_sessions:
        raise DataFeedError("Session calendar does not cover the requested strategy warmup.")
    return calendar


def _panels(bars: Sequence[Bar]) -> tuple[Bars, ...]:
    grouped = defaultdict(dict)
    for bar in bars:
        grouped[bar.end][bar.symbol] = bar
    return tuple(Bars(time, values) for time, values in sorted(grouped.items()))


class PaperTrading:
    """Own a complete paper session for factory-created user strategies.

    ``await run()`` runs in the foreground. ``await start()`` waits for readiness
    and returns its background task; ``await stop()`` disconnects cleanly.
    Strategies are reconstructed after a transport outage and their runtime
    checkpoints restored. Failed strategies remain paused while order tracking,
    operator actions, and healthy strategies continue. Invalid data and account/
    journal faults stop the session. Ambiguous callbacks are never retried.
    Shutdown does not cancel orders or liquidate positions.
    """
    capabilities = PAPER_CAPABILITIES

    def __init__(self, config: PaperAccountConfig,
                 strategies: Mapping[str, Callable[[], Strategy]] | Sequence[Callable[[], Strategy]],
                 instruments: Mapping[str, Instrument] | None = None, *, path: str | Path,
                 budgets: Mapping[str, Decimal], max_order_notional: Decimal,
                 market_collar_bps: Decimal,
                 subscriptions: Mapping[str, StockSubscription] | None = None,
                 option_requests: Sequence[OptionQuoteRequest] = (),
                 schedules: Sequence[ScheduleSpec | SessionSchedule | ClockSchedule] | None = None, poll_interval: float = 1,
                 history_interval: float = 30, quote_max_age: float = 30,
                 decision_max_age: float = 120, allow_short: bool = False,
                 max_contracts: int = 1, reconnect_attempts: int = 5,
                 reconnect_delay: float = 2,
                 broker_factory: Callable[[PaperAccountConfig, OrderJournal], PaperBroker] | None = None,
                 data_source: PaperDataSource | None = None,
                 strategy_versions: Mapping[str, str] | None = None,
                 event_history_limit: int = 10000, fail_fast: bool = False):
        self._strategy_contracts = {}
        entries = tuple(strategies.items()) if isinstance(strategies, Mapping) else tuple((None, factory) for factory in strategies)
        if not entries or any(not callable(factory) for _, factory in entries):
            raise ValueError("Supply strategy factories.")
        inspected = tuple((alias, factory, factory()) for alias, factory in entries)
        declared = not isinstance(strategies, Mapping) or any(strategy_contract(instance) is not None for _, _, instance in inspected)
        if declared:
            normalized = {}
            derived_instruments = {}
            derived_subscriptions = {}
            derived_schedules = {}
            derived_versions = {}
            for alias, factory, instance in inspected:
                metadata = getattr(instance, "metadata", None)
                contract = strategy_contract(instance)
                if not isinstance(metadata, StrategyMetadata) or contract is None:
                    raise ValueError("Managed declared strategies require StrategyMetadata and StrategyContract.")
                if alias is not None and alias != metadata.name:
                    raise ValueError("Strategy mapping names must match strategy metadata names.")
                if metadata.name in normalized:
                    raise ValueError("Strategy metadata names must be unique.")
                # Reuse the inspected instance on first startup; reconnect creates fresh state.
                initial = [instance]
                @wraps(factory)
                def create(factory=factory, initial=initial, contract=contract, metadata=metadata):
                    strategy = initial.pop() if initial else factory()
                    if strategy_contract(strategy) != contract or getattr(strategy, "metadata", None) != metadata:
                        raise ValueError("Strategy metadata/contract changed on reconnect; use a new deployment.")
                    return strategy
                normalized[metadata.name] = create
                self._strategy_contracts[metadata.name] = contract
                derived_versions[metadata.name] = metadata.version
                for instrument in contract.resolved_instruments:
                    if instrument.symbol in derived_instruments and derived_instruments[instrument.symbol] != instrument:
                        raise ValueError("Strategies disagree on instrument definitions.")
                    derived_instruments[instrument.symbol] = instrument
                for sub in contract.all_subscriptions:
                    if isinstance(sub, BarSubscription):
                        if sub.bar_size != "1 min":
                            raise ValueError("Managed paper data currently supports only '1 min' bar subscriptions.")
                        for symbol in sub.symbols:
                            derived_instruments.setdefault(symbol, Instrument(symbol))
                            candidate = StockSubscription(calendar=sub.calendar, warmup_sessions=contract.warmup_sessions if sub in contract.subscriptions else 0,
                                                          max_bars=max(10000, 1440 * (contract.warmup_sessions + 1)))
                            if symbol in derived_subscriptions:
                                prior = derived_subscriptions[symbol]
                                if prior.calendar != candidate.calendar:
                                    raise ValueError("Strategies disagree on a stock feed calendar.")
                                candidate = replace(candidate, warmup_sessions=max(prior.warmup_sessions, candidate.warmup_sessions),
                                                    max_bars=max(prior.max_bars, candidate.max_bars))
                            derived_subscriptions[symbol] = candidate
                    elif (schedule := event_schedule(sub)) is not None:
                        if schedule.namespace in derived_schedules and derived_schedules[schedule.namespace] != schedule:
                            raise ValueError("Strategies disagree on a named schedule.")
                        derived_schedules[schedule.namespace] = schedule
            if instruments is not None and dict(instruments) != derived_instruments:
                raise ValueError("Configured instruments must match strategy declarations.")
            instruments = derived_instruments
            strategies = normalized
            if subscriptions is None:
                subscriptions = derived_subscriptions
            else:
                for symbol, required in derived_subscriptions.items():
                    supplied = subscriptions.get(symbol)
                    if supplied is None or supplied.calendar != required.calendar or supplied.warmup_sessions < required.warmup_sessions:
                        raise ValueError("Configured subscriptions must cover declared symbols, calendars, and warmup sessions.")
            if schedules is None:
                schedules = tuple(derived_schedules.values())
            elif any(required not in schedules for required in derived_schedules.values()):
                raise ValueError("Configured schedules must include every declared strategy schedule.")
            strategy_versions = {**derived_versions, **(strategy_versions or {})}
        else:
            normalized = {}
            for alias, factory, instance in inspected:
                initial = [instance]
                @wraps(factory)
                def legacy_create(factory=factory, initial=initial):
                    strategy = initial.pop() if initial else factory()
                    if strategy_contract(strategy) is not None:
                        raise ValueError("A legacy strategy cannot acquire a contract during reconnect.")
                    return strategy
                normalized[alias] = legacy_create
            strategies = normalized
        if instruments is None:
            raise ValueError("Legacy strategies require explicit instruments.")
        assert isinstance(strategies, Mapping)
        schedules = tuple(schedules or ())
        if not strategies or any(not name or not callable(factory) for name, factory in strategies.items()):
            raise ValueError("Supply named strategy factories, for example {'alpha': MyStrategy}.")
        for label, value in (("poll_interval", poll_interval), ("history_interval", history_interval),
                             ("reconnect_delay", reconnect_delay), ("decision_max_age", decision_max_age)):
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{label} must be finite and positive.")
        if type(reconnect_attempts) is not int or reconnect_attempts < 0:
            raise ValueError("reconnect_attempts must be a nonnegative integer.")
        if len({spec.namespace for spec in schedules}) != len(schedules):
            raise ValueError("Schedule namespaces must be unique.")
        if any(isinstance(spec, ScheduleSpec) and spec.namespace.startswith("session:") for spec in schedules):
            raise ValueError("The session: namespace prefix is reserved for recurring schedules.")
        if type(event_history_limit) is not int or event_history_limit < 1:
            raise ValueError("event_history_limit must be a positive integer.")
        if type(fail_fast) is not bool:
            raise ValueError("fail_fast must be a bool.")
        self.fail_fast = fail_fast
        self.config, self.factories, self.instruments = config, dict(strategies), dict(instruments)
        self.path = Path(path)
        self.strategy_versions = dict(strategy_versions or {})
        if set(self.strategy_versions) - set(strategies):
            raise ValueError("Strategy versions must identify registered owners.")
        self.budgets = dict(budgets)
        self.subscriptions = dict(subscriptions) if subscriptions is not None else {
            symbol: StockSubscription() for symbol in instruments}
        if set(self.subscriptions) - set(self.instruments):
            raise ValueError("Every subscribed stock must be a configured instrument.")
        self.option_requests, self.schedule_specs = tuple(option_requests), tuple(schedules)
        self.poll_interval, self.history_interval = poll_interval, history_interval
        self.reconnect_attempts, self.reconnect_delay = reconnect_attempts, reconnect_delay
        self.decision_max_age = decision_max_age
        self.max_contracts = max_contracts
        self.event_history_limit = event_history_limit
        self.runtime_options: dict[str, Any] = dict(budgets=self.budgets, max_order_notional=max_order_notional,
                                    market_collar_bps=market_collar_bps, quote_max_age=quote_max_age,
                                    allow_short=allow_short, bar_max_age=decision_max_age)
        self.broker_factory = broker_factory or (lambda conf, journal: VerticalBroker(
            conf, journal, strategy_budgets=self.budgets, max_contracts=max_contracts))
        self.data_source = data_source or IBKRDataSource()
        self.runtime: PaperRuntime | None = None
        self.health: list[ConnectionHealth] = []
        self._last_status: RuntimeStatus | None = None
        self.last_error: BaseException | None = None
        self._stop = asyncio.Event()
        self._ready = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._running = False
        self._used = False
        self._calendars: dict[str, SessionCalendar] = {}
        self._calendar_date = None
        self._active_bar_symbols: set[str] = set()
        self._schedules: list[DurableSchedule] = []
        self._schedule_owners: dict[str, tuple[str, str]] = {}
        self._last_schedule_status: tuple[dict, ...] = ()

    def status(self):
        """Return detached positions, orders, errors, and strategy state."""
        from .status import runtime_status
        if self.runtime is None:
            raise RuntimeError("Start paper trading before requesting runtime status.")
        if self._running and self.runtime.snapshot is not None:
            return runtime_status(self.runtime)
        if self._running and self._last_status is not None:
            return replace(self._last_status, strategy_states={
                owner: "paused" if state == "paused" else "reconnecting"
                for owner, state in self._last_status.strategy_states.items()})
        if self._last_status is None:
            raise RuntimeError("The paper session has no completed account snapshot.")
        return self._last_status

    async def command(self, owner, command):
        """Explicit operator action, subject to the runtime's normal safety gates."""
        if self.runtime is None or not self._running:
            raise RuntimeError("An active paper session is required.")
        return await self.runtime.command(owner, command)

    def schedule_status(self) -> tuple[dict, ...]:
        """Detached delivery records with logical schedule, owner, and durable namespace.

        Each owner's schedule is claimed independently. An uncertain row blocks
        that owner's namespace until explicitly reconciled, never other owners.
        """
        if not self._running:
            return tuple(dict(row) for row in self._last_schedule_status)
        return tuple({**row, "schedule": self._schedule_owners[schedule.namespace][0],
                      "strategy_id": self._schedule_owners[schedule.namespace][1]}
                     for schedule in self._schedules for row in schedule.status())

    def _delivery(self, result: DeliveryResult) -> None:
        if self.fail_fast:
            result.raise_for_failure()

    def _deployment(self):
        """Reject accidental reuse of ownership/checkpoints for a different setup."""
        spec = dict(account=self.config.account, client_id=self.config.client_id,
            schedule_delivery="per-owner-v1",
            instruments={name: asdict(value) for name, value in self.instruments.items()},
            strategies={name: {"factory": f"{getattr(factory, '__module__', '')}.{getattr(factory, '__qualname__', type(factory).__qualname__)}",
                               "version": self.strategy_versions.get(name, "1")}
                        for name, factory in self.factories.items()},
            contracts={owner: asdict(contract) for owner, contract in self._strategy_contracts.items()},
            risk=self.runtime_options, max_contracts=self.max_contracts,
            event_history_limit=self.event_history_limit,
            subscriptions={name: {"history_seconds": sub.history_seconds,
                            "max_bars": sub.max_bars, "warmup_sessions": sub.warmup_sessions,
                            "calendar": sub.calendar if isinstance(sub.calendar, str) else asdict(sub.calendar)}
                           for name, sub in self.subscriptions.items()},
            options=[asdict(request) for request in self.option_requests],
            schedules=[{"kind": type(spec).__name__, **asdict(spec)} for spec in self.schedule_specs])
        serialized = json.dumps(spec, default=str, sort_keys=True, indent=2)
        manifest = self.path / "deployment.json"
        if manifest.exists():
            if manifest.read_text() != serialized:
                raise ValueError("Deployment configuration differs from this persistence path. Restore the original configuration or use a new path after reconciling existing exposure. Set strategy_versions to identify changed strategy parameters.")
        else:
            manifest.write_text(serialized)

    def _sync_schedules(self, schedules: list[DurableSchedule], started_at: datetime, now: datetime) -> None:
        """Materialize recurring years as they arrive, preserving durable namespaces."""
        existing = {logical for logical, _ in self._schedule_owners.values()}
        for declaration in self.schedule_specs:
            if isinstance(declaration, (SessionSchedule, ClockSchedule)):
                # Include the adjacent year: an offset before the first session
                # can make its timer due in the previous calendar year.
                horizon = min(9999, now.year + 1)
                if isinstance(declaration, SessionSchedule):
                    years = range(declaration.start.year, min(horizon, declaration.end.year if declaration.end else horizon) + 1)
                    prefix = f"session:{declaration.namespace}"
                else:
                    years = range(started_at.year, horizon + 1)
                    prefix = declaration.namespace
                specs = (declaration.materialize(year) for year in years
                         if f"{prefix}:{year}" not in existing)
            else:
                specs = (declaration,) if declaration.namespace not in existing else ()
            for spec in specs:
                for owner in self.factories:
                    contract = self._strategy_contracts.get(owner)
                    if contract is not None and declaration not in tuple(event_schedule(sub) for sub in contract.all_subscriptions):
                        continue
                    namespace = "managed:" + json.dumps((spec.namespace, owner), separators=(",", ":"))
                    schedules.append(DurableSchedule(self.path / "schedules.sqlite", namespace,
                        spec.timers, started_at=started_at, policy=spec.policy))
                    self._schedule_owners[namespace] = (spec.namespace, owner)
                existing.add(spec.namespace)

    def _retain(self, runtime: PaperRuntime) -> None:
        """Bound working memory; execution journals keep the authoritative audit trail."""
        del runtime.events[:-self.event_history_limit]
        del runtime.errors[:-self.event_history_limit]
        del self.health[:-self.event_history_limit]
        for symbol, subscription in self.subscriptions.items():
            del runtime.history[symbol][:-subscription.max_bars]

    async def start(self) -> asyncio.Task:
        """Launch and wait until restoration, data validation, and startup succeed."""
        if self._task is not None or self._used:
            raise RuntimeError("PaperTrading is single-use; construct a new runner to restart.")
        self._task = asyncio.create_task(self.run(), name="convexity-paper")
        ready = asyncio.create_task(self._ready.wait())
        try:
            await asyncio.wait((self._task, ready), return_when=asyncio.FIRST_COMPLETED)
            if self._task.done():
                await self._task
            return self._task
        except BaseException:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            raise
        finally:
            ready.cancel()
            await asyncio.gather(ready, return_exceptions=True)

    async def stop(self) -> None:
        """Request shutdown and await resource cleanup without changing positions."""
        self._stop.set()
        if self._task is not None and self._task is not asyncio.current_task():
            await self._task

    async def _wait(self, seconds: float):
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=seconds)
        except TimeoutError:
            pass

    def _calendar(self, symbol: str, now: datetime) -> SessionCalendar:
        if self._calendar_date != now.date():
            self._calendars.clear()
            self._calendar_date = now.date()
        if symbol not in self._calendars:
            self._calendars[symbol] = _warmup_calendar(self.subscriptions[symbol], now)
        return self._calendars[symbol]

    def _validate_history(self, symbol: str, bars: Sequence[Bar], now: datetime,
                          previous: Bar | None = None) -> None:
        calendar = self._calendar(symbol, now)
        if not calendar.sessions or now > calendar.sessions[-1].close + timedelta(days=1):
            raise DataFeedError(f"{symbol}: session calendar coverage has expired.")
        if not bars:
            raise DataFeedError(f"{symbol}: no completed history.")
        if any(bar.symbol != symbol or bar.end > now or bar.end - bar.start != timedelta(minutes=1) for bar in bars):
            raise DataFeedError(f"{symbol}: expected observed one-minute bars for the declared stock.")
        if any(a.end > b.start for a, b in zip(bars, bars[1:])):
            raise DataFeedError(f"{symbol}: history overlaps or is unordered.")
        chain = ([previous] if previous is not None else []) + list(bars)
        for bar in chain:
            if not any(s.open <= bar.start < bar.end <= s.close for s in calendar.sessions):
                raise DataFeedError(f"{symbol}: bar lies outside the supplied regular-session calendar.")
        for first, second in zip(chain, chain[1:]):
            # A gap is allowed only when every instant between bars is outside
            # trading sessions (overnight, holiday, weekend, early close).
            if any(max(first.end, session.open) < min(second.start, session.close)
                   for session in calendar.sessions):
                raise DataFeedError(f"{symbol}: missing regular-session bars after {first.end.isoformat()}.")
        active = next((s for s in calendar.sessions if s.open <= now <= s.close), None)
        if active and now >= active.open + timedelta(minutes=1) and (now - bars[-1].end).total_seconds() > self.decision_max_age:
            raise DataFeedError(f"{symbol}: latest decision data is stale.")

    def _active_feed_symbols(self, runtime: PaperRuntime) -> set[str]:
        if not self._strategy_contracts:
            return set(self.subscriptions)
        return {symbol for book in runtime._subscriptions.values() for sub in book.active.values()
                if isinstance(sub, BarSubscription) for symbol in sub.symbols}

    def _validate_decisions(self, runtime: PaperRuntime) -> None:
        for symbol in self._active_feed_symbols(runtime):
            # A newly requested feed is acquired on the next runner iteration.
            # Subscription acknowledgments do not invent observed history.
            if symbol not in self._active_bar_symbols:
                continue
            self._validate_history(symbol, runtime.history[symbol][-1:], _now())

    async def _history(self, runtime: PaperRuntime, *, initial: bool):
        backfill, decisions = [], []
        observations = {}
        active = self._active_feed_symbols(runtime)
        for symbol, subscription in self.subscriptions.items():
            if symbol not in active:
                continue
            data = await self.data_source.history(runtime.broker, symbol, subscription if initial else replace(subscription, warmup_sessions=0))
            observations[symbol] = tuple(data.bars)
        now = _now()
        # Use a common completed watermark. A faster feed cannot trigger a
        # decision while another subscribed stock is still missing that minute.
        watermark = min((values[-1].end for values in observations.values() if values), default=now)
        for symbol, observed in observations.items():
            self._validate_history(symbol, observed, now)
            values = tuple(bar for bar in observed if bar.end <= watermark)
            previous = runtime.history[symbol][-1] if runtime.history[symbol] else None
            fresh = tuple(bar for bar in values if previous is None or bar.end > previous.end)
            if initial:
                self._validate_history(symbol, values, now)
                if self.subscriptions[symbol].warmup_sessions:
                    completed = [session for session in self._calendar(symbol, now).sessions if session.close <= now]
                    opening = completed[-self.subscriptions[symbol].warmup_sessions].open
                    if not values or values[0].start > opening:
                        raise DataFeedError(f"{symbol}: feed does not cover declared warmup sessions.")
                backfill.extend(values)
            elif fresh:
                self._validate_history(symbol, fresh, now, previous if symbol in self._active_bar_symbols else None)
                if (now - fresh[-1].end).total_seconds() <= self.decision_max_age:
                    backfill.extend(fresh[:-1])
                    decisions.append(fresh[-1])
                else:
                    backfill.extend(fresh)
            else:
                self._validate_history(symbol, values, now)
        if initial:
            await runtime.warmup(_panels(backfill))
            for owner, contract in self._strategy_contracts.items():
                selected = []
                for symbol in contract.observed_symbols:
                    calendar = self._calendar(symbol, now)
                    completed = [session for session in calendar.sessions if session.close <= now]
                    selected_sessions = completed[-contract.warmup_sessions:] if contract.warmup_sessions else []
                    selected_sessions += [session for session in calendar.sessions if session.open <= now < session.close]
                    selected.extend(bar for bar in backfill if bar.symbol == symbol and
                                    any(session.open <= bar.start < bar.end <= session.close for session in selected_sessions))
                runtime.warmup_history(owner, selected)
        else:
            if backfill:
                await runtime.ingest_history(_panels(backfill))
            for event in _panels(decisions):
                self._delivery(await runtime.step(event))
        self._active_bar_symbols = active

    @staticmethod
    def _transport_failure(error: BaseException, broker: PaperBroker) -> bool:
        if isinstance(error, (ConnectionError, TimeoutError)):
            return True
        # A paper-account/journal violation is never a reconnect signal. The
        # stock broker currently reports socket loss as PaperAccountError.
        return (isinstance(error, PaperAccountError) and str(error) == "Paper broker is not connected."
                and getattr(broker, "_fault", None) is None)

    async def _observe(self, runtime):
        self._delivery(await runtime.refresh())
        if self.option_requests:
            quotes = await self.data_source.option_quotes(runtime.broker, self.option_requests)
            requested = {(r.symbol, r.expiry.strftime("%Y%m%d"), r.strike, r.right, r.trading_class, r.multiplier, r.currency)
                         for r in self.option_requests}
            observed = {(q.contract.symbol, q.contract.expiry[:8], q.contract.strike, q.contract.right,
                         q.contract.trading_class, q.contract.multiplier, q.contract.currency) for q in quotes}
            if len(quotes) != len(self.option_requests) or requested != observed:
                raise DataFeedError("The option source did not return every declared subscription.")
            self._delivery(await runtime.step(OptionQuotes(_now(), quotes)))

    async def run(self, *, until: datetime | None = None) -> None:
        """Run until stopped or a UTC-aware deadline; surface terminal failures."""
        if self._used:
            raise RuntimeError("PaperTrading is single-use.")
        if until is not None:
            aware(until)
        self._used = self._running = True
        self._task = asyncio.current_task()
        schedules = self._schedules
        started_at = _now()
        schedule_cutoff = started_at
        attempts = 0
        lock = None
        try:
            self.path.mkdir(parents=True, exist_ok=True)
            import fcntl
            lock = (self.path / "runner.lock").open("a")
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("Another PaperTrading runner owns this persistence path.") from error
            self._deployment()
            self._sync_schedules(schedules, started_at, _now())
            with OrderJournal(self.path / "orders.sqlite") as journal:
                while not self._stop.is_set() and (until is None or _now() < until):
                    broker = self.broker_factory(self.config, journal)
                    strategies = {owner: factory() for owner, factory in self.factories.items()}
                    runtime = PaperRuntime(broker, strategies, self.instruments, defer_start=True,
                        decision_guard=lambda: self._validate_decisions(runtime), **self.runtime_options)
                    self.runtime = runtime
                    started = False
                    try:
                        async with runtime:
                            try:
                                await self._history(runtime, initial=True)
                                self._delivery(await runtime.start())
                                started = True
                                health = ConnectionHealth(_now(), True, "Connected, reconciled, and history validated")
                                self.health.append(health)
                                self._delivery(await runtime.step(health))
                                if attempts:
                                    # A transport outage must not turn missed entries
                                    # into delayed orders; overdue-safe exits catch up.
                                    resumed_at = _now()
                                    schedule_cutoff = resumed_at
                                    for schedule in schedules:
                                        if schedule.policy == "skip":
                                            schedule.started_at = resumed_at
                                self._ready.set()
                                next_history = asyncio.get_running_loop().time() + self.history_interval
                                while not self._stop.is_set() and (until is None or _now() < until):
                                    # Data is validated before any timer/quote/fill decision.
                                    if asyncio.get_running_loop().time() >= next_history or self._active_feed_symbols(runtime) != self._active_bar_symbols:
                                        await self._history(runtime, initial=False)
                                        next_history = asyncio.get_running_loop().time() + self.history_interval
                                    self._validate_decisions(runtime)
                                    await self._observe(runtime)
                                    poll_time = _now()
                                    self._sync_schedules(schedules, schedule_cutoff, poll_time)
                                    while True:
                                        pending = []
                                        for schedule in schedules:
                                            owner = self._schedule_owners[schedule.namespace][1]
                                            if owner in runtime.paused:
                                                continue
                                            rows = schedule.status()
                                            if any(row["status"] == "uncertain" for row in rows):
                                                # The claim is not replayed; other owners have
                                                # independent durable delivery namespaces.
                                                continue
                                            pending.extend((datetime.fromisoformat(row["due"]), row["name"], schedule.namespace, schedule)
                                                for row in rows if row["status"] == "pending" and datetime.fromisoformat(row["due"]) <= poll_time)
                                        if not pending:
                                            break
                                        schedule = min(pending, key=lambda item: item[:3])[3]
                                        owner = self._schedule_owners[schedule.namespace][1]
                                        async def deliver(run):
                                            return await runtime.step(run.timer, owners=(owner,))
                                        try:
                                            await schedule.poll_async(poll_time, deliver, limit=1)
                                        except DeliveryError:
                                            # poll_async retains the failed claim as uncertain.
                                            # The runtime has paused its owner; keep observing
                                            # existing orders and delivering healthy owners.
                                            if self.fail_fast:
                                                raise
                                    attempts = 0
                                    self._retain(runtime)
                                    delay = self.poll_interval if until is None else min(self.poll_interval, max(0, (until - _now()).total_seconds()))
                                    await self._wait(delay)
                            except Exception as error:
                                health = ConnectionHealth(_now(), False, f"{type(error).__name__}: {error}")
                                self.health.append(health)
                                if started:
                                    self._delivery(await runtime.step(health))
                                raise
                        self._last_status = replace(self.status(), strategy_states={
                            owner: "paused" if owner in runtime.paused else "stopped" for owner in self.factories})
                        break
                    except BaseException as error:
                        if runtime.snapshot is not None:
                            self._last_status = replace(self.status(), strategy_states={
                                owner: "paused" if owner in runtime.paused else "stopped" for owner in self.factories},
                                errors=tuple(runtime.errors) + (f"{type(error).__name__}: {error}",))
                        self.health.append(ConnectionHealth(_now(), False, f"{type(error).__name__}: {error}"))
                        if not isinstance(error, Exception) or not self._transport_failure(error, broker) or attempts >= self.reconnect_attempts:
                            raise
                        attempts += 1
                        await self._wait(self.reconnect_delay * min(attempts, 10))
        except BaseException as error:
            self.last_error = error
            raise
        finally:
            self._last_schedule_status = self.schedule_status()
            self._running = False
            for schedule in schedules:
                schedule.close()
            if lock is not None:
                lock.close()
