"""Strategy-driven historical research with cached feeds and simulated execution."""
from __future__ import annotations

from convexity.runtime.account import FXRate

import asyncio
import csv
from collections.abc import Callable, Mapping, Sequence
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
import json
from pathlib import Path
from threading import Lock
from typing import Any, Protocol, TYPE_CHECKING
import weakref
from zoneinfo import ZoneInfo

from convexity.runtime import Bars, FillEvent, OrderUpdate, Rejection, Instrument, RiskLimits, Strategy, Timer
from convexity.runtime.contracts import BarSubscription, ClockSchedule, EventSubscription, StrategyContract, StrategyMetadata, event_schedule
from convexity.runtime.calendar_schedule import SessionSchedule
from convexity.runtime.types import SubscriptionUpdate
from convexity.runtime.capabilities import RESEARCH_CAPABILITIES
from .data import SessionCalendar
from .bar_size import parse_bar_size
from .exchange import ExecutionModel
from .runner import Backtest, BacktestResult, serialize

if TYPE_CHECKING:
    from .history import HistoricalDataset, HistoryRequest


class HistoricalProvider(Protocol):
    async def fetch(self, request: HistoryRequest, *, refresh: bool = False) -> HistoricalDataset: ...


class Research:
    """An inclusive session-date interval, independent of any trading strategy.

    ``Research(start, end).run(MyStrategy)`` discovers instruments, subscriptions,
    warmup and identity from the strategy. Infrastructure overrides belong to
    ``run``. The default historical source is Alpaca SIP, with read-only IBKR
    fallback and separate durable SQLite caches. Async applications use
    ``await research.run_async(MyStrategy)``.
    """

    capabilities = RESEARCH_CAPABILITIES

    def __init__(self, start: date | str, end: date | str):
        self.start = date.fromisoformat(start) if isinstance(start, str) else start
        self.end = date.fromisoformat(end) if isinstance(end, str) else end
        if type(self.start) is not date or type(self.end) is not date or self.start > self.end:
            raise ValueError("Supply inclusive ISO dates/date objects with start <= end.")
        self._instance_refs: list[weakref.ReferenceType[Strategy]] = []
        self._previous_instances: tuple[Strategy, ...] = ()
        self._lock = Lock()

    def _instantiate(self, factories):
        if isinstance(factories, Mapping):
            raise TypeError("Strategies name themselves through StrategyMetadata; pass a factory or sequence of factories.")
        values = (factories,) if callable(factories) else tuple(factories)
        if not values or any(not callable(factory) for factory in values):
            raise ValueError("Pass a strategy class/factory or a nonempty sequence of factories.")
        strategies = {}
        self._instance_refs = [reference for reference in self._instance_refs if reference() is not None]
        for factory in values:
            strategy = factory()
            if not isinstance(strategy, Strategy):
                raise TypeError("Strategy factories must return Strategy instances.")
            if not isinstance(strategy.metadata, StrategyMetadata):
                raise ValueError("Research strategies require metadata = StrategyMetadata(name=...).")
            if not isinstance(strategy.contract, StrategyContract):
                raise ValueError("Research strategies require contract = StrategyContract(instruments=..., subscriptions=...).")
            if any(strategy is reference() for reference in self._instance_refs):
                raise ValueError("Strategy factory reused a previous instance; return fresh state for every run.")
            if strategy.metadata.name in strategies:
                raise ValueError("Strategy metadata names must be unique within a research run.")
            strategies[strategy.metadata.name] = strategy
        self._previous_instances = tuple(strategies.values())
        self._instance_refs.extend(weakref.ref(strategy) for strategy in strategies.values())
        return strategies

    @staticmethod
    def _calendar_start(calendar: str, first: date, sessions: int) -> date:
        days = max(14, sessions * 2)
        while True:
            prior = SessionCalendar.from_exchange(calendar, first - timedelta(days=days), first - timedelta(days=1))
            if len(prior.sessions) >= sessions:
                return prior.sessions[-sessions].label
            days *= 2

    @staticmethod
    def _timers(subscriptions, calendar: SessionCalendar) -> tuple[Timer, ...]:
        values = []
        first, last = calendar.sessions[0].open, calendar.sessions[-1].close
        for declaration in subscriptions:
            subscription = event_schedule(declaration)
            if isinstance(subscription, ClockSchedule):
                zone = ZoneInfo(subscription.timezone)
                at = subscription.at
                for session in calendar.sessions:
                    local_date = session.open.astimezone(zone).date()
                    final_date = session.close.astimezone(zone).date()
                    while local_date <= final_date:
                        instant = datetime.combine(local_date, at, zone).astimezone(timezone.utc)
                        # Interpret the date in the subscription's timezone; a
                        # US afternoon can be the following morning in Asia.
                        if session.open <= instant <= session.close:
                            values.append(Timer(instant, subscription.name))
                        local_date += timedelta(days=1)
            elif isinstance(subscription, SessionSchedule):
                for year in range(max(1, first.year - 1), min(9999, last.year + 1) + 1):
                    values.extend(subscription.materialize(year).timers)
        return tuple(sorted({timer for timer in values if first <= timer.time <= last}, key=lambda timer: (timer.time, timer.name)))

    def run(self, strategies, **options) -> BacktestResult:
        """Run fresh strategy factories synchronously; raise with partial results on failure."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.run_async(strategies, **options))
        raise RuntimeError("An event loop is already running; use await research.run_async(MyStrategy).")

    async def run_async(self, strategies: Callable[[], Strategy] | Sequence[Callable[[], Strategy]], *,
                        provider: HistoricalProvider | None = None,
                        cache: str | Path | None = ".convexity/history.sqlite3",
                        execution: ExecutionModel | str = "standard",
                        initial_cash: Decimal | int | str | dict[str, Decimal] = 100_000,
                        base_currency: str = "USD", fx_rates: tuple[FXRate, ...] = (),
                        limits: RiskLimits | None = None,
                        output: str | Path | None = None, refresh: bool = False,
                        metadata: Mapping[str, Any] | None = None) -> BacktestResult:
        """Discover feed needs from contracts, fetch missing data, then replay events.

        ``cache=None`` disables the outer persistent cache for explicit providers.
        Execution uses daily prices unless subscriptions require finer timing.
        Execution prices are private to the simulator; only subscribed observations
        populate strategy history and trigger bar callbacks.
        """
        from .history import HistoryRequest, validate_coverage
        from .cache import CachedHistoricalProvider
        from .providers import FallbackHistoricalProvider
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("A Research instance cannot run concurrently; create another instance.")
        try:
            instances = self._instantiate(strategies)
            subscriptions = tuple(sub for strategy in instances.values() for sub in strategy.contract.all_subscriptions)
            unsupported = [sub.event_type.__name__ for sub in subscriptions
                           if isinstance(sub, EventSubscription) and sub.event_type not in (Timer, FillEvent, OrderUpdate, Rejection, SubscriptionUpdate)]
            if unsupported:
                raise NotImplementedError(f"This simulator has no event source for: {', '.join(sorted(set(unsupported)))}. Supported event subscriptions are Timer, FillEvent, OrderUpdate, Rejection and SubscriptionUpdate, plus bar subscriptions.")
            schedules = tuple(schedule for sub in subscriptions if (schedule := event_schedule(sub)) is not None)
            if any(isinstance(schedule, ClockSchedule) and schedule.at.microsecond for schedule in schedules):
                raise ValueError("Clock subscriptions currently require whole-second resolution.")
            bar_subscriptions = tuple(sub for sub in subscriptions if isinstance(sub, BarSubscription))
            calendars = {sub.calendar for sub in subscriptions if isinstance(sub, BarSubscription)} | {schedule.calendar for schedule in schedules}
            if len(calendars) > 1:
                raise ValueError("This simulator currently requires subscriptions on one exchange calendar.")
            calendar_name = next(iter(calendars), "XNYS")
            bar_sizes = {parse_bar_size(sub.bar_size).label for sub in bar_subscriptions}
            for strategy in instances.values():
                sizes_by_symbol = {}
                for sub in strategy.contract.all_subscriptions:
                    if isinstance(sub, BarSubscription):
                        for symbol in sub.symbols:
                            size = parse_bar_size(sub.bar_size).label
                            if symbol in sizes_by_symbol and sizes_by_symbol[symbol] != size:
                                raise ValueError("A strategy may subscribe to one bar resolution per symbol; use separate strategy instances for multiple resolutions of the same symbol.")
                            sizes_by_symbol[symbol] = size
            calendar = SessionCalendar.from_exchange(calendar_name, self.start, self.end)
            if not calendar.sessions:
                raise ValueError("Requested dates contain no exchange trading sessions.")
            first, last = calendar.sessions[0].open, calendar.sessions[-1].close
            timer_owners = {}
            for owner, strategy in instances.items():
                for timer in self._timers(strategy.contract.all_subscriptions, calendar):
                    timer_owners.setdefault(timer, []).append(owner)
            timers = tuple(sorted(timer_owners, key=lambda timer: (timer.time, timer.name)))
            if any(timer.time.microsecond for timer in timers):
                raise ValueError("Scheduled events currently require whole-second resolution.")
            timer_owners = {timer: tuple(owners) for timer, owners in timer_owners.items()}
            intraday_timers = any(session.open < timer.time < session.close for timer in timers for session in calendar.sessions)
            execution_size = "1 day"
            intraday_sizes = [parse_bar_size(size) for size in bar_sizes if parse_bar_size(size).unit == "second"]
            if intraday_sizes:
                from math import gcd
                from functools import reduce
                source_seconds = reduce(gcd, (parse_bar_size(size.source).count for size in intraday_sizes))
                execution_size = f"{source_seconds} secs"
            if intraday_timers:
                execution_size = "1 min" if execution_size == "1 day" else execution_size
                if any(timer.time.second or timer.time.microsecond for timer in timers):
                    execution_size = "1 sec"
            instruments = {}
            for strategy in instances.values():
                for instrument in strategy.contract.resolved_instruments:
                    if instrument.symbol in instruments and instruments[instrument.symbol] != instrument:
                        raise ValueError(f"Conflicting instrument definitions for {instrument.symbol}.")
                    instruments[instrument.symbol] = instrument
            for subscription in bar_subscriptions:
                for symbol in subscription.symbols:
                    instruments.setdefault(symbol, Instrument(symbol))
            model = ExecutionModel.preset(execution) if isinstance(execution, str) else execution
            if not isinstance(model, ExecutionModel):
                raise TypeError("execution must be an ExecutionModel or a named execution preset.")
            run_metadata: dict[str, Any] = dict(metadata or {})
            research_metadata: dict[str, Any] = dict(
                start=self.start, end=self.end,
                contracts={owner: serialize(strategy.contract) for owner, strategy in instances.items()},
                strategy_metadata={owner: serialize(strategy.metadata) for owner, strategy in instances.items()})
            run_metadata["research"] = research_metadata
            if not instruments:
                replay = Backtest(data=None,
                    strategies={owner: (lambda strategy=strategy: strategy) for owner, strategy in instances.items()},
                    execution=model, initial_cash=initial_cash, base_currency=base_currency, fx_rates=fx_rates, limits=limits, instruments=(), calendar=calendar,
                    timers=timers, timer_owners=timer_owners, start=first, end=last, metadata=run_metadata,
                    publish_bars=False)
                result = replay.run()
                if output is not None:
                    result.export(output)
                return result.raise_for_failure()
            warmup_sessions = max(strategy.contract.warmup_sessions for strategy in instances.values())
            # One completed session supplies a prior mark for startup order risk.
            # It is not strategy warmup and is never exposed as undeclared history.
            history_start = self._calendar_start(calendar_name, calendar.sessions[0].label, max(1, warmup_sessions))
            request = HistoryRequest(tuple(instruments), history_start, self.end,
                                     bar_size=execution_size, calendar=calendar_name)
            if provider is None:
                provider = FallbackHistoricalProvider(cache=cache)
            if cache is not None and not isinstance(provider, (CachedHistoricalProvider, FallbackHistoricalProvider)):
                provider = CachedHistoricalProvider(provider, cache)
            dataset = await provider.fetch(request, refresh=refresh)
            if {bar.symbol for bar in dataset.data.bars} != set(request.symbols):
                raise ValueError("Historical provider returned a different instrument universe.")
            expected = SessionCalendar.from_exchange(request.calendar, request.start, request.end)
            if dataset.calendar.sessions != expected.sessions:
                raise ValueError("Historical provider's session calendar differs from the requested exchange sessions.")
            validate_coverage(dataset.data, expected, request)
            resampled = {}
            for size in bar_sizes:
                resampled[size] = (dataset.data if size == parse_bar_size(execution_size).label else
                                   dataset.data.resample(size, expected, boundaries=(first,)))
            observations_by_owner = {}
            warmup = {}
            for owner, strategy in instances.items():
                panels = {}
                count = strategy.contract.warmup_sessions
                cutoff = expected.sessions[-len(calendar.sessions) - count].open if count else first
                prior = {}
                for subscription in strategy.contract.all_subscriptions:
                    if not isinstance(subscription, BarSubscription):
                        continue
                    stream = resampled[parse_bar_size(subscription.bar_size).label]
                    for bar in stream.bars:
                        if bar.symbol not in subscription.symbols:
                            continue
                        if subscription in strategy.contract.subscriptions and cutoff <= bar.start and bar.end <= first:
                            prior[(bar.symbol, bar.end)] = bar
                        elif first <= bar.start and bar.end <= last:
                            panels.setdefault(bar.end, {})[bar.symbol] = bar
                if panels:
                    observations_by_owner[owner] = tuple(Bars(instant, bars) for instant, bars in sorted(panels.items()))
                warmup[owner] = tuple(prior.values())
            research_metadata.update(
                fetched=serialize(request),
                history=dataset.manifest, cache_path=str(dataset.cache_path),
                execution_bar_size=execution_size, observation_bar_sizes=sorted(bar_sizes),
                execution_costs="Explicit modeling assumptions, not guaranteed broker fees or fills.",
                returns="Price-return simulation; dividends and corporate-action cash flows are not credited.",
                universe="Strategy-declared fixed universe; delistings and survivorship bias are not reconstructed.")
            replay = Backtest(data=dataset.data,
                              strategies={owner: (lambda strategy=strategy: strategy) for owner, strategy in instances.items()},
                              execution=model, initial_cash=initial_cash, base_currency=base_currency, fx_rates=fx_rates, limits=limits,
                              instruments=tuple(instruments.values()), calendar=dataset.calendar, timers=timers,
                              start=first, end=last, metadata=run_metadata,
                              publish_bars=False, preload_marks=True, strategy_warmup=warmup,
                              observations_by_owner=observations_by_owner, timer_owners=timer_owners)
            result = replay.run()
            if output is not None:
                self._export(dataset, result, Path(output))
            return result.raise_for_failure()
        finally:
            self._lock.release()

    @staticmethod
    def _export(dataset: HistoricalDataset, result: BacktestResult, destination: Path) -> None:
        result.export(destination)
        fields = ("symbol", "start", "end", "open", "high", "low", "close", "volume")
        with (destination / "bars.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for bar in dataset.data.bars:
                writer.writerow({key: serialize(getattr(bar, key)) for key in fields})
        with (destination / "sessions.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["session", "open", "close"])
            writer.writeheader()
            for session in dataset.calendar.sessions:
                writer.writerow(dict(session=session.label, open=session.open.isoformat(), close=session.close.isoformat()))
        manifest = dict(dataset.manifest)
        manifest.update(data_identity=dataset.data.identity, price_basis=dataset.data.price_basis,
                        source=dataset.data.source, calendar=serialize(dataset.calendar))
        (destination / "history_manifest.json").write_text(json.dumps(serialize(manifest), indent=2, sort_keys=True) + "\n")
