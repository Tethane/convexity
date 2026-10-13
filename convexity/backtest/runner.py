"""Fresh deterministic replay state, inspectable results, JSON/CSV export."""
from bisect import bisect_right
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, date
from decimal import Decimal
from pathlib import Path
from collections.abc import Callable, Mapping
import csv
import json
import hashlib
import inspect
import weakref
from typing import Any

from convexity.runtime.account import AccountState, FXRate
from convexity.runtime import Bars, Event, Fill, Instrument, Order, Position, RiskLimits, Runtime, Strategy, Timer
from convexity.runtime.types import Bar, aware, decimal
from convexity.runtime.contracts import strategy_contract
from .data import HistoricalData, SessionCalendar
from .exchange import ExecutionModel, SimulatedExchange


def serialize(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: serialize(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return {str(k): serialize(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [serialize(v) for v in value]
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    if isinstance(value, (Decimal, datetime, date)):
        return str(value)
    return value


@dataclass(frozen=True)
class BacktestResult:
    status: str
    diagnostics: tuple[str, ...]
    config: dict
    orders: tuple[Order, ...]
    fills: tuple[Fill, ...]
    positions: dict[str, dict[str, Position]]
    equity: tuple[dict, ...]
    cash_flows: tuple[dict, ...]
    events: tuple[Event, ...]
    metrics: dict[str, Decimal]
    account: AccountState | None = None

    @property
    def open_orders(self) -> tuple[Order, ...]:
        return tuple(o for o in self.orders if not o.terminal)

    def to_dict(self) -> dict:
        result = serialize(self)
        result["open_orders"] = serialize(self.open_orders)
        return result

    def raise_for_failure(self) -> "BacktestResult":
        if self.status != "completed":
            raise BacktestFailure(self)
        return self

    def export(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "result.json").write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")
        for name in ("fills", "equity", "cash_flows"):
            rows = serialize(getattr(self, name))
            with (directory / f"{name}.csv").open("w", newline="") as stream:
                if rows:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
        return directory


class BacktestFailure(RuntimeError):
    """A replay failed; its partial, inspectable result is retained."""

    def __init__(self, result: BacktestResult):
        self.result = result
        super().__init__("; ".join(result.diagnostics) or "Backtest failed.")


def _source_identity(strategy: Strategy) -> str | None:
    try:
        source = inspect.getsource(type(strategy))
    except (OSError, TypeError):
        return None
    return hashlib.sha256(source.encode()).hexdigest()


class Backtest:
    def __init__(self, *, data: HistoricalData | None, strategies: Mapping[str, Callable[[], Strategy]],
                 execution: ExecutionModel | None = None, initial_cash: Decimal | int | str | Mapping[str, Decimal] = 100_000,
                 base_currency: str = "USD", fx_rates: tuple[FXRate, ...] = (),
                 limits: RiskLimits | None = None, instruments: tuple[Instrument, ...] | None = None,
                 calendar: SessionCalendar | None = None, timers: tuple[Timer, ...] = (),
                 start: datetime | None = None, end: datetime | None = None,
                 metadata: Mapping | None = None, warmup: bool = False,
                 observations: HistoricalData | None = None, publish_bars: bool = True,
                 preload_marks: bool = False,
                 strategy_warmup: Mapping[str, tuple[Bar, ...]] | None = None,
                 observations_by_owner: Mapping[str, tuple[Bars, ...]] | None = None,
                 timer_owners: Mapping[Timer, tuple[str, ...]] | None = None):
        self.data = data
        self.observations = observations
        self.observations_by_owner = dict(observations_by_owner or {})
        self.timer_owners = timer_owners
        data_bars = data.bars if data is not None else ()
        self.publish_bars = publish_bars
        self.preload_marks = preload_marks
        self.strategy_warmup = dict(strategy_warmup or {})
        if observations is not None:
            if data is None:
                raise ValueError("Observations require execution data.")
            if observations.price_basis != data.price_basis:
                raise ValueError("Observation and execution price bases must match.")
            sources = {symbol: [bar for bar in data.bars if bar.symbol == symbol]
                       for symbol in {bar.symbol for bar in data.bars}}
            offsets = dict.fromkeys(sources, 0)
            for observed in observations.bars:
                source = sources.get(observed.symbol, [])
                offset = offsets.get(observed.symbol, 0)
                stop = offset
                while stop < len(source) and source[stop].start < observed.end:
                    stop += 1
                group = source[offset:stop]
                if not group:
                    raise ValueError("Observations must aggregate the execution dataset.")
                aggregate = Bar(observed.symbol, group[0].start, group[-1].end, group[0].open,
                                max(bar.high for bar in group), min(bar.low for bar in group),
                                group[-1].close, sum(bar.volume for bar in group))
                if aggregate != observed:
                    raise ValueError("Observations must aggregate the execution dataset exactly.")
                offsets[observed.symbol] = stop
            if any(offsets[symbol] != len(source) for symbol, source in sources.items()):
                raise ValueError("Observations must cover the execution dataset.")
        # Verify every strategy observation against completed execution prices.
        # The owner streams may differ in cadence and symbol set, but may never
        # invent OHLCV values or publish future execution information.
        if self.observations_by_owner:
            sources = {}
            for bar in data_bars:
                sources.setdefault(bar.symbol, []).append(bar)
            source_starts = {symbol: [bar.start for bar in stream] for symbol, stream in sources.items()}
            for owner, owner_panels in self.observations_by_owner.items():
                if owner not in strategies:
                    raise ValueError("Observation owners must be registered strategies.")
                previous = {}
                for panel in owner_panels:
                    for symbol, observed in panel.bars.items():
                        source = sources.get(symbol, ())
                        offset = bisect_right(source_starts.get(symbol, ()), observed.start) - 1
                        if offset < 0:
                            raise ValueError("Owner observations must aggregate execution data.")
                        stop = offset
                        while stop < len(source) and source[stop].start < observed.end:
                            stop += 1
                        group = source[offset:stop]
                        if not group or (symbol in previous and observed.start < previous[symbol]):
                            raise ValueError("Owner observations cannot overlap or omit execution data.")
                        aggregate = Bar(symbol, group[0].start, group[-1].end, group[0].open,
                                        max(bar.high for bar in group), min(bar.low for bar in group),
                                        group[-1].close, sum(bar.volume for bar in group))
                        if observed != aggregate:
                            raise ValueError("Owner observations must aggregate execution data exactly.")
                        previous[symbol] = observed.end
        self.strategies = dict(strategies)
        if not self.strategies or not all(callable(f) for f in self.strategies.values()):
            raise ValueError("Pass named strategy factories; every run needs fresh state.")
        self.execution = execution or ExecutionModel()
        self.initial_cash = dict(initial_cash) if isinstance(initial_cash, Mapping) else decimal(initial_cash)
        self.base_currency, self.fx_rates = base_currency, tuple(fx_rates)
        self.limits = limits or RiskLimits()
        self.instruments = instruments or tuple(Instrument(s) for s in sorted({b.symbol for b in data_bars}))
        if len({i.symbol for i in self.instruments}) != len(self.instruments) or {i.symbol for i in self.instruments} != {b.symbol for b in data_bars}:
            raise ValueError("Instrument identities must match the dataset exactly.")
        self.calendar = calendar
        self.timers = tuple(sorted(set(timers), key=lambda t: (t.time, t.name)))
        first = start or min((b.start for b in data_bars), default=None)
        last = end or max((b.end for b in data_bars), default=None)
        if first is None or last is None:
            raise ValueError("Event-only replay requires explicit start and end.")
        self.start: datetime = first
        self.end: datetime = last
        aware(self.start)
        aware(self.end)
        if self.start >= self.end or (data_bars and (self.start < min(b.start for b in data_bars) or self.end > max(b.end for b in data_bars))):
            raise ValueError("Requested interval must be covered by the input dataset.")
        if observations is not None and any(bar.start < boundary < bar.end
                                           for bar in observations.bars for boundary in (self.start, self.end)):
            raise ValueError("Start/end must not split a strategy observation.")
        self.metadata = dict(metadata or {})
        if type(warmup) is not bool:
            raise ValueError("warmup must be a bool.")
        self.warmup = warmup
        self._previous_instances: tuple[Strategy, ...] = ()
        self._instance_refs: list[weakref.ReferenceType[Strategy]] = []
        # Panels are aligned and non-overlapping (HistoricalData validates this).
        intervals = sorted({(b.start, b.end) for b in data_bars})
        starts = [start for start, _ in intervals]
        for time in [self.start, self.end] + [t.time for t in self.timers]:
            index = bisect_right(starts, time) - 1
            if index >= 0 and intervals[index][0] < time < intervals[index][1]:
                raise ValueError("Timers/start/end must fall on bar boundaries or between bars; intrabar scheduling requires finer data.")
        if calendar is not None and data is not None:
            opens = [s.open for s in calendar.sessions]
            panels: dict[int, list[tuple[datetime, datetime]]] = {}
            for begin, finish in intervals:
                index = bisect_right(opens, begin) - 1
                if index < 0 or finish > calendar.sessions[index].close:
                    raise ValueError(f"Bar at {begin} falls outside the supplied sessions.")
                panels.setdefault(index, []).append((begin, finish))
            for index, session in enumerate(calendar.sessions):
                if session.open >= self.start and session.close <= self.end:
                    panel = panels.get(index, [])
                    if not panel or panel[0][0] != session.open or panel[-1][1] != session.close:
                        raise ValueError(f"Missing session coverage for {session.label}.")
                    if any(a[1] != b[0] for a, b in zip(panel, panel[1:])):
                        raise ValueError(f"Intraday data gap in session {session.label}.")

    def run(self) -> BacktestResult:
        strategies = {name: factory() for name, factory in self.strategies.items()}
        if len({id(s) for s in strategies.values()}) != len(strategies):
            raise ValueError("Strategy factories must return distinct instances.")
        self._instance_refs = [reference for reference in self._instance_refs if reference() is not None]
        if any(s is reference() for s in strategies.values() for reference in self._instance_refs):
            raise ValueError("Strategy factory reused a previous instance; return fresh strategy state.")
        self._previous_instances = tuple(strategies.values())
        self._instance_refs.extend(weakref.ref(strategy) for strategy in strategies.values())
        instruments = {i.symbol: i for i in self.instruments}
        exchange = SimulatedExchange(instruments, self.execution)
        closes = tuple(s.close for s in self.calendar.sessions) if self.calendar else ()
        runtime = Runtime(strategies, exchange, instruments, self.initial_cash, self.limits, closes,
                          base_currency=self.base_currency, fx_rates=self.fx_rates)
        if self.preload_marks and self.data is not None:
            for bar in self.data.bars:
                if bar.end <= self.start:
                    runtime.marks[bar.symbol] = bar.close
                    runtime.mark_times[bar.symbol] = bar.end
        for owner, bars in self.strategy_warmup.items():
            runtime.warmup_history(owner, bars)
        if self.warmup and self.data is not None:
            # Legacy strategies share completed observations. Declared strategies
            # receive only their own requested sessions and observed symbols.
            source = self.observations or self.data
            for bar in source.bars:
                if bar.end <= self.start:
                    runtime.history[bar.symbol].append(bar)
            for bar in self.data.bars:
                if bar.end <= self.start:
                    runtime.marks[bar.symbol] = bar.close
                    runtime.mark_times[bar.symbol] = bar.end
            for owner, strategy in strategies.items():
                contract = strategy_contract(strategy)
                if contract is None or owner in self.strategy_warmup:
                    continue
                requested = contract.warmup_sessions
                if not requested or not contract.observed_symbols:
                    runtime.warmup_history(owner, ())
                    continue
                if self.calendar is None:
                    raise ValueError("Declared strategy warmup_sessions requires a session calendar.")
                completed = [session for session in self.calendar.sessions if session.close <= self.start]
                if len(completed) < requested:
                    raise ValueError(f"Not enough preceding sessions for {owner}'s declared warmup.")
                cutoff = completed[-requested].open
                owner_source = (tuple(bar for panel in self.observations_by_owner[owner] for bar in panel.bars.values())
                                if owner in self.observations_by_owner else source.bars)
                runtime.warmup_history(owner, tuple(bar for bar in owner_source
                    if bar.symbol in contract.observed_symbols and cutoff <= bar.start and bar.end <= self.start))
        panels = {p.time: p for p in (self.data.panels() if self.data is not None else ()) if self.start <= next(iter(p.bars.values())).start and p.time <= self.end}
        observations = {p.time: p for p in self.observations.panels()} if self.observations is not None else {}
        owner_observations = {}
        for owner, stream in self.observations_by_owner.items():
            for panel in stream:
                if self.start <= panel.time <= self.end:
                    owner_observations.setdefault(panel.time, {})[owner] = panel
        timer_map = {}
        for timer in self.timers:
            if self.start <= timer.time <= self.end:
                timer_map.setdefault(timer.time, []).append(timer)
        times = sorted(set(panels) | set(timer_map) | {self.end} | {t for t in closes if self.start <= t <= self.end})
        if not panels and self.data is not None:
            raise ValueError("Requested interval contains no complete bars.")
        try:
            runtime.start(self.start)
            for time in times:
                runtime.advance(time, panels.get(time), tuple(timer_map.get(time, ())),
                                publish_bars=self.publish_bars and self.observations is None, observation=observations.get(time),
                                observations_by_owner=owner_observations.get(time), timer_owners=self.timer_owners)
            runtime.stop()
        except Exception as exc:
            runtime.diagnostics.append(f"Replay stopped: {type(exc).__name__}: {exc}")
            runtime.record_equity()
        peak = runtime.initial_cash
        drawdown = Decimal(0)
        for point in runtime.equity_history:
            peak = max(peak, point["equity"])
            drawdown = max(drawdown, (peak - point["equity"]) / peak)
        config = dict(base_currency=self.base_currency, fx_rates=self.fx_rates,
                      cash_policy="native currency ledger; negative balances are currency loans; no automatic FX conversion",
                      margin_model="gross leverage; no broker-specific margin or liquidation model", engine_version="event-runtime-v3", initial_cash=self.initial_cash, start=self.start, end=self.end,
                      execution=self.execution, limits=self.limits, instruments=self.instruments,
                      data_identity=self.data.identity if self.data is not None else None,
                      data_source=self.data.source if self.data is not None else "event-only",
                      observation_identity=self.observations.identity if self.observations is not None else (self.data.identity if self.data is not None else None),
                      observation_identities={owner: hashlib.sha256(json.dumps(serialize(stream), sort_keys=True).encode()).hexdigest()
                                              for owner, stream in self.observations_by_owner.items()},
                      price_basis=self.data.price_basis if self.data is not None else None, calendar=self.calendar, timers=self.timers,
                      strategies={k: f"{type(s).__module__}.{type(s).__qualname__}" for k, s in strategies.items()},
                      strategy_config={k: serialize(getattr(s, "config", None)) for k, s in strategies.items()},
                      strategy_source_sha256={k: _source_identity(s) for k, s in strategies.items()},
                      warmup=self.warmup,
                      metadata=self.metadata,
                      ordering="borrow; FIFO fills; expiry; bars; named timers; serial callbacks; equity",
                      assumptions="Whole-share stocks; fees in instrument currency; aligned bars; fixed universe; price returns only; no dividend/corporate-action cash flows; no implicit liquidation")
        return BacktestResult("failed" if runtime.diagnostics else "completed", tuple(runtime.diagnostics),
                              config, tuple(runtime.orders.values()), tuple(runtime.fills),
                              {owner: {s: p for (o, s), p in runtime.positions.items() if o == owner} for owner in strategies},
                              tuple(runtime.equity_history), tuple(runtime.cash_flows), tuple(runtime.events),
                              dict(final_cash=runtime.cash, final_equity=runtime.equity,
                                   total_return=runtime.equity / runtime.initial_cash - 1,
                                   max_drawdown=drawdown, fees=runtime.fees, borrow_cost=runtime.borrow_cost),
                              account=runtime.account_state())
