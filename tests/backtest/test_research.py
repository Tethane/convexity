"""Research discovers event needs from strategies; fixtures never replace a real source."""

import asyncio
from collections.abc import Callable, Sequence
from datetime import date, timedelta
from decimal import Decimal
import inspect
import json
from pathlib import Path

import pytest

from convexity.backtest.bar_size import intervals
from convexity.backtest.data import HistoricalData, SessionCalendar
from convexity.backtest.exchange import ExecutionModel
from convexity.backtest.research import Research
from convexity.backtest.runner import BacktestFailure
from convexity.runtime import (
    Bar,
    Bars,
    Strategy,
    SubmitOrder,
    Timer,
    FillEvent,
    OrderUpdate,
    StrategyMetadata,
    StrategyContract,
    BarSubscription,
    ClockSubscription,
    EventSubscription,
)
from convexity.runtime.calendar_schedule import SessionSchedule


class FixtureProvider:
    def __init__(self):
        self.calls = []

    async def fetch(self, request, *, refresh=False):
        from convexity.backtest.history import HistoricalDataset

        self.calls.append((request, refresh))
        calendar = SessionCalendar.from_exchange(
            request.calendar, request.start, request.end
        )
        bars = tuple(
            Bar(
                symbol,
                start,
                end,
                Decimal(100),
                Decimal(102),
                Decimal(99),
                Decimal(101),
                100_000,
            )
            for start, end in intervals(calendar.sessions, request.bar_size)
            for symbol in request.symbols
        )
        data = HistoricalData(bars, source="explicit synthetic TEST FIXTURE")
        return HistoricalDataset(
            data=data,
            calendar=calendar,
            manifest={"source": data.source, "request": str(request)},
            cache_path=Path("/test-fixture"),
            cache_hit=len(self.calls) > 1,
        )


class BuyAtStart(Strategy):
    metadata = StrategyMetadata("buy-hold")
    contract = StrategyContract(("SPY",))

    def on_start(self, context):
        assert not context.history
        assert context.mark_times["SPY"] < context.time
        return SubmitOrder.market("SPY", "BUY", 1, client_order_id="entry")


class WarmBuy(BuyAtStart):
    contract = StrategyContract(
        ("SPY",), (BarSubscription(("SPY",)),), warmup_sessions=2
    )

    def on_start(self, context):
        assert len(context.history["SPY"]) == 2
        assert all(bar.end < context.time for bar in context.history["SPY"])
        return SubmitOrder.market("SPY", "BUY", 1)


def run(
    strategy: Callable[[], Strategy] | Sequence[Callable[[], Strategy]] = BuyAtStart,
    provider=None,
    **options,
):
    return Research("2024-01-08", "2024-01-10").run(
        strategy, provider=provider or FixtureProvider(), cache=None, **options
    )


def test_constructor_only_describes_period_and_strategy_owns_identity():
    assert set(inspect.signature(Research).parameters) == {"start", "end"}
    result = run()
    assert result.status == "completed"
    assert set(result.positions) == {"buy-hold"}
    assert result.positions["buy-hold"]["SPY"].quantity == 1
    assert len(result.fills) == 1
    assert result.fills[0].occurred_at.date() == date(2024, 1, 8)
    assert not any(isinstance(event, Bars) for event in result.events)


def test_strategy_warmup_then_executes_and_exports(tmp_path):
    provider = FixtureProvider()
    result = run(WarmBuy, provider, output=tmp_path)
    request, refresh = provider.calls[0]
    assert request.start == date(2024, 1, 4)
    assert request.end == date(2024, 1, 10)
    assert refresh is False
    assert result.fills[0].fee == 1
    events = [event for event in result.events if isinstance(event, Bars)]
    assert len(events) == 3
    exported = json.loads((tmp_path / "result.json").read_text())
    manifest = json.loads((tmp_path / "history_manifest.json").read_text())
    assert manifest["data_identity"] == exported["config"]["data_identity"]
    restored = HistoricalData.from_csv(
        tmp_path / "bars.csv", price_basis=manifest["price_basis"]
    )
    assert restored.identity == manifest["data_identity"]
    assert len(restored.bars) == 5
    assert (
        SessionCalendar.from_csv(tmp_path / "sessions.csv").sessions
        == result.config["calendar"].sessions
    )


def test_repeated_runs_are_identical_and_refresh_is_forwarded():
    provider = FixtureProvider()
    work = Research("2024-01-08", "2024-01-10")
    first = work.run(WarmBuy, provider=provider, cache=None)
    second = work.run(WarmBuy, provider=provider, cache=None, refresh=True)
    assert first.to_dict() == second.to_dict()
    assert provider.calls[-1][1] is True


def test_async_notebook_entrypoint_and_sync_error_without_coroutine_leak():
    async def execute():
        work = Research("2024-01-08", "2024-01-10")
        with pytest.raises(RuntimeError, match="await research.run_async"):
            work.run(BuyAtStart)
        result = await work.run_async(
            BuyAtStart, provider=FixtureProvider(), cache=None
        )
        assert result.status == "completed"

    asyncio.run(execute())


def test_strategy_failure_raises_with_partial_result_and_exports(tmp_path):
    class Broken(WarmBuy):
        def on_bar(self, event, context):
            raise ValueError("bad signal")

    with pytest.raises(BacktestFailure, match="bad signal") as failure:
        run(Broken, output=tmp_path)
    assert failure.value.result.status == "failed"
    assert json.loads((tmp_path / "result.json").read_text())["status"] == "failed"


def test_reusing_strategy_instance_fails_across_research_runs():
    instances = (BuyAtStart(), BuyAtStart())
    work = Research("2024-01-08", "2024-01-10")
    for instance in instances:
        work.run(lambda: instance, provider=FixtureProvider(), cache=None)
    with pytest.raises(ValueError, match="reused a previous instance"):
        work.run(lambda: instances[0], provider=FixtureProvider(), cache=None)


def test_provider_failure_never_falls_back_to_prices_or_execution():
    class Unavailable:
        async def fetch(self, request, *, refresh=False):
            raise RuntimeError("no market data permission")

    with pytest.raises(RuntimeError, match="no market data permission"):
        run(provider=Unavailable())


def test_monthly_schedule_uses_real_first_session_of_month():
    class Monthly(Strategy):
        metadata = StrategyMetadata("monthly")
        contract = StrategyContract(
            ("SPY",),
            (
                SessionSchedule(
                    "monthly", "rebalance", start=date(2024, 1, 8), frequency="monthly"
                ),
            ),
        )

    result = Research("2024-01-08", "2024-02-02").run(
        Monthly, provider=FixtureProvider(), cache=None
    )
    timers = [event for event in result.events if isinstance(event, Timer)]
    assert [timer.time.date() for timer in timers] == [date(2024, 2, 1)]


def test_zero_costs_require_explicit_model_and_are_honored():
    result = run(execution=ExecutionModel())
    assert result.fills[0].price == 100
    assert result.fills[0].fee == 0


def test_undeclared_order_is_failure_even_when_another_strategy_declares_symbol():
    class Bad(BuyAtStart):
        def on_start(self, context):
            return SubmitOrder.market("QQQ", "BUY", 1)

    class Other(Strategy):
        metadata = StrategyMetadata("other")
        contract = StrategyContract(("QQQ",))

    with pytest.raises(
        BacktestFailure, match="outside the strategy contract"
    ) as failure:
        run((Bad, Other))
    assert not failure.value.result.orders


def test_clock_only_strategy_trades_without_receiving_bars_or_history():
    seen = []

    class Overnight(Strategy):
        metadata = StrategyMetadata("overnight")
        contract = StrategyContract(
            ("SPY",),
            (
                ClockSubscription("buy", "15:50"),
                ClockSubscription("sell", "09:30"),
                EventSubscription(FillEvent),
            ),
        )

        def on_timer(self, event, context):
            assert not context.history
            seen.append(event)
            if event.name == "buy":
                return SubmitOrder.market("SPY", "BUY", 1)
            if context.positions.get("SPY") and context.positions["SPY"].quantity:
                return SubmitOrder.market("SPY", "SELL", 1)

        def on_fill(self, event, context):
            assert event.fill.available_at > event.fill.occurred_at

    provider = FixtureProvider()
    result = run(Overnight, provider, execution="frictionless")
    assert provider.calls[0][0].bar_size == "1 min"
    assert len(seen) == 6
    assert len(result.fills) == 5
    assert not any(isinstance(event, Bars) for event in result.events)
    assert all(fill.occurred_at.minute in (50, 30) for fill in result.fills)


def test_each_strategy_sees_its_own_warmup_and_observation_symbols():
    class NoWarmup(WarmBuy):
        metadata = StrategyMetadata("cold")
        contract = StrategyContract(("QQQ",), (BarSubscription(("QQQ",)),))

        def on_start(self, context):
            assert context.history == {"QQQ": ()}

        def on_bar(self, event, context):
            assert set(event.bars) == {"QQQ"}
            assert set(context.history) == {"QQQ"}

    result = run((WarmBuy, NoWarmup))
    assert result.status == "completed"


def test_contract_and_metadata_required_and_duplicate_names_rejected():
    with pytest.raises(ValueError, match="metadata"):
        run(Strategy)

    class NoContract(Strategy):
        metadata = StrategyMetadata("missing")

    with pytest.raises(ValueError, match="contract"):
        run(NoContract)
    with pytest.raises(ValueError, match="unique"):
        run((BuyAtStart, BuyAtStart))


def test_named_clocks_are_scoped_to_owners_even_when_names_match():
    seen = []

    class Morning(Strategy):
        metadata = StrategyMetadata("morning")
        contract = StrategyContract((), (ClockSubscription("trade", "09:30"),))

        def on_timer(self, event, context):
            seen.append((context.strategy_id, event.time.hour, event.time.minute))

    class Afternoon(Morning):
        metadata = StrategyMetadata("afternoon")
        contract = StrategyContract((), (ClockSubscription("trade", "15:50"),))

    class ForbiddenProvider:
        async def fetch(self, *args, **kwargs):
            raise AssertionError("Pure clocks need no historical prices")

    result = run((Morning, Afternoon), ForbiddenProvider())
    assert len(seen) == 6
    assert all(
        (hour, minute) == ((14, 30) if owner == "morning" else (20, 50))
        for owner, hour, minute in seen
    )
    assert result.metrics["final_cash"] == 100000
    assert result.config["data_source"] == "event-only"


def test_mixed_resolutions_publish_only_owner_requested_events_and_history():
    seen = {"daily": [], "minute": [], "mixed": []}

    class Daily(Strategy):
        metadata = StrategyMetadata("daily")
        contract = StrategyContract(("SPY",), (BarSubscription(("SPY",)),))

        def on_bar(self, event, context):
            seen[context.strategy_id].append(event)
            assert len(context.history["SPY"]) == len(seen[context.strategy_id])

    class Minute(Daily):
        metadata = StrategyMetadata("minute")
        contract = StrategyContract(("SPY",), (BarSubscription(("SPY",), "1 min"),))

    class Mixed(Strategy):
        metadata = StrategyMetadata("mixed")
        contract = StrategyContract(
            ("SPY", "QQQ"),
            (BarSubscription(("SPY",), "1 min"), BarSubscription(("QQQ",), "1 day")),
        )

        def on_bar(self, event, context):
            seen["mixed"].append(event)
            assert all(
                bar.end <= context.time
                for stream in context.history.values()
                for bar in stream
            )
            if "QQQ" in event.bars:
                assert event.bars["QQQ"].start < event.bars["SPY"].start

    result = run((Daily, Minute, Mixed))
    assert result.status == "completed"
    assert len(seen["daily"]) == 3
    assert len(seen["minute"]) == len(seen["mixed"]) == 390 * 3
    assert all(
        event.bars["SPY"].end - event.bars["SPY"].start == timedelta(minutes=1)
        for event in seen["minute"]
    )


def test_early_close_does_not_invent_afternoon_clock_events():
    class Clock(Strategy):
        metadata = StrategyMetadata("clock")
        contract = StrategyContract((), (ClockSubscription("late", "15:50"),))

    result = Research("2024-11-27", "2024-11-29").run(
        Clock, provider=FixtureProvider(), cache=None
    )
    assert [
        event.time.date() for event in result.events if isinstance(event, Timer)
    ] == [date(2024, 11, 27)]


def test_unavailable_event_source_fails_before_history_fetch():
    class BookUpdate:
        pass

    class Microstructure(Strategy):
        metadata = StrategyMetadata("microstructure")
        contract = StrategyContract(("SPY",), (EventSubscription(BookUpdate),))

    provider = FixtureProvider()
    with pytest.raises(NotImplementedError, match="no event source for: BookUpdate"):
        run(Microstructure, provider)
    assert provider.calls == []


def test_clock_timezone_uses_local_day_when_session_crosses_midnight():
    class Tokyo(Strategy):
        metadata = StrategyMetadata("tokyo")
        contract = StrategyContract(
            (), (ClockSubscription("morning", "05:50", timezone="Asia/Tokyo"),)
        )

    result = run(Tokyo)
    events = [event for event in result.events if isinstance(event, Timer)]
    assert len(events) == 3
    assert all((event.time.hour, event.time.minute) == (20, 50) for event in events)


def test_exported_history_local_replay_preserves_declared_owner_warmup(tmp_path):
    from convexity.backtest.runner import Backtest

    histories = []

    class Warm(Strategy):
        metadata = StrategyMetadata("warm")
        contract = StrategyContract(
            ("SPY",), (BarSubscription(("SPY",)),), warmup_sessions=2
        )

        def on_start(self, context):
            histories.append((context.strategy_id, tuple(context.history["SPY"])))
            return SubmitOrder.market("SPY", "BUY", 1)

    class Cold(Strategy):
        metadata = StrategyMetadata("cold")
        contract = StrategyContract(("SPY",), (BarSubscription(("SPY",)),))

        def on_start(self, context):
            histories.append((context.strategy_id, tuple(context.history["SPY"])))

    result = run((Warm, Cold), output=tmp_path, execution="frictionless")
    expected_histories = list(histories)
    histories.clear()
    data = HistoricalData.from_csv(tmp_path / "bars.csv")
    calendar = SessionCalendar.from_csv(tmp_path / "sessions.csv")
    replay = Backtest(
        data=data,
        strategies={"warm": Warm, "cold": Cold},
        calendar=calendar,
        start=result.config["start"],
        end=result.config["end"],
        warmup=True,
    ).run()
    assert replay.status == "completed"
    assert histories == expected_histories
    assert len(histories[0][1]) == 2
    assert histories[1][1] == ()
    assert replay.fills == result.fills
    assert replay.metrics == result.metrics
