"""Independent regressions for the strategy-owned research interface."""

from pathlib import Path
from decimal import Decimal

import pytest

from convexity.backtest.bar_size import intervals
from convexity.backtest.data import HistoricalData, SessionCalendar
from convexity.backtest.history import HistoricalDataset
from convexity.backtest.research import Research
from convexity.backtest.runner import BacktestFailure
from convexity.runtime import (
    Bar,
    BarSubscription,
    ClockSubscription,
    Strategy,
    StrategyContract,
    StrategyMetadata,
    SubmitOrder,
)


class ReviewProvider:
    async def fetch(self, request, *, refresh=False):
        calendar = SessionCalendar.from_exchange(
            request.calendar, request.start, request.end
        )
        bars = tuple(
            Bar(
                symbol,
                start,
                end,
                Decimal(100),
                Decimal(100),
                Decimal(100),
                Decimal(100),
                1000,
            )
            for start, end in intervals(calendar.sessions, request.bar_size)
            for symbol in request.symbols
        )
        return HistoricalDataset(
            HistoricalData(bars, source="review-fixture"),
            calendar,
            {"source": "review-fixture"},
            Path("/fixture"),
            False,
        )


def run(*strategies):
    return Research("2024-01-08", "2024-01-09").run(
        strategies, provider=ReviewProvider(), cache=None
    )


def test_changing_contract_during_callback_cannot_expand_authorization():
    class ChangesContract(Strategy):
        metadata = StrategyMetadata("changes")
        contract = StrategyContract(("SPY",))

        def on_start(self, context):
            self.contract = StrategyContract(("SPY", "QQQ"))
            return SubmitOrder.market("QQQ", "BUY", 1)

    class OtherOwner(Strategy):
        metadata = StrategyMetadata("other")
        contract = StrategyContract(("QQQ",))

    with pytest.raises(BacktestFailure) as failure:
        run(ChangesContract, OtherOwner)
    assert not failure.value.result.orders


def test_failed_command_batch_never_submits_its_valid_prefix():
    class BadBatch(Strategy):
        metadata = StrategyMetadata("bad-batch")
        contract = StrategyContract(("SPY",))

        def on_start(self, context):
            yield SubmitOrder.market("SPY", "BUY", 1)
            yield SubmitOrder.market("QQQ", "BUY", 1)

    with pytest.raises(
        BacktestFailure, match="outside the strategy contract"
    ) as failure:
        run(BadBatch)
    assert not failure.value.result.orders


def test_same_named_clocks_are_scoped_to_each_owners_declared_time():
    observed = {"early": [], "late": []}

    def factory(name, at):
        class Clock(Strategy):
            metadata = StrategyMetadata(name)
            contract = StrategyContract((), (ClockSubscription("decision", at),))

            def on_timer(self, event, context):
                observed[name].append(event.time)

        return Clock

    run(factory("early", "09:30"), factory("late", "15:50"))
    assert [stamp.hour for stamp in observed["early"]] == [14, 14]
    assert [stamp.hour for stamp in observed["late"]] == [20, 20]


def test_event_only_research_retains_contract_and_metadata_provenance():
    class Clock(Strategy):
        metadata = StrategyMetadata("clock", version="review-v2")
        contract = StrategyContract((), (ClockSubscription("decision", "09:30"),))

    result = run(Clock)
    provenance = result.config["metadata"]["research"]
    assert provenance["strategy_metadata"]["clock"]["version"] == "review-v2"
    assert provenance["contracts"]["clock"]["instruments"] == []


def test_mixed_owner_resolutions_never_leak_finer_history():
    seen = {"daily": 0, "minute": 0}

    def factory(name, size):
        class Observer(Strategy):
            metadata = StrategyMetadata(name)
            contract = StrategyContract(("SPY",), (BarSubscription(("SPY",), size),))

            def on_bar(self, event, context):
                seen[name] += 1
                expected = 390 * 60 if name == "daily" else 60
                assert all(
                    (bar.end - bar.start).total_seconds() == expected
                    for bar in context.history["SPY"]
                )
                assert len(context.history["SPY"]) == seen[name]

        return Observer

    run(factory("daily", "1 day"), factory("minute", "1 min"))
    assert seen == {"daily": 2, "minute": 780}


def paper_runner(path, strategies, **options):
    from decimal import Decimal
    from convexity.brokers.ibkr import PaperAccountConfig
    from convexity.runtime import PaperTrading

    return PaperTrading(
        PaperAccountConfig("DU123", 4002, True),
        strategies,
        path=path,
        budgets={strategy.metadata.name: Decimal(1000) for strategy in strategies},
        max_order_notional=Decimal(1000),
        market_collar_bps=Decimal(10),
        **options,
    )


@pytest.mark.parametrize("override", ["missing", "warmup", "calendar"])
def test_managed_paper_overrides_cannot_erase_declared_feed_requirements(
    tmp_path, override
):
    from convexity.runtime import StockSubscription

    class Observer(Strategy):
        metadata = StrategyMetadata("observer")
        contract = StrategyContract(
            ("A",), (BarSubscription(("A",), "1 min"),), warmup_sessions=2
        )

    subscriptions = {
        "missing": {},
        "warmup": {"A": StockSubscription(warmup_sessions=0)},
        "calendar": {"A": StockSubscription(calendar="XLON", warmup_sessions=2)},
    }[override]
    with pytest.raises(ValueError):
        paper_runner(tmp_path, [Observer], subscriptions=subscriptions)


def test_managed_paper_override_cannot_replace_declared_clock(tmp_path):
    from convexity.runtime.runner import ScheduleSpec

    class Clock(Strategy):
        metadata = StrategyMetadata("clock")
        contract = StrategyContract(("A",), (ClockSubscription("open", "09:30"),))

    with pytest.raises(ValueError):
        paper_runner(tmp_path, [Clock], schedules=(ScheduleSpec("unrelated", ()),))


def test_paper_startup_keeps_current_session_observations_for_warm_and_cold_owners(
    tmp_path, monkeypatch
):
    import asyncio
    from datetime import datetime, timezone
    from convexity.trading import OrderJournal
    from tests.runtime.test_paper import FakeBroker, bridge
    import convexity.runtime.runner as managed

    current = datetime(2025, 1, 8, 15, 2, tzinfo=timezone.utc)
    monkeypatch.setattr(managed, "_now", lambda: current)

    class Warm(Strategy):
        metadata = StrategyMetadata("warm")
        contract = StrategyContract(
            ("A",), (BarSubscription(("A",), "1 min"),), warmup_sessions=1
        )

    class Cold(Strategy):
        metadata = StrategyMetadata("cold")
        contract = StrategyContract(("A",), (BarSubscription(("A",), "1 min"),))

    calendar = SessionCalendar.from_exchange("XNYS", "2025-01-07", "2025-01-08")
    bars = tuple(
        Bar(
            "A", start, end, Decimal(100), Decimal(100), Decimal(100), Decimal(100), 100
        )
        for start, end in intervals(calendar.sessions, "1 min")
        if end <= current
    )

    class Feed:
        async def history(self, broker, symbol, subscription):
            return HistoricalData(bars, source="review-fixture")

    work = paper_runner(tmp_path, [Warm, Cold], data_source=Feed())

    async def exercise():
        with OrderJournal(tmp_path / "orders.sqlite") as journal:
            broker = FakeBroker(journal)
            strategies = {name: factory() for name, factory in work.factories.items()}
            async with bridge(broker, strategies, defer_start=True) as runtime:
                await work._history(runtime, initial=True)
                await runtime.start()
                warm = runtime.context("warm").history["A"]
                cold = runtime.context("cold").history["A"]
                assert len(warm) == 390 + 32
                assert len(cold) == 32
                assert warm[-1].end == cold[-1].end == current

    asyncio.run(exercise())


def test_default_research_path_reuses_sqlite_then_fetches_only_new_instrument(
    tmp_path, monkeypatch
):
    import convexity.backtest.providers as providers

    class RecordingProvider(ReviewProvider):
        def __init__(self):
            self.requests = []

        async def fetch(self, request, *, refresh=False):
            self.requests.append(request)
            return await super().fetch(request, refresh=refresh)

    provider = RecordingProvider()
    monkeypatch.setattr(providers, "AlpacaHistoricalData", lambda: provider)

    class SPY(Strategy):
        metadata = StrategyMetadata("spy")
        contract = StrategyContract(("SPY",))

        def on_start(self, context):
            return SubmitOrder.market("SPY", "BUY", 1)

    class QQQ(Strategy):
        metadata = StrategyMetadata("qqq")
        contract = StrategyContract(("QQQ",))

        def on_start(self, context):
            return SubmitOrder.market("QQQ", "BUY", 1)

    cache_path = tmp_path / "history.sqlite3"
    options = {"cache": cache_path, "execution": "frictionless"}
    first = Research("2024-01-08", "2024-01-09").run(SPY, **options)
    second = Research("2024-01-08", "2024-01-09").run(SPY, **options)
    assert len(provider.requests) == 1
    assert first.fills == second.fills
    expanded = Research("2024-01-08", "2024-01-09").run((SPY, QQQ), **options)
    assert [request.symbols for request in provider.requests] == [("SPY",), ("QQQ",)]
    assert len(expanded.fills) == 2
    assert cache_path.read_bytes().startswith(b"SQLite format 3")


def test_paper_startup_clock_catches_up_after_warmup_acquisition(tmp_path):
    import asyncio
    from datetime import datetime, timedelta, timezone
    from convexity.runtime import Bars
    from convexity.trading import OrderJournal
    from tests.runtime.test_paper import FakeBroker, bridge

    class Observer(Strategy):
        metadata = StrategyMetadata("observer")
        contract = StrategyContract(
            ("A",), (BarSubscription(("A",), "1 min"),), warmup_sessions=1
        )

        def on_start(self, context):
            assert len(context.history["A"]) == 1
            assert context.history["A"][0].end <= context.time

    async def exercise():
        with OrderJournal(tmp_path / "orders.sqlite") as journal:
            broker = FakeBroker(journal)
            async with bridge(
                broker, {"observer": Observer()}, defer_start=True
            ) as runtime:
                end = datetime.now(timezone.utc) - timedelta(minutes=1)
                # Model a broker snapshot taken before history acquisition crosses
                # a completed-bar boundary, without sleeping or live connections.
                runtime.time = end - timedelta(minutes=2)
                bar = Bar(
                    "A",
                    end - timedelta(minutes=1),
                    end,
                    Decimal(100),
                    Decimal(100),
                    Decimal(100),
                    Decimal(100),
                    100,
                )
                await runtime.warmup((Bars(end, {"A": bar}),))
                runtime.warmup_history("observer", (bar,))
                result = await runtime.start()
                result.raise_for_failure()

    asyncio.run(exercise())
