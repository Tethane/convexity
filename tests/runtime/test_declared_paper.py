"""Managed paper feeds derive schedules and warmup from strategy declarations."""

import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from convexity.backtest import HistoricalData, Session, SessionCalendar
from convexity.brokers.ibkr import PaperAccountConfig
from convexity.runtime import (
    Bar,
    BarSubscription,
    ClockSubscription,
    Instrument,
    PaperTrading,
    Strategy,
    StrategyContract,
    StrategyMetadata,
    StockSubscription,
)
from convexity.trading import OrderJournal
from tests.runtime.test_paper import FakeBroker, bridge


def runner(tmp_path, strategies, **kwargs):
    return PaperTrading(
        PaperAccountConfig("DU123", 4002, True),
        strategies,
        path=tmp_path,
        budgets={s.metadata.name: Decimal(1000) for s in strategies},
        max_order_notional=Decimal(1000),
        market_collar_bps=Decimal(10),
        **kwargs,
    )


def test_paper_names_instruments_and_clock_schedules_come_from_strategies(tmp_path):
    class Morning(Strategy):
        metadata = StrategyMetadata("morning")
        contract = StrategyContract(("A",), (ClockSubscription("trade", at="09:30"),))

    class Afternoon(Strategy):
        metadata = StrategyMetadata("afternoon")
        contract = StrategyContract(("A",), (ClockSubscription("trade", at="15:50"),))

    work = runner(tmp_path, [Morning, Afternoon])
    assert work.instruments == {"A": Instrument("A")}
    assert set(work.factories) == {"morning", "afternoon"}
    assert not work.subscriptions
    assert len(work.schedule_specs) == 2
    assert len({s.namespace for s in work.schedule_specs}) == 2


def test_paper_loads_shared_history_but_only_each_owners_declared_warmup(
    tmp_path, monkeypatch
):
    class Warm(Strategy):
        metadata = StrategyMetadata("warm")
        contract = StrategyContract(
            ("A",), (BarSubscription(("A",), "1 min"),), warmup_sessions=2
        )

    class Cold(Strategy):
        metadata = StrategyMetadata("cold")
        contract = StrategyContract(("A",), (BarSubscription(("A",), "1 min"),))

    now = datetime.now(timezone.utc)
    starts = [
        now.replace(hour=12, minute=0, second=0, microsecond=0) - timedelta(days=day)
        for day in (3, 2, 1)
    ]
    sessions = tuple(
        Session(start.date(), start, start + timedelta(minutes=2)) for start in starts
    )
    future = now + timedelta(days=1)
    calendar = SessionCalendar(
        (*sessions, Session(future.date(), future, future + timedelta(minutes=2)))
    )
    bars = tuple(
        Bar(
            "A",
            start + timedelta(minutes=minute),
            start + timedelta(minutes=minute + 1),
            Decimal(100),
            Decimal(100),
            Decimal(100),
            Decimal(100),
            100,
        )
        for start in starts
        for minute in range(2)
    )

    class Feed:
        async def history(self, broker, symbol, subscription):
            assert subscription.warmup_sessions == 2
            return HistoricalData(bars, "fixture")

    monkeypatch.setattr(
        "convexity.runtime.runner._warmup_calendar", lambda subscription, now: calendar
    )
    work = runner(tmp_path, [Warm, Cold], data_source=Feed())

    async def run():
        with OrderJournal(tmp_path / "orders.sqlite") as journal:
            broker = FakeBroker(journal)
            strategies = {name: factory() for name, factory in work.factories.items()}
            async with bridge(broker, strategies, defer_start=True) as runtime:
                await work._history(runtime, initial=True)
                await runtime.start()
                assert len(runtime.context("warm").history["A"]) == 4
                assert not runtime.context("cold").history["A"]

    asyncio.run(run())


def test_paper_derives_maximum_shared_warmup(tmp_path):
    class One(Strategy):
        metadata = StrategyMetadata("one")
        contract = StrategyContract(
            ("A",), (BarSubscription(("A",), "1 min"),), warmup_sessions=1
        )

    class Five(One):
        metadata = StrategyMetadata("five")
        contract = StrategyContract(
            ("A",), (BarSubscription(("A",), "1 min"),), warmup_sessions=5
        )

    work = runner(tmp_path, [One, Five])
    assert work.subscriptions["A"].warmup_sessions == 5


def test_mapping_factories_derive_instance_contracts_and_reuse_initial_instance(
    tmp_path,
):
    created = []

    class Configured(Strategy):
        def __init__(self):
            self.metadata = StrategyMetadata("configured")
            self.contract = StrategyContract(
                ("A",), (BarSubscription(("A",), "1 min"),), warmup_sessions=5
            )
            created.append(self)

    work = PaperTrading(
        PaperAccountConfig("DU123", 4002, True),
        {"configured": lambda: Configured()},
        path=tmp_path,
        budgets={"configured": Decimal(1000)},
        max_order_notional=Decimal(1000),
        market_collar_bps=Decimal(10),
    )
    assert work.subscriptions["A"].warmup_sessions == 5
    assert len(created) == 1
    assert work.factories["configured"]() is created[0]
    assert len(created) == 1


def test_typed_timer_events_materialize_per_owner_and_repeat_after_restart(tmp_path):
    from datetime import time
    from convexity.runtime import ClockSchedule, EventSubscription, Timer
    from convexity.runtime.subscriptions import SubscriptionBook

    class Morning(Strategy):
        metadata = StrategyMetadata("morning")
        contract = StrategyContract(
            ("A",),
            (EventSubscription(Timer, schedule=ClockSchedule("trade", time(9, 30))),),
        )

    class Afternoon(Strategy):
        metadata = StrategyMetadata("afternoon")
        contract = StrategyContract(
            ("A",),
            (EventSubscription(Timer, schedule=ClockSchedule("trade", time(15, 50))),),
        )

    started = datetime(2024, 1, 2, 0, tzinfo=timezone.utc)
    due = datetime(2024, 1, 3, 21, tzinfo=timezone.utc)
    work = runner(tmp_path, [Morning, Afternoon])
    assert all(isinstance(s, ClockSchedule) for s in work.schedule_specs)
    books = {
        "morning": SubscriptionBook(Morning.contract, "morning"),
        "afternoon": SubscriptionBook(Afternoon.contract, "afternoon"),
    }
    received = {owner: [] for owner in books}
    schedules = []
    work._sync_schedules(schedules, started, due)
    try:
        for schedule in schedules:
            _, owner = work._schedule_owners[schedule.namespace]

            def deliver(run, owner=owner):
                assert books[owner].select(run.timer) == run.timer
                received[owner].append(run.timer)

            schedule.poll(due, deliver)
        assert [e.time.hour for e in received["morning"]] == [14, 14]
        assert [e.time.hour for e in received["afternoon"]] == [20, 20]
    finally:
        for schedule in schedules:
            schedule.close()
    restarted = runner(tmp_path, [Morning, Afternoon])
    schedules = []
    next_due = datetime(2024, 1, 4, 21, tzinfo=timezone.utc)
    restarted._sync_schedules(schedules, started, next_due)
    delivered = []
    try:
        for schedule in schedules:
            delivered.extend(schedule.poll(next_due, lambda run: None))
        assert len(delivered) == 2
        assert all(run.timer.time.day == 4 for run in delivered)
    finally:
        for schedule in schedules:
            schedule.close()
