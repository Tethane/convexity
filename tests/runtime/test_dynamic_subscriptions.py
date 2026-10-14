"""Dynamic subscriptions are asynchronous, capability bounded, and replay safe."""

import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import cast

from convexity.backtest import SimulatedExchange
from convexity.runtime import (
    Bar,
    Bars,
    BarSubscription,
    ClockSubscription,
    Instrument,
    Runtime,
    Strategy,
    StrategyContract,
    Timer,
)
from convexity.runtime.state import StrategyState
from convexity.runtime.subscriptions import SubscriptionBook, subscription_id
from convexity.runtime.types import Subscribe, SubscriptionUpdate, Unsubscribe
from convexity.trading import OrderJournal
from tests.runtime.test_paper import FakeBroker, bridge

T = datetime(2026, 1, 2, 15, tzinfo=timezone.utc)
H = timedelta(hours=1)
A = BarSubscription(("A",), "1 hour")
B = BarSubscription(("B",), "1 hour")


def panel(instant):
    return Bars(
        instant,
        {
            symbol: Bar(
                symbol,
                instant - H,
                instant,
                Decimal(100),
                Decimal(100),
                Decimal(100),
                Decimal(100),
                100,
            )
            for symbol in ("A", "B")
        },
    )


def test_initial_identity_stable_activation_idempotent_and_unauthorized_rejected():
    contract = StrategyContract(("A",), (A,), available_subscriptions=(B,))
    book = SubscriptionBook(contract, "owner")
    assert subscription_id(A) == subscription_id(BarSubscription(("A",), "1 hour"))
    assert dict(book.active) == {subscription_id(A): A}
    event = book.apply(Subscribe(B, "temporary"), T)
    assert event.status == "active"
    snapshot = book.dump()
    assert book.apply(Subscribe(B, "temporary"), T + H).status == "active"
    assert book.dump() == snapshot
    assert book.apply(Subscribe(A, "temporary"), T).status == "rejected"
    assert (
        book.apply(Subscribe(BarSubscription(("C",)), "outside"), T).status
        == "rejected"
    )
    assert book.apply(Unsubscribe("unknown"), T).status == "rejected"
    assert book.apply(Unsubscribe("temporary"), T + H).status == "inactive"


def test_context_queued_commands_take_effect_at_callback_boundary_once():
    class Dynamic(Strategy):
        contract = StrategyContract(
            ("A",), (A, ClockSubscription("restart")), available_subscriptions=(B,)
        )

        def __init__(self):
            self.bars, self.lifecycle = [], []

        def on_start(self, context):
            context.unsubscribe(subscription_id(A))
            command = context.subscribe(B, subscription_id="temporary")
            assert (
                subscription_id(A) in context.subscriptions
                and "temporary" not in context.subscriptions
            )
            return command  # Returning the queued command must not apply it twice.

        def on_subscription(self, event, context):
            self.lifecycle.append(event)

        def on_bar(self, event, context):
            self.bars.append((event, context.history))
            if len(self.bars) == 1:
                context.unsubscribe("temporary")

        def on_timer(self, event, context):
            context.subscribe(B, subscription_id="temporary")

    dynamic = Dynamic()
    instruments = {symbol: Instrument(symbol) for symbol in ("A", "B")}
    runtime = Runtime({"dynamic": dynamic}, SimulatedExchange(instruments), instruments)
    runtime.start(T)
    assert [e.status for e in dynamic.lifecycle] == ["inactive", "active"]
    for i in (1, 2, 3, 4):
        instant = T + i * H
        runtime.advance(
            instant, panel(instant), (Timer(instant, "restart"),) if i == 3 else ()
        )
    assert [event.time for event, _ in dynamic.bars] == [T + H, T + 4 * H]
    assert all(set(event.bars) == {"B"} for event, _ in dynamic.bars)
    assert tuple(b.end for b in dynamic.bars[-1][1]["B"]) == (T + 4 * H,)
    assert not runtime.diagnostics


def test_unauthorized_subscription_reports_lifecycle_without_expanding_trade_contract():
    class Invalid(Strategy):
        contract = StrategyContract(("A",))

        def __init__(self):
            self.events = []

        def on_start(self, context):
            context.subscribe(B)

        def on_subscription(self, event, context):
            self.events.append(event)

    invalid = Invalid()
    instruments = {"A": Instrument("A")}
    runtime = Runtime({"owner": invalid}, SimulatedExchange(instruments), instruments)
    runtime.start(T)
    assert len(invalid.events) == 1 and invalid.events[0].status == "rejected"
    assert not runtime.context("owner").subscriptions


def test_persistent_subscription_state_scoped_by_account_owner_and_activation(tmp_path):
    contract = StrategyContract(("A",), (A,), available_subscriptions=(B,))
    book = SubscriptionBook(contract, "owner")
    book.apply(Unsubscribe(subscription_id(A)), T)
    book.apply(Subscribe(B, "temporary"), T + H)
    with_state = StrategyState(tmp_path / "state.sqlite", "account")
    with_state.save_subscriptions("owner", book.dump())
    restored = SubscriptionBook(
        contract, "owner", with_state.load_subscriptions("owner")
    )
    assert not restored.select(panel(T + H))
    delivered = restored.select(panel(T + 2 * H))
    assert isinstance(delivered, Bars) and set(delivered.bars) == {"B"}
    assert with_state.load_subscriptions("other") is None
    with_state.close()
    other_account = StrategyState(tmp_path / "state.sqlite", "other")
    assert other_account.load_subscriptions("owner") is None
    other_account.close()


def test_paper_restores_unsubscribed_initial_feed_on_reconnect(tmp_path):
    class Dynamic(Strategy):
        contract = StrategyContract(("A",), (A,))

        def __init__(self):
            self.acks = []

        def on_start(self, context):
            context.unsubscribe(subscription_id(A))

        def on_subscription(self, event, context):
            self.acks.append(event)

    async def run():
        with OrderJournal(tmp_path / "orders.sqlite") as journal:
            first = Dynamic()
            async with bridge(FakeBroker(journal), {"owner": first}) as runtime:
                assert not runtime.context("owner").subscriptions
                assert len(first.acks) == 1 and first.acks[0].status == "inactive"
            second = Dynamic()
            async with bridge(FakeBroker(journal), {"owner": second}) as runtime:
                assert not runtime.context("owner").subscriptions
                assert not second.acks

    asyncio.run(run())


def test_research_prefetches_capabilities_but_dispatches_only_after_activation():
    from convexity.backtest.research import Research
    from convexity.runtime import StrategyMetadata
    from tests.backtest.test_research import FixtureProvider

    inactive = BarSubscription(("QQQ",))
    observed = []

    class Dynamic(Strategy):
        metadata = StrategyMetadata("dynamic")
        contract = StrategyContract(
            ("SPY",),
            (BarSubscription(("SPY",)),),
            warmup_sessions=2,
            available_subscriptions=(inactive,),
        )

        def on_start(self, context):
            assert set(context.history) == {"SPY"} and len(context.history["SPY"]) == 2

        def on_bar(self, event, context):
            observed.append((event, context.history))
            if len(observed) == 1:
                context.subscribe(inactive)

    provider = FixtureProvider()
    Research("2024-01-08", "2024-01-10").run(Dynamic, provider=provider, cache=None)
    assert set(provider.calls[0][0].symbols) == {"SPY", "QQQ"}
    assert [set(event.bars) for event, _ in observed] == [
        {"SPY"},
        {"SPY", "QQQ"},
        {"SPY", "QQQ"},
    ]
    assert len(observed[1][1]["QQQ"]) == 1


def test_paper_feed_requests_follow_shared_active_demand(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from convexity.backtest import HistoricalData
    from convexity.brokers.ibkr import PaperAccountConfig
    from convexity.runtime import PaperRuntime, PaperTrading, StrategyMetadata
    from convexity.runtime.state import DeliveryResult

    first, second = BarSubscription(("A",), "1 min"), BarSubscription(("B",), "1 min")

    class Dynamic(Strategy):
        metadata = StrategyMetadata("dynamic")
        contract = StrategyContract(("A",), (first,), available_subscriptions=(second,))

    calls = []
    instant = datetime.now(timezone.utc).replace(second=0, microsecond=0)

    class Feed:
        async def option_quotes(self, broker, requests):
            return ()

        async def history(self, broker, symbol, subscription):
            calls.append(symbol)
            return HistoricalData(
                (
                    Bar(
                        symbol,
                        instant - timedelta(minutes=1),
                        instant,
                        Decimal(100),
                        Decimal(100),
                        Decimal(100),
                        Decimal(100),
                        100,
                    ),
                ),
                "fixture",
            )

    work = PaperTrading(
        PaperAccountConfig("DU123", 4002, True),
        (Dynamic,),
        path=tmp_path,
        budgets={"dynamic": Decimal(1000)},
        max_order_notional=Decimal(1000),
        market_collar_bps=Decimal(10),
        data_source=Feed(),
    )
    book = SubscriptionBook(Dynamic.contract, "dynamic")
    history = {"A": [], "B": []}

    async def ingest(events):
        for event in events:
            for symbol, bar in event.bars.items():
                history[symbol].append(bar)

    async def step(event):
        await ingest((event,))
        return DeliveryResult()

    runtime = cast(
        PaperRuntime,
        SimpleNamespace(
            _subscriptions={"dynamic": book},
            history=history,
            broker=None,
            ingest_history=ingest,
            step=step,
        ),
    )
    monkeypatch.setattr(work, "_validate_history", lambda *args: None)

    async def run():
        await work._history(runtime, initial=False)
        book.apply(Unsubscribe(subscription_id(first)), instant)
        await work._history(runtime, initial=False)
        book.apply(Subscribe(second, "second"), instant)
        await work._history(runtime, initial=False)

    asyncio.run(run())
    assert calls == ["A", "B"]


def test_invalid_subscription_rejection_remains_exportable(tmp_path):
    from convexity.backtest.research import Research
    from convexity.runtime import StrategyMetadata

    class Invalid(Strategy):
        metadata = StrategyMetadata("invalid-subscription")
        contract = StrategyContract(())

        def on_start(self, context):
            return Subscribe(object(), "invalid")

    result = Research("2024-01-08", "2024-01-09").run(Invalid)
    updates = [
        event for event in result.events if isinstance(event, SubscriptionUpdate)
    ]
    assert len(updates) == 1 and updates[0].status == "rejected"
    assert updates[0].subscription is None
    result.export(tmp_path / "export")
    assert (tmp_path / "export" / "result.json").exists()
