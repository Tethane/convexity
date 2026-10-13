"""End-to-end public paper runner tests with durable state and a controlled feed."""
import asyncio
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal as D
from typing import Any, Callable, cast

import pytest

from convexity.backtest import HistoricalData, Session, SessionCalendar
from convexity.brokers.ibkr_trading import PaperBroker
from convexity.brokers.ibkr import (AccountValue, Contract, PaperAccountConfig,
                                   PaperAccountSnapshot, Quote)
from convexity.runtime import Bar, Bars, FillEvent, Instrument, Strategy, SubmitOrder, Timer
from convexity.runtime.runner import DataFeedError, PaperTrading, ScheduleSpec, StockSubscription
from convexity.trading import Execution


def now():
    return datetime.now(timezone.utc)


CONTRACT = Contract(123, "A", "STK", "A", "USD", "SMART", "", D(0), "", "", ())


class Broker:
    def __init__(self, config, journal):
        self.config, self.journal = config, journal
        self.on_update: Callable[..., Any] | None = None
        self.closed = False
        self.sent = []
        self.fail = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        self.closed = True

    async def quote(self, request):
        if self.fail:
            raise ConnectionError("socket lost")
        stamp = now()
        return Quote(CONTRACT, "live", "live", D(100), D(100), D(10), D(10),
                     stamp, stamp, stamp, 30, True, "received", ())

    async def snapshot(self):
        if self.fail:
            raise ConnectionError("socket lost")
        stamp = now()
        return PaperAccountSnapshot(self.config.account, stamp, stamp,
            tuple(AccountValue(tag, "10000", "USD") for tag in
                  ("NetLiquidation", "TotalCashValue", "AvailableFunds")), (), ())

    async def submit(self, request, *, submission_guard=None):
        if submission_guard is not None:
            submission_guard()
        self.sent.append(request)
        self.journal.intent(request, self.config.account, len(self.sent), self.config.client_id, 123)
        self.journal.status(request.client_order_id, "Submitted", D(0))
        result = self.journal.get(request.client_order_id)
        assert self.on_update is not None
        self.on_update(result)
        return result

    def fill(self):
        request = self.sent[0]
        self.journal.execution(Execution("fill-1", request.client_order_id, D(1), D(100), "BUY",
                                       now().isoformat(), D(".35"), "USD"))
        assert self.on_update is not None
        self.on_update(self.journal.get(request.client_order_id))


class Feed:
    def __init__(self):
        end = now().replace(second=0, microsecond=0)
        self.values = [Bar("A", end - timedelta(minutes=i+1), end - timedelta(minutes=i),
                           D(100), D(100), D(100), D(100), 100) for i in (2, 1, 0)]
        self.calls = 0

    async def history(self, broker, symbol, subscription):
        self.calls += 1
        return HistoricalData(tuple(self.values), "test")

    async def option_quotes(self, broker, requests):
        return ()


class Recorder(Strategy):
    def __init__(self):
        self.events = []
        self.starts = []

    def on_start(self, context):
        self.starts.append(len(context.history["A"]))

    def on_event(self, event, context):
        self.events.append(event)
        if isinstance(event, Timer):
            return SubmitOrder("A", "BUY", 1, client_order_id="entry")


def app(tmp_path, factory: Callable[[], Strategy] = Recorder, **kwargs):
    current = now()
    calendar = SessionCalendar((Session(current.date(), current - timedelta(hours=3), current + timedelta(hours=3)),))
    brokers = []
    def broker_factory(config, journal):
        broker = Broker(config, journal)
        brokers.append(broker)
        return broker
    options: dict[str, Any] = dict(config=PaperAccountConfig("DU123", 4002, True), strategies={"alpha": factory},
                   instruments={"A": Instrument("A")}, path=tmp_path,
                   budgets={"alpha": D(1000)}, max_order_notional=D(1000), market_collar_bps=D(10),
                   subscriptions={"A": StockSubscription(calendar=calendar)},
                   poll_interval=.01, history_interval=.03, reconnect_delay=.01,
                   broker_factory=broker_factory, data_source=Feed())
    options.update(kwargs)
    return PaperTrading(**options), brokers


def test_warmup_before_start_and_broker_fills_pumped_without_application_refresh(tmp_path):
    async def run():
        strategy = Recorder()
        timer = Timer(now() + timedelta(milliseconds=30), "entry")
        runner, brokers = app(tmp_path, lambda: strategy, schedules=[ScheduleSpec("entry", (timer,))])
        task = await runner.start()
        assert strategy.starts == [3]
        assert not any(isinstance(event, Bars) for event in strategy.events)
        async with asyncio.timeout(2):
            while not brokers[0].sent:
                await asyncio.sleep(.01)
            brokers[0].fill()
            while not any(isinstance(event, FillEvent) for event in strategy.events):
                await asyncio.sleep(.01)
        await runner.stop()
        assert task.done() and brokers[0].closed
        assert runner.status().strategy_states == {"alpha": "stopped"}
        assert len(runner.status().recent_fills) == 1
    asyncio.run(run())


def test_transport_reconnect_constructs_new_strategy_and_preserves_startup_identity(tmp_path):
    async def run():
        instances = []
        def factory():
            strategy = Recorder()
            instances.append(strategy)
            return strategy
        runner, brokers = app(tmp_path, factory)
        await runner.start()
        brokers[0].fail = True
        async with asyncio.timeout(2):
            while len(instances) < 2 or not instances[1].events:
                await asyncio.sleep(.01)
        await runner.stop()
        assert brokers[0].closed and brokers[1].closed
        assert instances[0].starts == [3]
        assert instances[1].starts == []
        assert any(not health.connected for health in runner.health)
    asyncio.run(run())


def test_strategy_error_stops_without_reconnect_and_schedule_remains_uncertain(tmp_path):
    class Broken(Recorder):
        def on_event(self, event, context):
            if isinstance(event, Timer):
                raise RuntimeError("signal failed")
    async def run():
        timer = Timer(now() + timedelta(milliseconds=30), "bad")
        runner, brokers = app(tmp_path, Broken, schedules=[ScheduleSpec("bad", (timer,))], fail_fast=True)
        with pytest.raises(Exception, match="paused|failed|signal"):
            await runner.run(until=now() + timedelta(seconds=2))
        assert len(brokers) == 1 and brokers[0].closed
        import sqlite3
        with sqlite3.connect(tmp_path / "schedules.sqlite") as db:
            assert db.execute("SELECT status FROM schedule_runs").fetchone()[0] == "uncertain"
    asyncio.run(run())


def test_missing_bar_blocks_startup_and_regular_session_closed_gap_is_allowed(tmp_path):
    async def run():
        strategy = Recorder()
        feed = Feed()
        feed.values.pop(1)
        runner, brokers = app(tmp_path, lambda: strategy, data_source=feed)
        with pytest.raises(DataFeedError, match="missing regular-session"):
            await runner.start()
        assert not strategy.starts and not brokers[0].sent
        assert brokers[0].closed
    asyncio.run(run())
    current = now().replace(second=0, microsecond=0)
    prior = current - timedelta(days=1)
    calendar = SessionCalendar((Session(prior.date(), prior - timedelta(hours=1), prior),
                                Session(current.date(), current, current + timedelta(hours=1))))
    runner, _ = app(tmp_path / "other", subscriptions={"A": StockSubscription(calendar=calendar)})
    before = Bar("A", prior - timedelta(minutes=1), prior, D(100), D(100), D(100), D(100), 1)
    after = Bar("A", current, current + timedelta(minutes=1), D(100), D(100), D(100), D(100), 1)
    runner._validate_history("A", (after,), current + timedelta(minutes=1), before)


def test_stale_history_fails_before_strategy_start(tmp_path):
    async def run():
        feed = Feed()
        feed.values = [replace(bar, start=bar.start-timedelta(minutes=30), end=bar.end-timedelta(minutes=30))
                       for bar in feed.values]
        runner, brokers = app(tmp_path, data_source=feed)
        with pytest.raises(DataFeedError, match="stale"):
            await runner.start()
        assert not brokers[0].sent
    asyncio.run(run())


def test_concurrent_runner_cannot_share_deployment_path(tmp_path):
    async def run():
        first, _ = app(tmp_path)
        second, brokers = app(tmp_path)
        await first.start()
        with pytest.raises(RuntimeError, match="Another PaperTrading"):
            await second.start()
        assert not brokers
        await first.stop()
    asyncio.run(run())


def test_backfill_is_ingested_without_stale_decisions_and_only_newest_bar_is_delivered(tmp_path):
    async def run():
        strategy = Recorder()
        feed = Feed()
        complete = list(feed.values)
        feed.values = complete[:1]
        runner, brokers = app(tmp_path, lambda: strategy, data_source=feed, decision_max_age=240)
        await runner.start()
        feed.values = complete
        async with asyncio.timeout(2):
            while not any(isinstance(event, Bars) for event in strategy.events):
                await asyncio.sleep(.01)
        await runner.stop()
        decisions = [event for event in strategy.events if isinstance(event, Bars)]
        assert [event.time for event in decisions] == [complete[-1].end]
        assert runner.runtime is not None
        assert len(runner.runtime.history["A"]) == 3
        assert not brokers[0].sent
    asyncio.run(run())


def test_invalid_account_failure_is_never_retried_as_a_disconnection(tmp_path):
    from convexity.brokers.ibkr import PaperAccountError
    class InvalidBroker(Broker):
        async def snapshot(self):
            raise PaperAccountError("Expected account is unavailable")
    async def run():
        created = []
        def factory(config, journal):
            broker = InvalidBroker(config, journal)
            created.append(broker)
            return broker
        runner, _ = app(tmp_path, broker_factory=factory)
        with pytest.raises(PaperAccountError, match="Expected account"):
            await runner.run()
        assert len(created) == 1
    asyncio.run(run())


def test_transient_reconnect_budget_is_bounded(tmp_path):
    class DisconnectedBroker(Broker):
        async def __aenter__(self):
            raise ConnectionError("unavailable")
    async def run():
        created = []
        def factory(config, journal):
            broker = DisconnectedBroker(config, journal)
            created.append(broker)
            return broker
        runner, _ = app(tmp_path, broker_factory=factory, reconnect_attempts=2)
        with pytest.raises(ConnectionError, match="unavailable"):
            await runner.run()
        assert len(created) == 3
    asyncio.run(run())


def test_changed_deployment_identity_requires_distinct_path(tmp_path):
    async def run():
        first, _ = app(tmp_path)
        await first.run(until=now() + timedelta(milliseconds=30))
        second, brokers = app(tmp_path, budgets={"alpha": D(2000)})
        with pytest.raises(ValueError, match="Deployment configuration differs"):
            await second.run()
        assert not brokers
    asyncio.run(run())


def test_cancellation_releases_broker_and_process_lock(tmp_path):
    async def run():
        first, brokers = app(tmp_path)
        task = await first.start()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert brokers[0].closed
        import fcntl
        with (tmp_path / "runner.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    asyncio.run(run())


def test_cancelling_start_while_history_is_loading_cleans_up_child(tmp_path):
    class BlockedFeed(Feed):
        def __init__(self):
            super().__init__()
            self.entered = asyncio.Event()
        async def history(self, *args):
            self.entered.set()
            await asyncio.Event().wait()
    async def run():
        feed = BlockedFeed()
        runner, brokers = app(tmp_path, data_source=feed)
        starter = asyncio.create_task(runner.start())
        await feed.entered.wait()
        starter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await starter
        assert runner._task is not None
        assert brokers[0].closed and runner._task.done()
        assert runner.status().strategy_states == {"alpha": "stopped"}
    asyncio.run(run())


def test_schedules_deliver_globally_chronologically_across_namespaces(tmp_path):
    class Timers(Recorder):
        def on_event(self, event, context):
            self.events.append(event)
    async def run():
        strategy = Timers()
        current = now()
        early = Timer(current - timedelta(seconds=3), "early")
        middle = Timer(current - timedelta(seconds=2), "middle")
        late = Timer(current - timedelta(seconds=1), "late")
        runner, _ = app(tmp_path, lambda: strategy, schedules=(
            ScheduleSpec("z", (middle,), "catch_up"), ScheduleSpec("a", (early, late), "catch_up")))
        await runner.run(until=now() + timedelta(milliseconds=300))
        assert [event.name for event in strategy.events if isinstance(event, Timer)] == ["early", "middle", "late"]
    asyncio.run(run())


def test_option_subscriptions_use_execution_connection_and_remove_tick_handler(monkeypatch):
    from types import SimpleNamespace
    from datetime import date
    from convexity.brokers.ibkr_data import read_session_option_quotes
    from convexity.brokers.ibkr_options import OptionQuoteRequest
    from convexity.brokers import ibkr
    class Event:
        def __init__(self):
            self.handlers = []
        def __iadd__(self, handler):
            self.handlers.append(handler)
            return self
        def __isub__(self, handler):
            self.handlers.remove(handler)
            return self
    contract = SimpleNamespace(symbol="A", lastTradeDateOrContractMonth="20261218", strike=100,
                               right="C", multiplier="100", currency="USD", tradingClass="A", secType="OPT", conId=456)
    class Client:
        def __init__(self):
            self.pendingTickersEvent = Event()
            self.requested = []
        async def reqContractDetailsAsync(self, requested):
            self.requested.append(requested)
            return [SimpleNamespace(contract=contract)]
        def reqMarketDataType(self, value):
            assert value == 1
        async def reqTickersAsync(self, *contracts, **kwargs):
            assert contracts == (contract,)
            return [SimpleNamespace(contract=contract)]
    async def run():
        client = Client()
        broker = SimpleNamespace(_ib=client, config=SimpleNamespace(timeout=1), _check=lambda: None)
        monkeypatch.setattr(ibkr, "_stock_quote", lambda *args: "observed")
        result = await read_session_option_quotes(cast(PaperBroker, broker), [OptionQuoteRequest("A", date(2026, 12, 18), D(100), "C", "A")])
        assert result == ("observed",)
        assert len(client.requested) == 1
        assert not client.pendingTickersEvent.handlers
    asyncio.run(run())


def test_slow_broker_refresh_cannot_trade_after_data_crosses_freshness_limit(tmp_path):
    class SlowBroker(Broker):
        def __init__(self, config, journal):
            super().__init__(config, journal)
            self.snapshots = 0
        async def snapshot(self):
            self.snapshots += 1
            if self.snapshots == 3:
                await asyncio.sleep(.3)
            return await super().snapshot()
    async def run():
        feed = Feed()
        end = now() - timedelta(milliseconds=5)
        feed.values = [Bar("A", end - timedelta(minutes=1), end, D(100), D(100), D(100), D(100), 100)]
        brokers = []
        def factory(config, journal):
            broker = SlowBroker(config, journal)
            brokers.append(broker)
            return broker
        timer = Timer(now() + timedelta(milliseconds=200), "entry")
        runner, _ = app(tmp_path, broker_factory=factory, data_source=feed, decision_max_age=.15,
                        schedules=[ScheduleSpec("entry", (timer,))], history_interval=60)
        with pytest.raises(Exception, match="stale"):
            await runner.run(until=now() + timedelta(seconds=1))
        assert not brokers[0].sent
    asyncio.run(run())


def test_managed_failure_isolates_owner_while_fills_operator_actions_and_healthy_timers_continue(tmp_path):
    from convexity.runtime import CancelOrder, Rejection
    from convexity.brokers.ibkr import Position as BrokerPosition
    class Failed(Recorder):
        def on_start(self, context):
            return SubmitOrder("A", "BUY", 2, client_order_id="initial")
        def on_event(self, event, context):
            self.events.append(event)
            if isinstance(event, Timer):
                raise RuntimeError("broken signal")
    class Healthy(Recorder):
        def on_event(self, event, context):
            self.events.append(event)
    class TrackingBroker(Broker):
        async def snapshot(self):
            snapshot = await super().snapshot()
            quantity = sum((execution.quantity * (1 if execution.side == "BUY" else -1)
                            for record in self.journal.orders() for execution in record.executions), D(0))
            return replace(snapshot, positions=(BrokerPosition(CONTRACT, quantity, D(100)),))
        async def cancel(self, identity):
            record = self.journal.get(identity)
            self.journal.status(identity, "Cancelled", record.filled_quantity)
            assert self.on_update is not None
            self.on_update(self.journal.get(identity))
            return self.journal.get(identity)
        def fill(self):
            super().fill()
            self.journal.status(self.sent[0].client_order_id, "Submitted", D(1))
            assert self.on_update is not None
            self.on_update(self.journal.get(self.sent[0].client_order_id))
    async def run():
        failed, healthy = Failed(), Healthy()
        brokers = []
        def factory(config, journal):
            broker = TrackingBroker(config, journal)
            brokers.append(broker)
            return broker
        current = now()
        first = Timer(current - timedelta(seconds=1), "first")
        second = Timer(current + timedelta(milliseconds=150), "second")
        runner, _ = app(tmp_path, strategies={"bad": lambda: failed, "good": lambda: healthy},
                        budgets={"bad": D(1000), "good": D(1000)}, broker_factory=factory,
                        schedules=(ScheduleSpec("shared", (first, second), "catch_up"),))
        task = await runner.start()
        async with asyncio.timeout(3):
            while runner.status().strategy_states["bad"] != "paused":
                await asyncio.sleep(.01)
            assert not task.done() and not brokers[0].closed
            brokers[0].fill()
            while not runner.status().recent_fills:
                await asyncio.sleep(.01)
            assert runner.status().positions["bad"]["A"].quantity == 1
            # Failed strategies cannot open entries, but remain manageable.
            rejected = await runner.command("bad", SubmitOrder("A", "BUY", 1, client_order_id="forbidden"))
            assert any(isinstance(event, Rejection) for event in rejected)
            await runner.command("bad", CancelOrder("bad:initial"))
            await runner.command("bad", SubmitOrder("A", "SELL", 1, client_order_id="reduce"))
            assert brokers[0].sent[-1].side == "SELL"
            while len([event for event in healthy.events if isinstance(event, Timer)]) < 2:
                await asyncio.sleep(.01)
            states = {(row["strategy_id"], row["name"]): row["status"] for row in runner.schedule_status()}
            assert states == {("bad", "first"): "uncertain", ("bad", "second"): "pending",
                              ("good", "first"): "completed", ("good", "second"): "completed"}
            assert [event.name for event in failed.events if isinstance(event, Timer)] == ["first"]
            assert runner.status().strategy_states == {"bad": "paused", "good": "active"}
            assert not task.done()
        await runner.stop()
        assert len(brokers) == 1 and brokers[0].closed
        assert runner.schedule_status()
    asyncio.run(run())


def test_startup_strategy_failure_keeps_healthy_strategy_and_monitoring_ready(tmp_path):
    class BadStart(Recorder):
        def on_start(self, context):
            raise RuntimeError("startup calculation failed")
    async def run():
        healthy = Recorder()
        runner, brokers = app(tmp_path, strategies={"bad": BadStart, "good": lambda: healthy},
                               budgets={"bad": D(1000), "good": D(1000)})
        task = await runner.start()
        assert runner.status().strategy_states == {"bad": "paused", "good": "active"}
        assert healthy.starts == [3]
        async with asyncio.timeout(2):
            while not healthy.events:
                await asyncio.sleep(.01)
        assert not task.done() and not brokers[0].closed
        await runner.stop()
    asyncio.run(run())


def test_restart_with_unresolved_owner_preserves_isolation_and_healthy_future_schedule(tmp_path):
    class Broken(Recorder):
        def on_event(self, event, context):
            self.events.append(event)
            if isinstance(event, Timer):
                raise RuntimeError("failed timer")
    class Healthy(Recorder):
        def on_event(self, event, context):
            self.events.append(event)
    async def run():
        healthy_instances, broken_instances = [], []
        def bad_factory():
            strategy = Broken()
            broken_instances.append(strategy)
            return strategy
        def good_factory():
            strategy = Healthy()
            healthy_instances.append(strategy)
            return strategy
        current = now()
        timers = (Timer(current - timedelta(seconds=1), "first"),
                  Timer(current + timedelta(milliseconds=500), "later"))
        options: dict[str, Any] = dict(strategies={"bad": bad_factory, "good": good_factory},
                       budgets={"bad": D(1000), "good": D(1000)},
                       schedules=(ScheduleSpec("shared", timers, "catch_up"),))
        first, _ = app(tmp_path, **options)
        await first.start()
        async with asyncio.timeout(2):
            while first.status().strategy_states["bad"] != "paused":
                await asyncio.sleep(.01)
        await first.stop()
        second, brokers = app(tmp_path, subscriptions=first.subscriptions, **options)
        task = await second.start()
        assert second.status().strategy_states == {"bad": "paused", "good": "active"}
        async with asyncio.timeout(3):
            while not any(isinstance(event, Timer) and event.name == "later" for event in healthy_instances[1].events):
                await asyncio.sleep(.01)
        assert [event.name for event in healthy_instances[1].events if isinstance(event, Timer)] == ["later"]
        assert not any(isinstance(event, Timer) for event in broken_instances[1].events)
        assert not task.done() and not brokers[0].closed
        assert any(row["strategy_id"] == "bad" and row["name"] == "first" and row["status"] == "uncertain"
                   for row in second.schedule_status())
        await second.stop()
    asyncio.run(run())
