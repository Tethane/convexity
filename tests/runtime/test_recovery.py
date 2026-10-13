"""Fault injection at the durable callback/command and schedule boundaries."""
import asyncio
import sqlite3
from datetime import timedelta
from decimal import Decimal as D

import pytest

from convexity.runtime import AccountUpdate, Bar, Bars, ConnectionHealth, DurableSchedule, FillEvent, StockQuotes, Strategy, SubmitOrder, Timer
from convexity.runtime.state import DeliveryError, DeliveryResult, StrategyState
from convexity.strategies import TargetStrategy
from convexity.trading import Execution, OrderJournal, StockOrder
from tests.runtime.test_paper import FakeBroker, bridge, now


def bars(age=0):
    end = now() - timedelta(seconds=age)
    return Bars(end, {"A": Bar("A", end - timedelta(minutes=1), end, D(100), D(100), D(100), D(100), 100)})


def test_runtime_failure_cannot_complete_schedule_even_if_adapter_discards_result(tmp_path):
    class Broken(Strategy):
        def on_event(self, event, context):
            if isinstance(event, Timer):
                raise RuntimeError("signal failure")

    async def run():
        timer = Timer(now(), "exit")
        with OrderJournal(tmp_path / "journal.sqlite") as journal, DurableSchedule(
                tmp_path / "schedule.sqlite", "exit", (timer,), started_at=timer.time) as schedule:
            async with bridge(FakeBroker(journal), {"s": Broken()}) as runtime:
                async def deliver(run):
                    await runtime.step(run.timer)  # deliberately discard result
                with pytest.raises(DeliveryError, match="signal failure"):
                    await schedule.poll_async(now(), deliver)
                assert schedule.status()[0]["status"] == "uncertain"
                assert "signal failure" in schedule.status()[0]["detail"]
                assert runtime.state.unresolved("s")
                with pytest.raises(DeliveryError):
                    await runtime.step(Timer(now(), "another"))
    asyncio.run(run())


def test_schedule_checks_explicit_failed_delivery_results(tmp_path):
    timer = Timer(now(), "exit")
    with DurableSchedule(tmp_path / "schedule.sqlite", "exit", (timer,), started_at=timer.time) as schedule:
        with pytest.raises(DeliveryError):
            schedule.poll(now(), lambda run: DeliveryResult(("paused",)))
        assert schedule.status()[0]["status"] == "uncertain"


def test_completed_target_decision_resumes_after_restart_without_repeating_timer(tmp_path):
    async def run():
        path = tmp_path / "journal.sqlite"
        with OrderJournal(path) as journal:
            broker = FakeBroker(journal)
            original = TargetStrategy(lambda context: {"A": 2})
            timer = Timer(now(), "rebalance")
            async with bridge(broker, {"s": original}) as runtime:
                await runtime.step(timer)
                broker.fill(1, commission=D(".35"))
                await runtime.refresh()
                await broker.cancel(broker.sent[0].client_order_id)
                await runtime.refresh()
            restored = TargetStrategy(lambda context: pytest.fail("Completed timer was replayed"))
            second = FakeBroker(journal)
            async with bridge(second, {"s": restored}) as runtime:
                assert restored.targets == {"A": 2}
                assert restored._last_timer == timer.time
                await runtime.step(timer)
                await runtime.step(bars())
                assert len(second.sent) == 1
                assert second.sent[0].quantity == 1
                assert second.sent[0].side == "BUY"
    asyncio.run(run())


def test_crash_between_commands_preserves_decision_but_prevents_automatic_replay(tmp_path):
    class Stateful(Strategy):
        def __init__(self):
            self.decisions = 0

        def dump_state(self):
            return {"decisions": self.decisions}

        def load_state(self, state):
            if state:
                self.decisions = state["decisions"]

        def on_event(self, event, context):
            if isinstance(event, Timer):
                self.decisions += 1
                return SubmitOrder("A", "BUY", 1), SubmitOrder("A", "BUY", 1)

    class Crash(FakeBroker):
        async def submit(self, request, *, submission_guard=None):
            if submission_guard is not None:
                submission_guard()
            if self.sent:
                raise asyncio.CancelledError("process interrupted during command batch")
            return await super().submit(request)

    async def run():
        path = tmp_path / "journal.sqlite"
        with OrderJournal(path) as journal:
            first = Crash(journal)
            async with bridge(first, {"s": Stateful()}) as runtime:
                with pytest.raises(asyncio.CancelledError):
                    await runtime.step(Timer(now(), "rebalance"))
                assert len(first.sent) == 1
            second, strategy = FakeBroker(journal), Stateful()
            async with bridge(second, {"s": strategy}) as runtime:
                assert strategy.decisions == 1
                assert runtime.paused == {"s"}
                with pytest.raises(DeliveryError):
                    await runtime.step(Timer(now(), "rebalance"))
                assert not second.sent
                assert len(runtime.state.unresolved("s")) == 1
            state = StrategyState(path, "DU123")
            identity = state.unresolved("s")[0]["identity"]
            with pytest.raises(ValueError, match="evidence"):
                state.resolve("s", identity, detail=" ")
            state.resolve("s", identity, detail="Verified first order acknowledged; second never submitted.")
            state.close()
            resumed = Stateful()
            async with bridge(FakeBroker(journal), {"s": resumed}) as runtime:
                assert not runtime.paused
                assert resumed.decisions == 1
    asyncio.run(run())


def test_startup_orders_are_not_repeated_on_reconnect(tmp_path):
    class Startup(Strategy):
        def on_start(self, context):
            return SubmitOrder("A", "BUY", 1)

    async def run():
        with OrderJournal(tmp_path / "journal.sqlite") as journal:
            first, second = FakeBroker(journal), FakeBroker(journal)
            async with bridge(first, {"s": Startup()}):
                assert len(first.sent) == 1
            async with bridge(second, {"s": Startup()}):
                assert not second.sent
    asyncio.run(run())


def test_history_warmup_and_backfill_never_invoke_trading_callbacks(tmp_path):
    class EveryBar(Strategy):
        def __init__(self):
            self.observed = []

        def on_start(self, context):
            self.observed.append(len(context.history["A"]))

        def on_event(self, event, context):
            if isinstance(event, Bars):
                return SubmitOrder("A", "BUY", 1)

    async def run():
        with OrderJournal(tmp_path / "journal.sqlite") as journal:
            broker, strategy = FakeBroker(journal), EveryBar()
            async with bridge(broker, {"s": strategy}, defer_start=True) as runtime:
                old = bars(age=3600)
                await runtime.warmup((old,))
                assert not strategy.observed and not broker.sent
                (await runtime.start()).raise_for_failure()
                assert strategy.observed == [1]
                with pytest.raises(ValueError, match="Stale"):
                    await runtime.step(bars(age=1800))
                await runtime.ingest_history((bars(age=1800),))
                assert not broker.sent
                await runtime.step(bars())
                assert len(broker.sent) == 1
    asyncio.run(run())


def test_refresh_delivers_observations_and_disconnect_disables_commands(tmp_path):
    class Observe(Strategy):
        def __init__(self):
            self.events = []

        def on_event(self, event, context):
            self.events.append(event)
            if isinstance(event, AccountUpdate):
                assert context.cash == 10000
            if isinstance(event, ConnectionHealth) and not event.connected:
                return SubmitOrder("A", "BUY", 1)

    async def run():
        with OrderJournal(tmp_path / "journal.sqlite") as journal:
            broker, strategy = FakeBroker(journal), Observe()
            async with bridge(broker, {"s": strategy}) as runtime:
                await runtime.refresh()
                assert any(isinstance(e, StockQuotes) for e in strategy.events)
                assert any(isinstance(e, AccountUpdate) for e in strategy.events)
                async def unavailable():
                    pytest.fail("Disconnected health delivery requested broker data")
                broker.snapshot = unavailable
                result = await runtime.step(ConnectionHealth(now(), False, "socket lost"))
                assert not result.successful and not broker.sent
                assert any("disconnected" in error for error in result.errors)
    asyncio.run(run())


@pytest.mark.parametrize("arrival", ["downtime", "late_commission", "already_delivered"])
def test_reconciled_fill_callbacks_update_checkpoint_once_across_restarts(tmp_path, arrival):
    class Counter(Strategy):
        def __init__(self):
            self.fills = 0
            self.deliveries = 0

        def dump_state(self):
            return {"fills": self.fills}

        def load_state(self, state):
            if state:
                self.fills = state["fills"]

        def on_event(self, event, context):
            if isinstance(event, Timer):
                return SubmitOrder("A", "BUY", 1)
            if isinstance(event, FillEvent):
                self.fills += 1
                self.deliveries += 1

    async def run():
        with OrderJournal(tmp_path / "journal.sqlite") as journal:
            broker, strategy = FakeBroker(journal), Counter()
            async with bridge(broker, {"s": strategy}) as runtime:
                await runtime.step(Timer(now(), "buy"))
                if arrival != "downtime":
                    broker.fill(1, commission=D(".35") if arrival == "already_delivered" else None)
                    await runtime.refresh()
                assert strategy.fills == (1 if arrival == "already_delivered" else 0)
            if arrival == "downtime":
                journal.execution(Execution("execution-1", broker.sent[0].client_order_id, D(1), D(100),
                                            "BUY", now().isoformat(), D(".35"), "USD"))
            elif arrival == "late_commission":
                journal.commission("DU123", "execution-1", D(".35"), "USD")
            restored = Counter()
            async with bridge(FakeBroker(journal), {"s": restored}) as runtime:
                assert restored.fills == 1
                assert restored.deliveries == (0 if arrival == "already_delivered" else 1)
                await runtime.refresh()
                assert restored.fills == 1
            twice = Counter()
            async with bridge(FakeBroker(journal), {"s": twice}):
                assert twice.fills == 1 and twice.deliveries == 0
    asyncio.run(run())


def test_first_deployment_baselines_preexisting_fills_even_after_late_commission(tmp_path):
    class NoHistoricalCallbacks(Strategy):
        def on_event(self, event, context):
            if isinstance(event, FillEvent):
                pytest.fail("A first deployment must not replay an adopted historical fill")

    async def run():
        with OrderJournal(tmp_path / "journal.sqlite") as journal:
            journal.intent(StockOrder("legacy", "s", "A", "BUY", 1, D(100)), "DU123", 1, 71, 123)
            journal.status("legacy", "Filled", D(1))
            journal.execution(Execution("legacy-fill", "legacy", D(1), D(100), "BUY", now().isoformat()))
            async with bridge(FakeBroker(journal), {"s": NoHistoricalCallbacks()}):
                pass
            journal.commission("DU123", "legacy-fill", D(".35"), "USD")
            async with bridge(FakeBroker(journal), {"s": NoHistoricalCallbacks()}) as runtime:
                assert runtime.context("s").position("A") == 1
                assert not runtime.paused
    asyncio.run(run())


def test_completed_observation_retention_preserves_recovery_and_trading_identities(tmp_path):
    path = tmp_path / "state.sqlite"
    state = StrategyState(path, "DU123")
    for kind in ("StockQuotes", "AccountUpdate", "ConnectionHealth"):
        state.claim("s", f"{kind}:resolved")
        state.resolve("s", f"{kind}:resolved", detail="Reconciled observation effects.")
        for index in range(150):
            identity = f"{kind}:{index}"
            assert state.claim("s", identity)
            state.complete("s", identity)
    retained = ("startup", "fill:execution", "order:owned:filled", "Timer:decision", "Bars:history")
    state.baseline("s", retained)
    state.claim("s", "StockQuotes:uncertain")
    # An idempotent completion must not remove unresolved or reconciled rows.
    state.complete("s", "StockQuotes:149")
    state.baseline("other", (f"StockQuotes:{i}" for i in range(150)))
    state.close()
    with sqlite3.connect(path) as db:
        for kind in ("StockQuotes", "AccountUpdate", "ConnectionHealth"):
            rows = db.execute("SELECT identity,status FROM strategy_callbacks WHERE owner='s' AND identity LIKE ?",
                              (kind + ":%",)).fetchall()
            assert sum(status == "completed" for _, status in rows) == 100
            assert (kind + ":resolved", "resolved") in rows
        assert db.execute("SELECT status FROM strategy_callbacks WHERE identity='StockQuotes:uncertain'").fetchone()[0] == "uncertain"
        assert db.execute("SELECT COUNT(*) FROM strategy_callbacks WHERE owner='other'").fetchone()[0] == 150
        for identity in retained:
            assert db.execute("SELECT status FROM strategy_callbacks WHERE owner='s' AND identity=?", (identity,)).fetchone()[0] == "completed"


def test_large_consumed_history_does_not_exhaust_feedback_budget_on_restart(tmp_path):
    class Counter(Strategy):
        def __init__(self):
            self.fills = 0

        def on_event(self, event, context):
            if isinstance(event, FillEvent):
                self.fills += 1

    async def run():
        with OrderJournal(tmp_path / "journal.sqlite") as journal:
            for index in range(1001):
                identity = f"historical-{index}"
                journal.intent(StockOrder(identity, "s", "A", "BUY", 1, D(100)), "DU123", index + 1, 71, 123)
                journal.status(identity, "Cancelled", D(0))
            async with bridge(FakeBroker(journal), {"s": Counter()}):
                pass  # Baseline completed historical orders at first deployment.
            journal.intent(StockOrder("new-fill", "s", "A", "BUY", 1, D(100)), "DU123", 2000, 71, 123)
            journal.execution(Execution("downtime-fill", "new-fill", D(1), D(100), "BUY", now().isoformat(), D(".35"), "USD"))
            journal.status("new-fill", "Filled", D(1))
            strategy = Counter()
            async with bridge(FakeBroker(journal), {"s": strategy}) as runtime:
                assert not runtime.paused
                assert strategy.fills == 1
    asyncio.run(run())


def test_finite_prequeued_broker_backlog_does_not_count_as_callback_feedback(tmp_path):
    async def run():
        with OrderJournal(tmp_path / "journal.sqlite") as journal:
            journal.intent(StockOrder("known", "s", "A", "BUY", 1, D(100)), "DU123", 1, 71, 123)
            journal.status("known", "Cancelled", D(0))
            broker = FakeBroker(journal)
            async with bridge(broker, {"s": Strategy()}) as runtime:
                assert broker.on_update is not None
                for _ in range(1001):
                    broker.on_update(journal.get("known"))
                result = await runtime.refresh()
                assert result.successful and not runtime.paused
    asyncio.run(run())


def test_scoped_timers_keep_healthy_delivery_independent_of_paused_owner(tmp_path):
    class Broken(Strategy):
        def on_event(self, event, context):
            if isinstance(event, Timer):
                raise RuntimeError("broken decision")

    class Healthy(Strategy):
        def __init__(self):
            self.timers = []

        def on_event(self, event, context):
            if isinstance(event, Timer):
                self.timers.append(event)

    async def run():
        with OrderJournal(tmp_path / "journal.sqlite") as journal:
            healthy = Healthy()
            async with bridge(FakeBroker(journal), {"bad": Broken(), "good": healthy}) as runtime:
                first = Timer(now(), "rebalance")
                with pytest.raises(DeliveryError, match="bad"):
                    await runtime.step(first)
                assert healthy.timers == [first]
                second = Timer(now(), "rebalance")
                result = await runtime.step(second, owners=("good",))
                assert result.successful and result.errors == ()
                assert healthy.timers == [first, second]
                assert runtime.paused == {"bad"}
                assert runtime.state.unresolved("bad")
                with pytest.raises(DeliveryError, match="bad"):
                    await runtime.step(second, owners=("bad",))
                for owners in ((), ("missing",), ("good", "good"), "good"):
                    with pytest.raises(ValueError):
                        await runtime.step(Timer(now(), "invalid"), owners=owners)
    asyncio.run(run())
