"""Stock command semantics shared by simulation and the paper adapter."""
import asyncio
from dataclasses import replace
from decimal import Decimal as D

import pytest

from convexity.runtime import CancelOrder, FillEvent, OrderUpdate, Rejection, Strategy, SubmitOrder, Timer
from convexity.trading import OrderJournal
from tests.runtime.test_integrity import runtime
from tests.runtime.test_paper import FakeBroker, bridge, now


class Commands(Strategy):
    def __init__(self, command):
        self.command: SubmitOrder | CancelOrder = command
        self.received = []

    def on_event(self, event, context):
        self.received.append(event)
        if isinstance(event, Timer):
            return self.command


@pytest.mark.parametrize("kind", ["limit", "market"])
@pytest.mark.parametrize("tif", ["DAY", "GTC"])
def test_paper_preserves_original_command_public_identity_and_tif(tmp_path, kind, tif):
    command = SubmitOrder("A", "BUY", 1, kind, D(99) if kind == "limit" else None, tif, "entry")
    async def run():
        with OrderJournal(tmp_path / "orders.sqlite") as journal:
            broker, strategy = FakeBroker(journal), Commands(command)
            async with bridge(broker, {"s": strategy}) as live:
                await live.step(Timer(now(), "entry"))
                order = live.context("s").orders[0]
                assert order.order_id == "s:entry"
                assert order.request == command
                assert broker.sent[0].time_in_force == tif
                strategy.command = CancelOrder("s:entry")
                await live.step(Timer(now(), "cancel"))
                assert live.context("s").orders[0].status == "canceled"
                assert not live.paused
    asyncio.run(run())


@pytest.mark.parametrize("changed", [
    SubmitOrder("A", "BUY", 1, client_order_id="entry"),
    SubmitOrder("A", "BUY", 1, "limit", D(99), "DAY", "entry"),
    SubmitOrder("A", "BUY", 1, "limit", D(98), "GTC", "entry"),
])
def test_complete_command_conflicts_rejected_across_journal_reopen(tmp_path, changed):
    initial = SubmitOrder("A", "BUY", 1, "limit", D(99), "GTC", "entry")
    path = tmp_path / "orders.sqlite"
    async def run():
        with OrderJournal(path) as journal:
            async with bridge(FakeBroker(journal), {"s": Commands(initial)}) as live:
                await live.step(Timer(now(), "first"))
        with OrderJournal(path) as journal:
            strategy, broker = Commands(changed), FakeBroker(journal)
            async with bridge(broker, {"s": strategy}) as live:
                await live.step(Timer(now(), "changed"))
                assert not broker.sent
                assert not live.paused
                rejected = [event for event in strategy.received if isinstance(event, Rejection)]
                assert len(rejected) == 1 and "different intent" in rejected[0].reason
                assert live.context("s").orders[0].request == initial
    asyncio.run(run())


def test_equivalent_decimal_command_is_idempotent_and_fill_keeps_public_id(tmp_path):
    async def run():
        with OrderJournal(tmp_path / "orders.sqlite") as journal:
            broker = FakeBroker(journal)
            strategy = Commands(SubmitOrder("A", "BUY", 1, "limit", D("99.0"), client_order_id="entry"))
            async with bridge(broker, {"s": strategy}) as live:
                await live.step(Timer(now(), "first"))
                assert isinstance(strategy.command, SubmitOrder)
                strategy.command = replace(strategy.command, limit_price=D("99.00"))
                await live.step(Timer(now(), "retry"))
                assert len(broker.sent) == 1
                assert not any(isinstance(event, Rejection) for event in strategy.received)
                broker.fill(1, commission=D(".35"))
                await live.refresh()
                assert [event.fill.order_id for event in strategy.received if isinstance(event, FillEvent)] == ["s:entry"]
    asyncio.run(run())


def test_unknown_cancel_is_rejection_without_pausing(tmp_path):
    async def run():
        with OrderJournal(tmp_path / "orders.sqlite") as journal:
            strategy = Commands(CancelOrder("s:missing"))
            async with bridge(FakeBroker(journal), {"s": strategy}) as live:
                await live.step(Timer(now(), "cancel"))
                assert not live.paused
                assert len([event for event in strategy.received if isinstance(event, Rejection)]) == 1
    asyncio.run(run())


def test_simulation_risk_rejection_has_shared_command_event():
    live = runtime()
    command = SubmitOrder("UNKNOWN", "BUY", 1, client_order_id="entry")
    live.command("s", command)
    rejected = [event for event in live.events if isinstance(event, Rejection)]
    assert len(rejected) == 1 and rejected[0].command == command
    assert live.orders["s:entry"].status == "rejected"


def test_broker_rejection_emits_shared_event_once(tmp_path):
    async def run():
        with OrderJournal(tmp_path / "orders.sqlite") as journal:
            broker = FakeBroker(journal)
            command = SubmitOrder("A", "BUY", 1, client_order_id="entry")
            strategy = Commands(command)
            async with bridge(broker, {"s": strategy}) as live:
                await live.step(Timer(now(), "first"))
                identity = broker.sent[0].client_order_id
                journal.status(identity, "Rejected", D(0))
                assert broker.on_update is not None
                broker.on_update(journal.get(identity))
                broker.on_update(journal.get(identity))
                await live.refresh()
                rejected = [event for event in strategy.received if isinstance(event, Rejection)]
                assert len(rejected) == 1 and rejected[0].command == command
    asyncio.run(run())


def test_operator_can_cancel_and_reduce_paused_owner_but_cannot_add_risk(tmp_path):
    async def run():
        with OrderJournal(tmp_path / "orders.sqlite") as journal:
            broker = FakeBroker(journal)
            async with bridge(broker, {"s": Strategy()}) as live:
                await live.command("s", SubmitOrder("A", "BUY", 2, client_order_id="entry"))
                broker.fill(1, commission=D(".35"))
                await live.refresh()
                live.paused.add("s")
                events = await live.command("s", SubmitOrder("A", "BUY", 1, client_order_id="increase"))
                assert any(isinstance(event, Rejection) for event in events)
                assert len(broker.sent) == 1
                await live.command("s", CancelOrder("s:entry"))
                await live.command("s", SubmitOrder("A", "SELL", 1, client_order_id="exit"))
                assert len(broker.sent) == 2 and broker.sent[-1].side == "SELL"
                assert live.paused == {"s"}
    asyncio.run(run())


@pytest.mark.parametrize("expired", ["decision", "quote"])
def test_real_broker_transmit_guard_rechecks_after_contract_qualification(tmp_path, monkeypatch, expired):
    from datetime import timedelta
    from convexity.brokers import ibkr_trading
    from convexity.runtime import paper as paper_module
    from tests.brokers.test_ibkr import config
    from tests.brokers.test_ibkr_trading import TradingClient

    stamp = now()
    clock = [stamp]
    stale = [False]
    class Clock:
        @staticmethod
        def now(tz=None):
            return clock[0]
    monkeypatch.setattr(paper_module, "datetime", Clock)

    class DelayedClient(TradingClient):
        async def reqContractDetailsAsync(self, stock):
            await asyncio.sleep(0)
            stale[0] = True
            if expired == "quote":
                clock[0] += timedelta(seconds=31)
            return self.details

    class Broker(ibkr_trading.PaperBroker):
        def __init__(self, config, journal):
            super().__init__(config, journal)
            self.data = FakeBroker(journal)

        async def quote(self, request):
            value = await self.data.quote(request)
            return replace(value, bid_received_at=stamp, ask_received_at=stamp, completed_at=stamp)

        async def snapshot(self):
            return await self.data.snapshot()

    def decision_guard():
        if expired == "decision" and stale[0]:
            raise RuntimeError("Decision history expired during qualification")

    async def run():
        with OrderJournal(tmp_path / "orders.sqlite") as journal:
            client = DelayedClient(journal)
            monkeypatch.setattr(ibkr_trading, "_new_client", lambda: client)
            broker = Broker(config(), journal)
            strategy = Commands(SubmitOrder("A", "BUY", 1, client_order_id="entry"))
            async with bridge(broker, {"s": strategy}, decision_guard=decision_guard) as live:
                if expired == "decision":
                    from convexity.runtime import DeliveryError
                    with pytest.raises(DeliveryError, match="expired"):
                        await live.step(Timer(stamp, "entry"))
                else:
                    await live.step(Timer(stamp, "entry"))
                assert stale[0], "The check must reach awaited broker contract qualification"
                assert not client.sent
                assert not journal.orders(), "Rejected preflight must not create an uncertain broker intent"
                assert any("expired" in error for error in live.errors)
    asyncio.run(run())


def test_operator_entry_obeys_decision_guard_but_verified_exit_can_bypass(tmp_path):
    def stale_decisions():
        raise RuntimeError("Decision feed stale")

    async def run():
        with OrderJournal(tmp_path / "orders.sqlite") as journal:
            broker = FakeBroker(journal)
            async with bridge(broker, {"s": Strategy()}) as live:
                await live.command("s", SubmitOrder("A", "BUY", 1, client_order_id="entry"))
                broker.fill(1, commission=D(".35"))
                await live.refresh()
                live.decision_guard = stale_decisions
                with pytest.raises(RuntimeError, match="stale"):
                    await live.command("s", SubmitOrder("A", "BUY", 1, client_order_id="increase"))
                assert len(broker.sent) == 1
                live.paused.add("s")
                await live.command("s", SubmitOrder("A", "SELL", 1, client_order_id="exit"))
                assert len(broker.sent) == 2 and broker.sent[-1].side == "SELL"
    asyncio.run(run())
