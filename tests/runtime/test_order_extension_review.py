"""Independent adversarial checks for amendments, contingent orders and subscriptions."""
import asyncio
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal as D
from typing import Any, cast

import pytest

from convexity.backtest.exchange import SimulatedExchange
from convexity.runtime import Bar, Bars, Instrument, OCOOrders, Runtime, Strategy, SubmitOrder
from convexity.trading import OrderJournal

T = datetime(2025, 1, 2, 14, 30, tzinfo=timezone.utc)


def test_oco_execution_does_not_reserve_the_peer_it_atomically_cancels():
    instruments = {"A": Instrument("A")}
    runtime = Runtime({"owner": Strategy()}, SimulatedExchange(instruments), instruments, initial_cash=1000)
    runtime.marks["A"] = D(100)
    runtime.start(T)
    runtime.command("owner", OCOOrders((
        SubmitOrder.limit("A", "BUY", 10, 100, time_in_force="GTC"),
        SubmitOrder.limit("A", "BUY", 10, 99, time_in_force="GTC")), "entry"))
    end = T + timedelta(minutes=1)
    runtime.advance(end, Bars(end, {"A": Bar("A", T, end, D(100), D(100), D(99), D(100), 100)}))
    assert [(fill.order_id, fill.quantity) for fill in runtime.fills] == [("owner:entry:0", 10)]
    assert runtime.orders["owner:entry:1"].status == "canceled"
    assert runtime.cash == 0


def test_rejected_modification_requires_observed_terms_not_status_echo(tmp_path, monkeypatch):
    from convexity.brokers import ibkr_trading
    from tests.brokers.test_ibkr import config
    from tests.brokers.test_ibkr_order_management import ManagedClient, stock

    async def exercise():
        with OrderJournal(tmp_path / "journal.sqlite3") as journal:
            client = ManagedClient(journal)
            monkeypatch.setattr(ibkr_trading, "_new_client", lambda: client)
            async with ibkr_trading.PaperBroker(config(), journal) as broker:
                await broker.submit(stock())
                client.acknowledge("order")
                await broker.replace("order", replace(stock(), limit_price=D(101)), command_id="amend")
                cast(Any, client).errorEvent.emit(10, 201, "Modification rejected", None)
                assert journal.get("order").uncertain
                client.acknowledge("order", terms=False)
                assert journal.get("order").uncertain
                assert journal.get("order").request.limit_price == 100
    asyncio.run(exercise())


def test_dynamic_same_name_clock_pool_does_not_cross_calendar_holidays():
    from convexity.runtime import ClockSubscription, StrategyContract, Timer
    from convexity.runtime.subscriptions import SubscriptionBook
    us = ClockSubscription("trade", "10:00", calendar="XNYS")
    uk = ClockSubscription("trade", "10:00", calendar="XLON")
    book = SubscriptionBook(StrategyContract((), (us,), available_subscriptions=(uk,)), "owner")
    uk_holiday_event = Timer(datetime(2025, 7, 4, 14, 0, tzinfo=timezone.utc), "trade")
    assert uk_holiday_event in uk.materialize(2025).timers
    assert uk_holiday_event not in us.materialize(2025).timers
    assert book.select(uk_holiday_event) is None


def test_paper_oco_replacement_keeps_existing_group_risk_reservation(tmp_path):
    from convexity.runtime import ReplaceOrder, Rejection
    from tests.runtime.test_paper import bridge
    from tests.runtime.test_paper_order_management import ManagedBroker

    async def exercise():
        with OrderJournal(tmp_path / "journal.sqlite3") as journal:
            broker = ManagedBroker(journal)
            async with bridge(broker, {"owner": Strategy()}) as runtime:
                command = OCOOrders((SubmitOrder.limit("A", "BUY", 9, 100),
                                     SubmitOrder.limit("A", "BUY", 9, 99)), "entry")
                events = await runtime.command("owner", command)
                assert not any(isinstance(event, Rejection) for event in events)
                events = await runtime.command("owner", ReplaceOrder("owner:entry:0", limit_price=D(98), client_command_id="lower"))
                assert not any(isinstance(event, Rejection) for event in events)
                assert next(order for order in runtime.context("owner").orders if order.order_id == "owner:entry:0").request.limit_price == 98
                replay = await runtime.command("owner", command)
                assert not any(isinstance(event, Rejection) for event in replay)
                assert len(broker.sent) == 2
    asyncio.run(exercise())


def test_replacement_equal_to_executions_requires_cancel_instead_of_fabricating_fill():
    from convexity.runtime import ReplaceOrder, Rejection
    instruments = {"A": Instrument("A")}
    runtime = Runtime({"owner": Strategy()}, SimulatedExchange(instruments), instruments)
    runtime.marks["A"] = D(100)
    runtime.start(T)
    runtime.command("owner", SubmitOrder("A", "BUY", 3, client_order_id="entry"))
    end = T + timedelta(minutes=1)
    runtime.advance(end, Bars(end, {"A": Bar("A", T, end, D(100), D(100), D(100), D(100), 1)}))
    runtime.command("owner", ReplaceOrder("owner:entry", quantity=1))
    assert isinstance(runtime.events[-1], Rejection)
    assert "cancel" in runtime.events[-1].reason.lower()
    assert runtime.orders["owner:entry"].status == "partially_filled"
    assert runtime.orders["owner:entry"].request.quantity == 3
    assert len(runtime.fills) == 1


def test_paper_oco_replacement_cannot_hide_unresolved_peer(tmp_path):
    from convexity.runtime import ReplaceOrder, Rejection
    from tests.runtime.test_paper import bridge
    from tests.runtime.test_paper_order_management import ManagedBroker

    async def exercise():
        with OrderJournal(tmp_path / "journal.sqlite3") as journal:
            broker = ManagedBroker(journal)
            async with bridge(broker, {"owner": Strategy()}) as runtime:
                await runtime.command("owner", OCOOrders((SubmitOrder.limit("A", "BUY", 1, 100),
                                                         SubmitOrder.limit("A", "BUY", 1, 99)), "entry"))
                peer = journal.broker_order_id("DU123", "owner:entry:1")
                journal.uncertain(peer, "Pending broker reconciliation")
                events = await runtime.command("owner", ReplaceOrder("owner:entry:0", limit_price=D(98), client_command_id="lower"))
                assert any(isinstance(event, Rejection) for event in events)
                first = next(order for order in runtime.context("owner").orders if order.order_id == "owner:entry:0")
                assert first.request.limit_price == 100
    asyncio.run(exercise())


@pytest.mark.parametrize("cash,maximum", [(10000, D(1000)), (1000, None)])
def test_stop_trigger_price_is_reserved_before_submission(cash, maximum):
    from convexity.runtime import RiskLimits
    instruments = {"A": Instrument("A")}
    runtime = Runtime({"owner": Strategy()}, SimulatedExchange(instruments), instruments,
                      initial_cash=cash, limits=RiskLimits(max_order_notional=maximum))
    runtime.marks["A"] = D(100)
    runtime.start(T)
    runtime.command("owner", SubmitOrder.stop("A", "BUY", 10, 105, time_in_force="GTC", client_order_id="entry"))
    assert runtime.orders["owner:entry"].status == "rejected"
    assert not runtime.fills


def test_simulated_cancel_rejection_preserves_live_order_and_reason():
    from convexity.runtime import CancelOrder, Rejection
    class Denied(SimulatedExchange):
        def cancel(self, order_id, time):
            raise ValueError("Venue denied cancellation")
    instruments = {"A": Instrument("A")}
    runtime = Runtime({"owner": Strategy()}, Denied(instruments), instruments)
    runtime.marks["A"] = D(100)
    runtime.start(T)
    runtime.command("owner", SubmitOrder("A", "BUY", 1, client_order_id="entry"))
    runtime.command("owner", CancelOrder("owner:entry"))
    assert isinstance(runtime.events[-1], Rejection)
    assert "Venue denied cancellation" in runtime.events[-1].reason
    assert runtime.orders["owner:entry"].status == "working"
    assert not runtime.orders["owner:entry"].cancel_pending
