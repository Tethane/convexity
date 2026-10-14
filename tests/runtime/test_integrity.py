"""Public runtime boundaries preserve causality, intent and the cash ledger."""

from dataclasses import replace
from decimal import Decimal as D
from datetime import datetime, timedelta, timezone

import pytest

from convexity.backtest import SimulatedExchange
from convexity.runtime import (
    Bar,
    Bars,
    Fill,
    FillEvent,
    Instrument,
    OrderUpdate,
    Runtime,
    Strategy,
    SubmitOrder,
    Timer,
)

T = datetime(2026, 1, 2, tzinfo=timezone.utc)
H = timedelta(hours=1)


def runtime():
    instruments = {"A": Instrument("A")}
    result = Runtime({"s": Strategy()}, SimulatedExchange(instruments), instruments)
    result.start(T)
    result.advance(
        T + H,
        Bars(T + H, {"A": Bar("A", T, T + H, D(100), D(100), D(100), D(100), 100)}),
    )
    return result


def test_future_or_replayed_bars_cannot_enter_observed_history():
    r = runtime()
    assert r.time is not None
    future = Bars(
        T + 3 * H,
        {"A": Bar("A", T + 2 * H, T + 3 * H, D(200), D(200), D(200), D(200), 100)},
    )
    with pytest.raises(ValueError, match="clock"):
        r.advance(T + 2 * H, future)
    with pytest.raises(ValueError, match="overlap"):
        r.advance(T + H, Bars(T + H, {"A": r.history["A"][0]}))
    assert r.time == T + H
    assert r.marks["A"] == 100
    assert len(r.history["A"]) == 1


def test_misdated_timers_fail_before_runtime_mutation():
    r = runtime()
    assert r.time is not None
    with pytest.raises(ValueError, match="Timer"):
        r.advance(T + 2 * H, timers=(Timer(T + 3 * H, "future"),))
    assert r.time == T + H


def test_unknown_owner_cannot_consume_capacity():
    r = runtime()
    assert r.time is not None
    with pytest.raises(ValueError, match="registered"):
        r.command("ghost", SubmitOrder("A", "BUY", 10))
    assert not r.orders


def test_order_updates_cannot_rewrite_intent_or_fake_accounted_fills():
    r = runtime()
    assert r.time is not None
    r.command("s", SubmitOrder("A", "BUY", 10))
    order = next(iter(r.orders.values()))
    for incoming in (
        replace(order, request=SubmitOrder("A", "SELL", 10)),
        replace(order, filled_quantity=5, status="partially_filled"),
    ):
        with pytest.raises(ValueError):
            r.apply(OrderUpdate(r.time, incoming))
    assert r.cash == 100_000 and not r.positions
    assert r.orders[order.order_id] == order
    r.apply(OrderUpdate(r.time, replace(order, status="canceled")))
    with pytest.raises(ValueError, match="terminal"):
        r.apply(OrderUpdate(r.time, order))


def test_invalid_fill_is_rejected_before_accounting_and_duplicate_is_idempotent():
    r = runtime()
    assert r.time is not None
    r.command("s", SubmitOrder("A", "BUY", 10))
    order = next(iter(r.orders.values()))
    args = ("exec", order.order_id, "s", "A", "BUY")
    for quantity, price, fee in ((1.5, 100, 0), (1, "Infinity", 0), (1, 100, -1)):
        with pytest.raises(ValueError):
            Fill(*args, quantity, price, fee, r.time, r.time)  # pyright: ignore[reportArgumentType] -- deliberately invalid inputs
    fill = Fill(*args, 5, D(100), D(1), r.time, r.time)
    with pytest.raises(ValueError, match="timestamps"):
        r.apply(FillEvent(r.time, replace(fill, occurred_at=T)))
    r.apply(FillEvent(r.time, fill))
    r.apply(FillEvent(r.time, fill))
    assert r.cash == 99_499
    assert r.account_positions["A"] == 5
    assert len(r.fills) == 1


def test_completed_orders_leave_active_books_but_keep_audit_and_identity():
    r = runtime()
    assert isinstance(r.exchange, SimulatedExchange)
    for index in range(1, 21):
        r.command("s", SubmitOrder("A", "BUY", 1, client_order_id=str(index)))
        begin, end = T + index * H, T + (index + 1) * H
        r.advance(
            end,
            Bars(end, {"A": Bar("A", begin, end, D(100), D(100), D(100), D(100), 100)}),
        )
        assert not r._active_orders
        assert not r.exchange._working
    assert len(r.orders) == len(r.fills) == 20
    assert len(r.context("s").orders) == 20
    order = r.orders["s:1"]
    with pytest.raises(ValueError, match="identity already exists"):
        r.exchange.submit(order)
    r.command("s", order.request)
    assert len(r.orders) == 20


def test_cancel_removes_active_capacity_and_keeps_order_history():
    from convexity.runtime import CancelOrder

    r = runtime()
    r.command("s", SubmitOrder("A", "BUY", 1))
    order = next(iter(r.orders.values()))
    r.command("s", CancelOrder(order.order_id))
    assert isinstance(r.exchange, SimulatedExchange)
    assert not r._active_orders and not r.exchange._working
    assert r.orders[order.order_id].status == "canceled"


def test_context_orders_preserve_owner_ordering_and_immutable_snapshots():
    instruments = {"A": Instrument("A")}
    r = Runtime(
        {"a": Strategy(), "b": Strategy()}, SimulatedExchange(instruments), instruments
    )
    r.start(T)
    r.advance(
        T + H,
        Bars(T + H, {"A": Bar("A", T, T + H, D(100), D(100), D(100), D(100), 100)}),
    )
    r.command("a", SubmitOrder("A", "BUY", 1))
    r.command("b", SubmitOrder("A", "BUY", 1))
    old = r.context("a")
    r.advance(
        T + 2 * H,
        Bars(
            T + 2 * H,
            {"A": Bar("A", T + H, T + 2 * H, D(100), D(100), D(100), D(100), 100)},
        ),
    )
    assert not old.orders[0].terminal
    assert len(old.history["A"]) == 1
    for owner in r.strategies:
        assert r.context(owner).orders == tuple(
            order for order in r.orders.values() if order.strategy_id == owner
        )
        assert r.context(owner).orders[0].status == "filled"
