"""Deterministic command lifecycle, contingent execution and amendment regressions."""
from datetime import datetime, timedelta, timezone
from decimal import Decimal as D

from convexity.backtest.exchange import ExecutionModel, SimulatedExchange
from convexity.runtime.engine import Runtime
from convexity.runtime.types import (Bar, Bars, BracketOrder, CancelOrder, CancelOrders, Instrument,
    OCOOrders, OrderUpdate, ReplaceOrder, Rejection, Strategy, SubmitOrder)

T = datetime(2024, 1, 2, 14, 30, tzinfo=timezone.utc)


def runtime(*, model=None, strategies=None):
    instruments = {"A": Instrument("A"), "B": Instrument("B")}
    result = Runtime(strategies or {"owner": Strategy()}, SimulatedExchange(instruments, model), instruments,
                     initial_cash=100_000, session_closes=(T + timedelta(hours=6),))
    result.marks.update(A=D(100), B=D(100))
    result.start(T)
    return result


def advance(runtime, *, opening=100, high=110, low=90, close=100, volume=100):
    start = runtime.time
    end = start + timedelta(minutes=1)
    runtime.advance(end, Bars(end, {symbol: Bar(symbol, start, end, D(opening), D(high), D(low), D(close), volume)
                                    for symbol in ("A", "B")}))


def test_stop_market_gaps_and_intrabar_trigger_have_explicit_timing():
    engine = runtime()
    engine.command("owner", SubmitOrder.stop("A", "BUY", 1, 105, time_in_force="GTC"))
    advance(engine, opening=108, high=110, low=100, close=108)
    assert (engine.fills[0].price, engine.fills[0].occurred_at) == (108, T)
    engine.command("owner", SubmitOrder.stop("A", "SELL", 1, 95, time_in_force="GTC"))
    advance(engine)
    assert engine.fills[-1].price == 95
    assert engine.fills[-1].occurred_at == engine.time


def test_stop_limit_does_not_assume_order_of_intrabar_high_and_low():
    engine = runtime()
    engine.command("owner", SubmitOrder.stop_limit("A", "BUY", 1, 105, 104, time_in_force="GTC"))
    advance(engine, opening=100, high=110, low=99)
    assert not engine.fills
    advance(engine, opening=106, high=110, low=105, close=106)
    assert not engine.fills
    advance(engine, opening=103, high=105, low=100, close=103)
    assert engine.fills[0].price == 103


def test_replacement_resets_fifo_preserves_id_and_deduplicates_command():
    engine = runtime()
    for identity in ("first", "second"):
        engine.command("owner", SubmitOrder.limit("A", "BUY", 2, 100, time_in_force="GTC", client_order_id=identity))
    command = ReplaceOrder("owner:first", limit_price=D(101), client_command_id="revision-one")
    engine.command("owner", command)
    changed = engine.orders["owner:first"]
    engine.command("owner", command)
    assert engine.orders["owner:first"] == changed
    assert changed.revision == 1
    assert not engine.fills
    advance(engine, volume=1)
    assert engine.fills[0].order_id == "owner:second"
    engine.command("owner", ReplaceOrder("owner:first", limit_price=D(102), client_command_id="revision-one"))
    assert isinstance(engine.events[-1], Rejection)
    assert engine.orders["owner:first"] == changed


def test_replacement_total_quantity_retains_executions_and_cannot_reduce_below_them():
    engine = runtime()
    engine.command("owner", SubmitOrder("A", "BUY", 5, client_order_id="order"))
    advance(engine, volume=2)
    engine.command("owner", ReplaceOrder("owner:order", quantity=1))
    assert isinstance(engine.events[-1], Rejection)
    engine.command("owner", ReplaceOrder("owner:order", quantity=3))
    assert engine.orders["owner:order"].filled_quantity == 2
    assert engine.orders["owner:order"].remaining == 1
    assert len(engine.fills) == 1
    advance(engine)
    assert [fill.quantity for fill in engine.fills] == [2, 1]
    assert engine.orders["owner:order"].status == "filled"


def test_pending_cancel_rejects_replace_but_acknowledged_fill_can_win_cancel_race():
    engine = runtime(model=ExecutionModel(cancel_delay_bars=1))
    engine.command("owner", SubmitOrder("A", "BUY", 1, client_order_id="order"))
    engine.command("owner", CancelOrder("owner:order"))
    engine.command("owner", ReplaceOrder("owner:order", quantity=2))
    assert isinstance(engine.events[-1], Rejection)
    advance(engine)
    assert engine.orders["owner:order"].status == "filled"
    assert engine.orders["owner:order"].revision == 0


def test_cancel_orders_snapshot_filters_only_owned_orders():
    engine = runtime(strategies={"owner": Strategy(), "other": Strategy()})
    for owner, symbol, side, identity in (("owner", "A", "BUY", "a"), ("owner", "B", "BUY", "b"),
                                          ("owner", "A", "SELL", "sell"), ("other", "A", "BUY", "a")):
        engine.command(owner, SubmitOrder(symbol, "BUY" if side == "BUY" else "SELL", 1, client_order_id=identity))
    engine.command("owner", CancelOrders(symbol="A", side="BUY"))
    assert engine.orders["owner:a"].status == "canceled"
    assert all(engine.orders[key].status == "working" for key in ("owner:b", "owner:sell", "other:a"))


def test_bracket_children_wait_for_full_parent_and_ambiguous_bar_chooses_stop():
    engine = runtime()
    command = BracketOrder(SubmitOrder("A", "BUY", 2), take_profit=D(105), stop_loss=D(95), client_order_id="trade")
    engine.command("owner", command)
    engine.command("owner", command)
    assert len(engine.orders) == 3
    assert not engine.orders["owner:trade:stop_loss"].active
    advance(engine, volume=1)
    assert len(engine.fills) == 1
    assert not engine.orders["owner:trade:stop_loss"].active
    advance(engine, volume=100)
    assert len(engine.fills) == 2  # No exit from the parent's own completion interval.
    assert engine.orders["owner:trade:stop_loss"].active
    advance(engine)
    assert engine.fills[-1].order_id == "owner:trade:stop_loss"
    assert engine.fills[-1].price == 95
    assert engine.orders["owner:trade:take_profit"].status == "canceled"
    assert engine.positions[("owner", "A")].quantity == 0


def test_canceled_partial_bracket_parent_leaves_observable_unprotected_position():
    engine = runtime()
    engine.command("owner", BracketOrder(SubmitOrder("A", "BUY", 3), D(105), D(95), "trade"))
    advance(engine, volume=1)
    engine.command("owner", CancelOrder("owner:trade:entry"))
    assert engine.positions[("owner", "A")].quantity == 1
    assert all(order.status == "canceled" for order in engine.orders.values())
    advance(engine)
    assert len(engine.fills) == 1


def test_oco_cancels_peer_on_partial_execution_and_winning_residual_remains():
    engine = runtime()
    engine.command("owner", OCOOrders((SubmitOrder.limit("A", "BUY", 3, 100, time_in_force="GTC"),
                                        SubmitOrder.limit("A", "BUY", 3, 99, time_in_force="GTC")), "entry"))
    advance(engine, volume=1)
    assert engine.orders["owner:entry:0"].status == "partially_filled"
    assert engine.orders["owner:entry:1"].status == "canceled"
    advance(engine)
    assert sum(fill.quantity for fill in engine.fills) == 3


def test_group_validation_is_atomic_and_bracket_quantity_amendment_rejected():
    engine = runtime()
    engine.command("owner", OCOOrders((SubmitOrder("A", "BUY", 1),
                                        SubmitOrder.limit("A", "BUY", 1, "99.001", time_in_force="GTC")), "bad"))
    assert not engine.orders
    assert isinstance(engine.events[-1], Rejection)
    engine.command("owner", BracketOrder(SubmitOrder("A", "BUY", 3), D(105), D(95), "trade"))
    engine.command("owner", ReplaceOrder("owner:trade:entry", quantity=2))
    assert isinstance(engine.events[-1], Rejection)
    assert engine.orders["owner:trade:entry"].request.quantity == 3


def test_triggered_stop_limit_remains_triggered_after_quantity_amendment():
    engine = runtime()
    engine.command("owner", SubmitOrder.stop_limit("A", "BUY", 2, 105, 104, time_in_force="GTC", client_order_id="stop"))
    advance(engine, opening=100, high=110, low=99)
    assert not engine.fills
    engine.command("owner", ReplaceOrder("owner:stop", quantity=3))
    advance(engine, opening=103, high=104, low=100, close=103)
    assert engine.fills[0].quantity == 3
    assert engine.fills[0].price == 103


def test_inactive_bracket_child_cancel_latency_can_complete_before_entry_fill():
    engine = runtime(model=ExecutionModel(cancel_delay_bars=1))
    engine.command("owner", BracketOrder(SubmitOrder.limit("A", "BUY", 1, 50, time_in_force="GTC"), D(105), D(45), "trade"))
    engine.command("owner", CancelOrder("owner:trade:take_profit"))
    advance(engine)
    advance(engine)
    assert engine.orders["owner:trade:take_profit"].status == "canceled"
    assert engine.orders["owner:trade:entry"].status == "working"


def test_bracket_exits_do_not_require_short_capacity():
    engine = runtime(model=ExecutionModel(allow_short=False))
    engine.command("owner", BracketOrder(SubmitOrder("A", "BUY", 2), D(105), D(95), "trade"))
    assert len(engine.orders) == 3
    advance(engine)
    assert len(engine.fills) == 1
    advance(engine)
    assert len(engine.fills) == 2
    assert engine.positions[("owner", "A")].quantity == 0


def test_amended_bracket_stop_retains_conservative_ambiguity_resolution():
    engine = runtime()
    engine.command("owner", BracketOrder(SubmitOrder("A", "BUY", 1), D(105), D(95), "trade"))
    advance(engine)
    engine.command("owner", ReplaceOrder("owner:trade:stop_loss", stop_price=D(96)))
    advance(engine)
    assert engine.fills[-1].order_id == "owner:trade:stop_loss"
    assert engine.fills[-1].price == 96


def test_bracket_stop_limit_ambiguity_does_not_invent_profitable_exit():
    engine = runtime()
    engine.command("owner", BracketOrder(SubmitOrder("A", "BUY", 1), D(105), D(95), "trade", stop_limit_price=D(94)))
    advance(engine)
    advance(engine)
    assert len(engine.fills) == 1
    assert engine.orders["owner:trade:take_profit"].status == "working"
    advance(engine)
    assert engine.fills[-1].order_id == "owner:trade:stop_loss"
