from datetime import datetime, timedelta, timezone
from decimal import Decimal

from convexity.backtest import SimulatedExchange
from convexity.runtime import Bar, Bars, Context, Instrument, Order, Position, Runtime, Strategy, SubmitOrder
from convexity.strategies import MonthlyLongShort, StockRule, rebalance_orders

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def context(positions=None, orders=(), history=None):
    return Context(NOW, "s", Decimal(1000), Decimal(1000), positions or {}, {},
                   {"A": Decimal(10), "B": Decimal(20)}, {}, orders, history or {})


def test_target_minus_current_pending_and_reversal():
    c = context({"A": Position(10), "B": Position(-5)})
    assert rebalance_orders(c, {"A": -4, "B": 3}) == (
        SubmitOrder("A", "SELL", 10), SubmitOrder("B", "BUY", 5))
    pending = Order("o", "s", SubmitOrder("A", "BUY", 3), NOW, 1)
    assert rebalance_orders(context({"A": Position(2)}, (pending,)), {"A": 5}) == ()


def test_repeat_after_routing_does_not_duplicate():
    instruments = {"A": Instrument("A")}
    runtime = Runtime({"s": Strategy()}, SimulatedExchange(instruments), instruments, 1000)
    runtime.start(NOW)
    bar = Bar("A", NOW, NOW + timedelta(minutes=1), Decimal(10), Decimal(10), Decimal(10), Decimal(10), 100)
    runtime.advance(bar.end, Bars(bar.end, {"A": bar}))
    for command in rebalance_orders(runtime.context("s"), {"A": 5}):
        runtime.command("s", command)
    assert rebalance_orders(runtime.context("s"), {"A": 5}) == ()
    assert len(runtime.orders) == 1


def test_rule_worked_example_and_short_skip():
    history = {s: tuple(Bar(s, NOW + timedelta(days=i), NOW + timedelta(days=i, hours=1),
                           Decimal(p), Decimal(p), Decimal(p), Decimal(p), 100) for i, p in enumerate(prices))
               for s, prices in {"A": (8, 9, 10), "B": (25, 23, 20)}.items()}
    c = context(history=history)
    strategy = MonthlyLongShort(StockRule(("A", "B")))
    assert strategy.target(c) == {"A": 30, "B": -15}
    unavailable = MonthlyLongShort(StockRule(("A", "B"), shortable=("A",)))
    assert unavailable.target(c) is None
    assert "short unavailable" in unavailable.last_decision


def test_exit_cancels_openings_but_allows_reductions_to_rest_across_restart():
    from dataclasses import replace
    from convexity.runtime import CancelOrder, Timer

    rule = StockRule(("A", "B"), exit_timer="exit", exit_retry_seconds=20)
    opening = Order("opening", "s", SubmitOrder("B", "SELL", 2), NOW, 1)
    reducing = Order("closing", "s", SubmitOrder("A", "SELL", 5), NOW, 2,
                     "partially_filled", 2)
    c = context({"A": Position(3)}, (opening, reducing))
    c = replace(c, time=NOW + timedelta(seconds=2))
    assert MonthlyLongShort(rule).on_event(Timer(c.time, "exit"), c) == (CancelOrder("opening"),)
    # A new strategy instance uses the persisted acceptance time, not an in-memory retry clock.
    c = replace(c, time=NOW + timedelta(seconds=20), orders=(reducing,))
    assert MonthlyLongShort(rule).on_event(Timer(c.time, "exit"), c) == (CancelOrder("closing"),)
    c = replace(c, orders=(replace(reducing, cancel_pending=True),))
    assert MonthlyLongShort(rule).on_event(Timer(c.time, "exit"), c) == ()
    c = replace(c, orders=(replace(reducing, status="canceled"),))
    assert MonthlyLongShort(rule).on_event(Timer(c.time, "exit"), c) == (SubmitOrder("A", "SELL", 3),)


def test_exit_does_not_allow_crossing_flat_order_to_rest():
    from convexity.runtime import CancelOrder, Timer

    crossing = Order("crossing", "s", SubmitOrder("A", "SELL", 6), NOW, 1)
    c = context({"A": Position(3)}, (crossing,))
    strategy = MonthlyLongShort(StockRule(("A", "B"), exit_timer="exit"))
    assert strategy.on_event(Timer(NOW, "exit"), c) == (CancelOrder("crossing"),)


def test_exit_retry_setting_rejects_unbounded_values():
    import pytest

    for value in (0, -1, 301, float("nan"), float("inf"), True):
        with pytest.raises(ValueError, match="exit_retry_seconds"):
            StockRule(("A", "B"), exit_retry_seconds=value)
