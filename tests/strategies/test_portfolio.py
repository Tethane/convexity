from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from convexity.runtime import Context, Position, SubmitOrder, Order, Timer, Bars, Bar
from convexity.strategies import (
    equal_weights,
    target_weight_orders,
    TargetWeightStrategy,
)

NOW = datetime(2024, 1, 8, 21, tzinfo=timezone.utc)


def context(
    *, positions=None, cash=1000, equity=1000, marks=None, orders=(), stale=False
):
    marks = marks or {"A": Decimal(100), "B": Decimal(100)}
    return Context(
        NOW,
        "weights",
        Decimal(cash),
        Decimal(equity),
        positions or {},
        {},
        marks,
        {s: NOW - timedelta(days=10) if stale else NOW for s in marks},
        orders,
        {},
    )


def test_equal_custom_weights_and_integer_cash_reserve():
    assert equal_weights(["A", "A", "B"]) == {"A": Decimal(".5"), "B": Decimal(".5")}
    plan = target_weight_orders(context(), {"A": 0.7, "B": 0.3}, cash_buffer=0)
    assert plan.targets == {"A": 7, "B": 3}
    assert [(o.symbol, o.quantity) for o in plan.orders] == [("A", 7), ("B", 3)]
    assert plan.turnover == 1
    assert target_weight_orders(context(), {"A": 1}).targets["A"] == 9


def test_closes_removed_positions_before_spending_proceeds():
    plan = target_weight_orders(
        context(positions={"A": Position(10)}, cash=0), {"B": 1}, cash_buffer=0
    )
    assert [(o.symbol, o.side, o.quantity) for o in plan.orders] == [("A", "SELL", 10)]
    assert plan.targets == {"A": 0, "B": 10}


def test_commission_and_cost_estimates_bound_available_cash():
    plan = target_weight_orders(
        context(), {"A": 1}, cash_buffer=0, cost_bps=100, commission_per_order=1
    )
    assert plan.orders[0].quantity == 9
    assert plan.estimated_cost == 10
    assert any("cash constraint" in value for value in plan.diagnostics)


def test_missing_stale_future_prices_reported_and_not_traded():
    assert not target_weight_orders(context(stale=True), {"A": 1}).orders
    assert "Stale" in target_weight_orders(context(stale=True), {"A": 1}).diagnostics[0]
    assert not target_weight_orders(context(), {"MISSING": 1}).orders
    future = context()
    from dataclasses import replace

    future = replace(future, mark_times={"A": NOW + timedelta(seconds=1)})
    assert "Future" in target_weight_orders(future, {"A": 1}).diagnostics[0]


def test_working_orders_prevent_duplicate_cash_commitments():
    order = Order("1", "weights", SubmitOrder("A", "BUY", 5), NOW, 1)
    plan = target_weight_orders(
        context(orders=(order,)), {"A": 0.5, "B": 0.5}, cash_buffer=0
    )
    assert not plan.orders


def test_invalid_weights_and_unsupported_fractional_execution():
    with pytest.raises(ValueError):
        target_weight_orders(context(), {"A": 0.8, "B": 0.8})
    with pytest.raises(ValueError):
        target_weight_orders(context(), {"A": -0.1})
    with pytest.raises(NotImplementedError, match="whole shares"):
        target_weight_orders(context(), {"A": 1}, fractional=True)


def test_strategy_freezes_share_targets_between_scheduled_decisions():
    strategy = TargetWeightStrategy(lambda ctx: {"A": 1}, cash_buffer=0)
    orders = strategy.on_event(Timer(NOW, "rebalance"), context())
    assert orders is not None and orders[0].quantity == 10
    later = context(
        positions={"A": Position(10)}, cash=0, equity=2000, marks={"A": Decimal(200)}
    )
    bar = Bar(
        "A",
        NOW - timedelta(hours=1),
        NOW,
        Decimal(200),
        Decimal(200),
        Decimal(200),
        Decimal(200),
        1000,
    )
    assert strategy.on_event(Bars(NOW, {"A": bar}), later) == ()
    assert strategy.targets == {"A": 10}


def test_stale_target_does_not_become_an_unintended_liquidation_when_price_recovers():
    strategy = TargetWeightStrategy(lambda ctx: {"A": 1}, cash_buffer=0)
    held = {"A": Position(5)}
    strategy.on_event(Timer(NOW, "rebalance"), context(positions=held, stale=True))
    assert strategy.targets == {}
    observed = Bar(
        "A",
        NOW - timedelta(hours=1),
        NOW,
        Decimal(100),
        Decimal(100),
        Decimal(100),
        Decimal(100),
        1000,
    )
    assert strategy.on_event(Bars(NOW, {"A": observed}), context(positions=held)) == ()
    assert strategy.plans[-1].diagnostics[0].endswith("wait for rebalance")


def test_solver_roundoff_is_accepted_but_material_leverage_is_rejected():
    plan = target_weight_orders(
        context(),
        {"A": 0.5000000001, "B": 0.5, "C": -1e-10},
        cash_buffer=0,
    )
    assert plan.weights["C"] == 0
    assert sum(plan.weights.values()) == 1
    assert sum(order.quantity * 100 for order in plan.orders) <= 1000

    for weights in ({"A": 1.0001}, {"A": 1, "B": -0.0001}):
        with pytest.raises(ValueError, match="Target weights"):
            target_weight_orders(context(), weights)
