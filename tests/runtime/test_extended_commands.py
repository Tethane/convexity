"""Public extended commands preserve explicit intent and stable derived identities."""

from dataclasses import replace
from decimal import Decimal

import pytest

import convexity as cx
from convexity.runtime.orders import amended_request, expand_orders


def test_stop_intents_preserve_trigger_separately_from_limit():
    order = cx.SubmitOrder.stop_limit(
        "SPY", "SELL", 10, stop_price="95", limit_price="94"
    )
    assert order.order_type == "stop_limit"
    assert order.stop_price == Decimal(95) and order.limit_price == Decimal(94)
    updated = amended_request(
        order, cx.ReplaceOrder("alpha:entry", quantity=12, stop_price=Decimal("96"))
    )
    assert updated.quantity == 12 and updated.stop_price == 96
    assert updated.limit_price == 94 and updated.side == "SELL"
    for change in ({"stop_price": None}, {"stop_price": 0}, {"time_in_force": "IOC"}):
        with pytest.raises(ValueError):
            replace(order, **change)
    with pytest.raises(ValueError):
        cx.SubmitOrder.market("SPY", "BUY", 1, time_in_force="invalid")  # pyright: ignore[reportArgumentType] -- deliberate runtime rejection


def test_bracket_has_deterministic_ids_and_gtc_protection():
    group = cx.BracketOrder(
        cx.SubmitOrder.limit("SPY", "BUY", 10, 100),
        take_profit=Decimal(110),
        stop_loss=Decimal(95),
        client_order_id="entry",
    )
    parent, stop, profit = expand_orders(group)
    assert [order.client_order_id for order in (parent, stop, profit)] == [
        "entry:entry",
        "entry:stop_loss",
        "entry:take_profit",
    ]
    assert stop.side == profit.side == "SELL"
    assert stop.quantity == profit.quantity == parent.quantity == 10
    assert stop.time_in_force == profit.time_in_force == "GTC"
    assert stop.order_type == "stop" and profit.order_type == "limit"
    with pytest.raises(ValueError, match="identity"):
        expand_orders(
            replace(group, entry=replace(parent, client_order_id="unrelated"))
        )


def test_oco_child_ids_derived_without_silently_overriding_user_ids():
    orders = (
        cx.SubmitOrder.limit("SPY", "SELL", 5, 110),
        cx.SubmitOrder.stop("SPY", "SELL", 5, 95),
    )
    assert [
        order.client_order_id for order in expand_orders(cx.OCOOrders(orders, "exits"))
    ] == ["exits:0", "exits:1"]
    with pytest.raises(ValueError, match="identities"):
        expand_orders(
            cx.OCOOrders(
                (replace(orders[0], client_order_id="some-id"), orders[1]), "exits"
            )
        )


def test_cancellation_filter_and_empty_amendment_validation():
    assert cx.CancelOrders().symbol is None
    with pytest.raises(ValueError):
        cx.CancelOrders(side="short")  # pyright: ignore[reportArgumentType] -- deliberate runtime rejection
    with pytest.raises(ValueError):
        cx.ReplaceOrder("entry")
    with pytest.raises(ValueError):
        cx.ReplaceOrder("entry", quantity=0)


def test_public_capabilities_are_explicit_about_backend_differences():
    assert cx.ReplaceOrder in cx.RESEARCH_CAPABILITIES.commands
    assert cx.Subscribe in cx.PAPER_CAPABILITIES.commands
    assert "stop_limit" in cx.PAPER_CAPABILITIES.order_types
    assert "IOC" not in cx.PAPER_CAPABILITIES.time_in_force
    assert not cx.RESEARCH_CAPABILITIES.native_order_groups
