"""Immediate order policies must not turn into resting or partial FOK orders."""
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from convexity.backtest import Backtest, ExecutionModel, HistoricalData
from convexity.runtime import Bar, Bars, Strategy, SubmitOrder


def replay(tif, *, limit=None, volume=3, quantity=5):
    start = datetime(2025, 1, 2, 14, 30, tzinfo=timezone.utc)
    bars = HistoricalData(tuple(Bar("SPY", start + timedelta(minutes=i),
        start + timedelta(minutes=i + 1), 100, 105, 90, 100, volume) for i in range(4)))
    class Buyer(Strategy):
        sent = False
        def on_event(self, event, context):
            if isinstance(event, Bars) and not self.sent:
                self.sent = True
                return SubmitOrder("SPY", "BUY", quantity,
                    order_type="limit" if limit is not None else "market",
                    limit_price=limit, time_in_force=tif)
    return Backtest(data=bars, strategies={"buyer": Buyer}).run().raise_for_failure()


def test_ioc_fills_available_quantity_once_then_cancels():
    result = replay("IOC")
    assert [fill.quantity for fill in result.fills] == [3]
    assert result.orders[0].status == "canceled"
    assert result.orders[0].filled_quantity == 3
    assert not result.open_orders


def test_fok_cancels_without_partial_execution():
    result = replay("FOK")
    assert not result.fills
    assert result.orders[0].status == "canceled"
    assert result.metrics["final_cash"] == 100000


def test_fok_executes_entire_quantity_if_available():
    result = replay("FOK", quantity=3)
    assert [fill.quantity for fill in result.fills] == [3]
    assert result.orders[0].status == "filled"


@pytest.mark.parametrize("tif", ["IOC", "FOK"])
def test_immediate_limit_cannot_wait_for_intrabar_crossing(tif):
    result = replay(tif, limit=Decimal(95), quantity=1)
    assert not result.fills
    assert result.orders[0].status == "canceled"


def test_presets_are_explicit_and_unknown_names_fail():
    assert ExecutionModel.preset("frictionless") == ExecutionModel()
    assert ExecutionModel.preset().volume_participation == Decimal(".1")
    assert ExecutionModel.preset("conservative").limit_rule == "penetration"
    with pytest.raises(ValueError, match="preset"):
        ExecutionModel.preset("magic")
