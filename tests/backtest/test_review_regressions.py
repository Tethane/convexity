from decimal import Decimal as D

"""Execution edge cases discovered during the independent MVP review."""
from datetime import datetime, timedelta, timezone

import pytest

from convexity.backtest import Backtest, HistoricalData, SimulatedExchange
from convexity.runtime import Bar, Bars, RiskLimits, Runtime, Strategy, SubmitOrder

T = datetime(2026, 1, 2, tzinfo=timezone.utc)
H = timedelta(hours=1)


class BuyOnce(Strategy):
    def on_event(self, event, context):
        if isinstance(event, Bars) and not context.orders:
            return SubmitOrder("A", "BUY", 10)


def test_opening_gap_rechecks_order_notional_limit():
    bars = HistoricalData(
        tuple(
            Bar(
                "A",
                T + n * H,
                T + (n + 1) * H,
                D(price),
                D(price),
                D(price),
                D(price),
                100,
            )
            for n, price in enumerate((100, 110))
        )
    )
    result = Backtest(
        data=bars,
        strategies={"s": BuyOnce},
        limits=RiskLimits(max_order_notional=D(1000)),
    ).run()
    assert result.status == "completed"
    assert not result.fills
    assert result.orders[0].status == "rejected"
    assert result.metrics["final_cash"] == 100_000


def test_ohlc_crossing_day_expiry_cannot_claim_post_expiry_fill():
    from convexity.runtime import Instrument

    instruments = {"A": Instrument("A")}
    r = Runtime(
        {"s": Strategy()},
        SimulatedExchange(instruments),
        instruments,
        session_closes=(T + 2 * H,),
    )
    r.start(T)
    r.advance(
        T + H,
        Bars(T + H, {"A": Bar("A", T, T + H, D(100), D(100), D(100), D(100), 100)}),
    )
    r.command("s", SubmitOrder("A", "BUY", 1, "limit", D(90), "DAY"))
    with pytest.raises(ValueError, match="expiry"):
        r.advance(
            T + 3 * H,
            Bars(
                T + 3 * H,
                {"A": Bar("A", T + H, T + 3 * H, D(100), D(100), D(90), D(90), 100)},
            ),
        )
    assert not r.fills
    assert r.cash == 100_000
