from dataclasses import replace
from collections.abc import Callable
from typing import TypedDict
from datetime import datetime, timedelta, timezone, date
from decimal import Decimal as D
import json
from zoneinfo import ZoneInfo

import pytest

from convexity.backtest import (
    Backtest,
    ExecutionModel,
    HistoricalData,
    Session,
    SessionCalendar,
    SimulatedExchange,
)
from convexity.runtime import (
    Bar,
    Bars,
    CancelOrder,
    Fill,
    FillEvent,
    Instrument,
    Order,
    OrderUpdate,
    RiskLimits,
    Runtime,
    Strategy,
    SubmitOrder,
    Timer,
)

T = datetime(2024, 1, 2, 14, 30, tzinfo=timezone.utc)


def data(prices=(100, 110, 120, 130), volume=100, symbols=("A",)):
    return HistoricalData(
        tuple(
            Bar(
                s,
                T + timedelta(hours=i),
                T + timedelta(hours=i + 1),
                D(p),
                D(p + 2),
                D(p - 2),
                D(p),
                volume,
            )
            for i, p in enumerate(prices)
            for s in symbols
        )
    )


class BuyOnce(Strategy):
    def __init__(self, quantity=10, **kwargs):
        self.quantity = quantity
        self.kwargs = kwargs
        self.sent = False

    def on_event(self, event, context):
        if isinstance(event, Bars) and not self.sent:
            self.sent = True
            return SubmitOrder("A", "BUY", self.quantity, **self.kwargs)


def run(factory: Callable[[], Strategy] = BuyOnce, bars=None, **kwargs):
    return Backtest(
        data=bars or data(), strategies={"test": factory}, initial_cash=10_000, **kwargs
    ).run()


def test_market_orders_cannot_fill_signal_bar_and_returns_open_position():
    result = run()
    assert result.status == "completed"
    assert len(result.fills) == 1
    fill = result.fills[0]
    assert fill.price == 110
    assert fill.occurred_at == T + timedelta(hours=1)
    assert fill.available_at == T + timedelta(hours=2)
    assert result.positions["test"]["A"].quantity == 10
    assert result.metrics["final_cash"] == 8900
    assert result.metrics["final_equity"] == 10200
    assert result.metrics["total_return"] == D(".02")


def test_repeated_runs_identical_and_factory_creates_fresh_state(tmp_path):
    backtest = Backtest(data=data(), strategies={"test": BuyOnce})
    a, b = backtest.run(), backtest.run()
    assert a.to_dict() == b.to_dict()
    a.export(tmp_path / "a")
    b.export(tmp_path / "b")
    assert (tmp_path / "a/result.json").read_bytes() == (
        tmp_path / "b/result.json"
    ).read_bytes()
    assert json.loads((tmp_path / "a/result.json").read_text())["status"] == "completed"


def test_timer_routing_validates_once_and_passes_only_current_timers(monkeypatch):
    first, last = (
        Timer(T + timedelta(hours=1), "first"),
        Timer(T + timedelta(hours=3), "last"),
    )
    observed = []

    class Clock(Strategy):
        def on_timer(self, event, context):
            observed.append(event.name)

    advance = Runtime.advance

    def checked(self, time, bars=None, timers=(), **kwargs):
        assert set(kwargs.get("timer_owners") or {}) <= set(timers)
        return advance(self, time, bars, timers, **kwargs)

    monkeypatch.setattr(Runtime, "advance", checked)
    result = Backtest(
        data=data(),
        strategies={"test": Clock},
        timers=(first, last),
        timer_owners={first: ("test",), last: ()},
    ).run()
    assert result.status == "completed" and observed == ["first"]
    with pytest.raises(ValueError, match="registered strategy"):
        Backtest(
            data=data(),
            strategies={"test": Clock},
            timers=(first, last),
            timer_owners={last: ("unknown",)},
        )


def test_friction_and_commissions_are_hand_calculable():
    result = run(
        execution=ExecutionModel(
            spread_bps=D(20),
            slippage_bps=D(10),
            commission_per_share=D(".1"),
            minimum_commission=D(2),
        )
    )
    # 110 * 1.002 = 110.22; commission max(1,2) = 2.
    assert result.fills[0].price == D("110.22")
    assert result.fills[0].fee == 2
    assert result.metrics["final_cash"] == D("8895.80")


def test_shared_volume_fifo_partial_fills_and_per_order_minimum_fee():
    result = Backtest(
        data=data(volume=10),
        strategies={"first": lambda: BuyOnce(4), "second": lambda: BuyOnce(4)},
        execution=ExecutionModel(
            volume_participation=D(".5"),
            commission_per_share=D(".1"),
            minimum_commission=D(1),
        ),
    ).run()
    assert [(f.strategy_id, f.quantity) for f in result.fills] == [
        ("first", 4),
        ("second", 1),
        ("second", 3),
    ]
    assert sum(f.fee for f in result.fills) == 2
    assert result.fills[1].available_at < result.fills[2].available_at


def test_competing_strategies_reserve_same_capacity_once():
    result = Backtest(
        data=data((100, 100)),
        initial_cash=1000,
        strategies={"first": lambda: BuyOnce(6), "second": lambda: BuyOnce(6)},
    ).run()
    assert [o.status for o in result.orders] == ["filled", "rejected"]
    assert result.orders[1].reason and "capacity" in result.orders[1].reason


def test_gap_rechecks_capacity_at_execution():
    result = run(lambda: BuyOnce(100), bars=data((100, 200)))
    assert not result.fills
    assert result.orders[0].status == "rejected"
    assert result.orders[0].reason is not None and "gap" in result.orders[0].reason


def test_limit_touch_penetration_gap_and_limit_protection():
    bars = data((100, 102, 99))  # second low = 100, third open = 99
    touch = run(lambda: BuyOnce(1, order_type="limit", limit_price=D(100)), bars=bars)
    strict = run(
        lambda: BuyOnce(1, order_type="limit", limit_price=D(100)),
        bars=bars,
        execution=ExecutionModel(limit_rule="penetration"),
    )
    assert touch.fills[0].price == 100
    assert touch.fills[0].occurred_at == T + timedelta(hours=2)
    assert strict.fills[0].price == 99
    assert strict.fills[0].available_at == T + timedelta(hours=3)
    costly = run(
        lambda: BuyOnce(1, order_type="limit", limit_price=D(100)),
        bars=bars,
        execution=ExecutionModel(slippage_bps=D(200)),
    )
    assert costly.fills[0].price <= 100


def test_submission_delay_skips_one_eligible_bar():
    result = run(execution=ExecutionModel(submission_delay_bars=1))
    assert result.fills[0].price == 120


class CancelAfterSecondBar(BuyOnce):
    def on_event(self, event, context):
        output = super().on_event(event, context)
        if output:
            return output
        if isinstance(event, Bars) and len(context.history["A"]) == 2:
            return CancelOrder(context.orders[0].order_id)


def test_cancel_delay_permits_fill_race_then_cancels_remaining():
    result = run(
        lambda: CancelAfterSecondBar(10),
        bars=data(volume=2),
        execution=ExecutionModel(cancel_delay_bars=1),
    )
    assert [f.quantity for f in result.fills] == [2, 2]
    assert result.orders[0].status == "canceled"
    assert result.orders[0].filled_quantity == 4
    assert not result.orders[0].cancel_pending
    immediate = run(lambda: CancelAfterSecondBar(10), bars=data(volume=2))
    assert len(immediate.fills) == 1
    assert immediate.orders[0].status == "canceled"


def test_pending_orders_remain_at_end_and_no_implicit_liquidation():
    result = run(lambda: BuyOnce(10), bars=data((100, 100), volume=2))
    assert len(result.open_orders) == 1
    assert result.open_orders[0].remaining == 8
    assert result.positions["test"]["A"].quantity == 2


def test_readonly_context_and_no_future_data():
    seen = []

    class Inspect(Strategy):
        def on_event(self, event, context):
            if isinstance(event, Bars):
                assert max(b.end for b in context.history["A"]) == event.time
                assert len(context.history["A"]) == len(seen) + 1
                with pytest.raises(TypeError):
                    context.marks["A"] = D(1)  # pyright: ignore[reportIndexIssue] -- deliberate immutable mapping mutation
                seen.append(context)

    result = run(Inspect)
    assert result.status == "completed"
    assert len(seen[0].history["A"]) == 1
    assert seen[0].marks["A"] == 100


def test_strategy_failure_marks_run_incomplete_and_other_strategy_continues():
    class Broken(Strategy):
        def on_event(self, event, context):
            raise RuntimeError("bad signal")

    result = Backtest(
        data=data(), strategies={"broken": Broken, "healthy": BuyOnce}
    ).run()
    assert result.status == "failed"
    assert "bad signal" in result.diagnostics[0]
    assert result.fills[0].strategy_id == "healthy"


def test_duplicate_execution_books_once_and_closing_short_realizes_pnl():
    instruments = {"A": Instrument("A")}
    rt = Runtime({"test": Strategy()}, SimulatedExchange(instruments), instruments)
    rt.start(T)
    rt.marks["A"] = D(100)
    rt.command("test", SubmitOrder("A", "SELL", 5))
    order = next(iter(rt.orders.values()))
    fill = Fill("fill-1", order.order_id, "test", "A", "SELL", 5, D(100), D(1), T, T)
    rt.apply(FillEvent(T, fill))
    rt.apply(FillEvent(T, fill))
    assert rt.cash == 100499
    assert len(rt.fills) == 1
    rt.command("test", SubmitOrder("A", "BUY", 5))
    close = list(rt.orders.values())[-1]
    rt.apply(
        FillEvent(
            T, Fill("fill-2", close.order_id, "test", "A", "BUY", 5, D(90), D(1), T, T)
        )
    )
    assert rt.positions[("test", "A")].quantity == 0
    assert rt.positions[("test", "A")].realized_pnl == 50
    assert rt.cash == 100048


def test_short_availability_and_strategy_limits():
    class Short(Strategy):
        def on_event(self, event, context):
            if isinstance(event, Bars) and len(context.history["A"]) == 1:
                return SubmitOrder("A", "SELL", 10)

    result = run(Short, execution=ExecutionModel(allow_short=False))
    assert result.orders[0].status == "rejected"
    result = run(limits=RiskLimits(strategy_gross_limits={"test": D(500)}))
    assert result.orders[0].status == "rejected"


def test_day_expiry_after_last_session_bar_and_missing_calendar_rejected():
    calendar = SessionCalendar((Session(date(2024, 1, 2), T, T + timedelta(hours=4)),))
    result = run(
        lambda: BuyOnce(10, order_type="limit", limit_price=D(50), time_in_force="DAY"),
        calendar=calendar,
    )
    assert result.orders[0].status == "expired"
    assert not result.fills
    missing = run(lambda: BuyOnce(1, time_in_force="DAY"))
    assert missing.orders[0].status == "rejected"


def test_calendar_dst_early_close_and_monthly_timers():
    zone = ZoneInfo("America/New_York")
    sessions = tuple(
        Session(
            day,
            datetime(day.year, day.month, day.day, 9, 30, tzinfo=zone),
            datetime(day.year, day.month, day.day, close, tzinfo=zone),
        )
        for day, close in [
            (date(2024, 3, 8), 16),
            (date(2024, 3, 11), 16),
            (date(2024, 11, 29), 13),
        ]
    )
    calendar = SessionCalendar(sessions)
    assert len(calendar.monthly("rebalance")) == 2
    timers = calendar.daily("exit", offset=timedelta(minutes=-5))
    assert timers[-1].time.astimezone(zone).hour == 12 and timers[-1].time.minute == 55
    assert sessions[0].open.utcoffset() != sessions[1].open.utcoffset()
    assert (
        len(calendar.daily("open", at="open")) == 3
    )  # No invented weekend/holiday sessions.


def test_equal_time_timer_after_bars_and_duplicate_timer_fires_once():
    seen = []

    class Observe(Strategy):
        def on_event(self, event, context):
            if isinstance(event, Timer):
                seen.append((event.name, len(context.history["A"])))

    timer = Timer(T + timedelta(hours=1), "rebalance")
    result = run(Observe, timers=(timer, timer))
    assert result.status == "completed"
    assert seen == [("rebalance", 1)]


def test_intrabar_schedule_fails_explicitly():
    with pytest.raises(ValueError, match="intrabar"):
        run(timers=(Timer(T + timedelta(minutes=15), "unsafe"),))


def test_data_coverage_invalid_bars_actions_and_unsupported_types(tmp_path):
    with pytest.raises(ValueError, match="OHLC"):
        Bar("A", T, T + timedelta(hours=1), D(100), D(99), D(90), D(100), 10)
    with pytest.raises(ValueError, match="Incomplete"):
        HistoricalData(data(symbols=("A", "B")).bars[:-1])
    with pytest.raises(ValueError, match="overlapping"):
        HistoricalData((data().bars[0],) * 2)
    with pytest.raises(ValueError, match="adjusted"):
        HistoricalData(data().bars, price_basis="adjusted")
    with pytest.raises(ValueError, match="stop price"):
        SubmitOrder("A", "BUY", 1, order_type="stop")  # A stop needs its trigger price.
    path = tmp_path / "bars.csv"
    path.write_text(
        "symbol,start,end,open,high,low,close,volume,split_factor\nA,2024-01-02T14:30:00+00:00,2024-01-02T15:30:00+00:00,100,101,99,100,100,2\n"
    )
    with pytest.raises(ValueError, match="Corporate"):
        HistoricalData.from_csv(path)


def test_borrow_uses_fill_time_including_open_and_close_bar():
    class RoundTripShort(Strategy):
        def on_event(self, event, context):
            if isinstance(event, Bars):
                n = len(context.history["A"])
                if n == 1:
                    return SubmitOrder("A", "SELL", 10)
                if n == 2:
                    return SubmitOrder("A", "BUY", 10)

    result = run(
        RoundTripShort,
        bars=data((100, 100, 100)),
        execution=ExecutionModel(annual_borrow_rate=D(".365")),
    )
    expected = D(1000) * D(".365") / D(365 * 24)
    assert abs(result.metrics["borrow_cost"] - expected) < D("1e-20")
    assert result.positions["test"]["A"].quantity == 0
    assert abs(result.metrics["final_equity"] - (D(10000) - expected)) < D("1e-20")


def test_sell_limit_gap_and_adverse_rounding():
    class Sell(Strategy):
        def on_event(self, event, context):
            if isinstance(event, Bars) and len(context.history["A"]) == 1:
                return SubmitOrder(
                    "A", "SELL", 1, order_type="limit", limit_price=D(100)
                )

    result = run(
        Sell, bars=data((100, 110)), execution=ExecutionModel(slippage_bps=D(1))
    )
    assert result.fills[0].price == D("109.98")


def test_close_is_allowed_after_adverse_move_breaches_gross_limit():
    class Cover(Strategy):
        def on_event(self, event, context):
            if isinstance(event, Bars):
                n = len(context.history["A"])
                if n == 1:
                    return SubmitOrder("A", "SELL", 80)
                if n == 3:
                    return SubmitOrder("A", "BUY", 80)

    result = run(Cover, bars=data((100, 100, 200, 200)))
    assert len(result.fills) == 2
    assert result.positions["test"]["A"].quantity == 0


def test_reused_strategy_instance_rejected_on_repeat_run():
    instance = BuyOnce()
    backtest = Backtest(data=data(), strategies={"test": lambda: instance})
    backtest.run()
    with pytest.raises(ValueError, match="reused"):
        backtest.run()


def test_csv_demo_closes_research_loop_under_two_execution_models():
    from pathlib import Path
    from convexity.strategies import MonthlyLongShort, StockRule

    root = Path("examples/data")
    bars = HistoricalData.from_csv(root / "synthetic_stocks.csv")
    calendar = SessionCalendar.from_csv(root / "synthetic_sessions.csv")

    class Inputs(TypedDict):
        data: HistoricalData
        strategies: dict[str, Callable[[], Strategy]]
        calendar: SessionCalendar
        timers: tuple[Timer, ...]

    kwargs = Inputs(
        data=bars,
        strategies={
            "monthly": lambda: MonthlyLongShort(
                StockRule(tuple(sorted({b.symbol for b in bars.bars})))
            )
        },
        calendar=calendar,
        timers=calendar.monthly("rebalance"),
    )
    baseline = Backtest(**kwargs).run()
    costly = Backtest(
        **kwargs, execution=ExecutionModel(slippage_bps=D(25), minimum_commission=D(3))
    ).run()
    assert baseline.status == costly.status == "completed"
    assert len(baseline.fills) == 10
    assert {f.side for f in baseline.fills} == {"BUY", "SELL"}
    assert costly.metrics["final_equity"] < baseline.metrics["final_equity"]
    assert all(
        f.available_at
        > next(o.accepted_at for o in baseline.orders if o.order_id == f.order_id)
        for f in baseline.fills
    )


def test_calendar_rejects_omitted_data_for_declared_session():
    calendar = SessionCalendar((Session(date(2024, 1, 2), T, T + timedelta(hours=4)),))
    complete = data()
    missing = HistoricalData(complete.bars[:1] + complete.bars[2:])
    with pytest.raises(ValueError, match="gap"):
        run(bars=missing, calendar=calendar)


def test_utc_normalization_preserves_elapsed_time_across_dst():
    zone = ZoneInfo("America/New_York")
    start = datetime(2024, 3, 8, 16, tzinfo=zone)
    end = datetime(2024, 3, 11, 16, tzinfo=zone)
    bar = Bar("A", start, end, D(100), D(101), D(99), D(100), 100)
    assert (bar.end - bar.start).total_seconds() == 71 * 3600
    assert Timer(end, "close").time.utcoffset() == timedelta(0)
