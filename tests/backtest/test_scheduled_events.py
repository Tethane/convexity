"""End-to-end recurring event delivery and the overnight order-ID regression."""

from collections import Counter
from dataclasses import replace
from datetime import date, datetime, time, timedelta, timezone
import json
from typing import Any, cast
from zoneinfo import ZoneInfo

import pytest
import convexity as cx
from convexity.runtime.subscriptions import SubscriptionBook, subscription_id
from examples.overnight_hold import OvernightHold
from tests.backtest.test_research import FixtureProvider


def test_overnight_repeats_every_session_and_fills_after_each_clock(tmp_path):
    received = []

    class Observed(OvernightHold):
        def on_timer(self, event, context):
            received.append(event)
            return super().on_timer(event, context)

    provider = FixtureProvider()
    result = cx.Research(date(2024, 1, 2), date(2024, 1, 12)).run(
        Observed, provider=provider, cache=None, output=tmp_path
    )
    assert result.status == "completed"
    sessions = cx.SessionCalendar.from_exchange(
        "XNYS", date(2024, 1, 2), date(2024, 1, 12)
    ).sessions
    assert len(sessions) == 9
    assert Counter(e.name for e in received) == {"enter": 9, "exit": 9}
    assert received == [e for e in result.events if isinstance(e, cx.Timer)]
    assert (
        len(result.fills) == 17
    )  # Final entry remains open at the end of this interval.
    assert [f.side for f in result.fills] == ["BUY", "SELL"] * 8 + ["BUY"]
    assert len({order.order_id for order in result.orders}) == 17
    assert not any(isinstance(e, cx.Rejection) for e in result.events)
    entries = [event for event in received if event.name == "enter"]
    exits = [event for event in received if event.name == "exit"][1:]
    decisions = sorted(entries + exits, key=lambda e: e.time)
    for fill, decision in zip(result.fills, decisions):
        assert fill.occurred_at == decision.time
        assert fill.available_at == decision.time + timedelta(minutes=1)
    assert provider.calls[0][0].bar_size == "1 min"
    assert json.loads((tmp_path / "result.json").read_text())["status"] == "completed"


def test_fixed_entry_id_reproduces_flat_curve_while_clocks_keep_firing():
    class FixedID(OvernightHold):
        def on_timer(self, event, context):
            command = super().on_timer(event, context)
            return (
                replace(command, client_order_id="entry")
                if command and command.side == "BUY"
                else command
            )

        def on_rejection(self, event, context):
            pass

    result = cx.Research(date(2024, 1, 2), date(2024, 1, 12)).run(
        FixedID, provider=FixtureProvider(), cache=None
    )
    assert Counter(e.name for e in result.events if isinstance(e, cx.Timer)) == {
        "enter": 9,
        "exit": 9,
    }
    assert len(result.fills) == 2
    assert any(
        isinstance(e, cx.Rejection) and "another intent" in e.reason
        for e in result.events
    )
    after_exit = [
        point["equity"]
        for point in result.equity
        if point["time"] >= result.fills[1].available_at
    ]
    assert len(set(after_exit)) == 1


def test_typed_timer_schedule_respects_dst_holidays_early_close_and_owners():
    seen = {"morning": [], "afternoon": []}

    class Morning(cx.Strategy):
        metadata = cx.StrategyMetadata("morning")
        contract = cx.StrategyContract(
            (),
            (
                cx.EventSubscription(
                    cx.Timer, schedule=cx.ClockSchedule("tick", time(9, 30))
                ),
            ),
        )

        def on_timer(self, event, context):
            seen["morning"].append(event)

    class Afternoon(cx.Strategy):
        metadata = cx.StrategyMetadata("afternoon")
        contract = cx.StrategyContract(
            (),
            (
                cx.EventSubscription(
                    cx.Timer, schedule=cx.ClockSchedule("tick", time(15, 50))
                ),
            ),
        )

        def on_timer(self, event, context):
            seen["afternoon"].append(event)

    result = cx.Research(date(2024, 11, 1), date(2024, 11, 29)).run(
        (Morning, Afternoon)
    )
    sessions = cx.SessionCalendar.from_exchange(
        "XNYS", date(2024, 11, 1), date(2024, 11, 29)
    ).sessions
    assert result.status == "completed"
    assert len(seen["morning"]) == len(sessions)
    assert len(seen["afternoon"]) == len(sessions) - 1
    assert seen["morning"][0].time.hour == 13 and seen["morning"][1].time.hour == 14
    assert all(
        e.time.astimezone(ZoneInfo("America/New_York")).time() == time(15, 50)
        for e in seen["afternoon"]
    )
    assert all(
        e.time.date() != date(2024, 11, 28) for events in seen.values() for e in events
    )
    assert seen["afternoon"][-1].time.date() == date(2024, 11, 27)


def test_timer_event_accepts_session_relative_schedule():
    schedule = cx.SessionSchedule("daily-open", "open", date(2024, 1, 2), at="open")
    seen = []

    class Observer(cx.Strategy):
        metadata = cx.StrategyMetadata("observer")
        contract = cx.StrategyContract(
            (), (cx.EventSubscription(cx.Timer, schedule=schedule),)
        )

        def on_timer(self, event, context):
            seen.append(event)

    cx.Research(date(2024, 1, 2), date(2024, 1, 5)).run(Observer)
    assert len(seen) == 4
    assert all(e.time.hour == 14 and e.time.minute == 30 for e in seen)


def test_dynamic_typed_timer_subscription_activation_and_restore():
    first = cx.EventSubscription(
        cx.Timer, schedule=cx.ClockSchedule("tick", time(9, 30))
    )
    second = cx.EventSubscription(
        cx.Timer, schedule=cx.ClockSchedule("tick", time(15, 50))
    )
    contract = cx.StrategyContract((), (first,), available_subscriptions=(second,))
    book = SubscriptionBook(contract, "owner")
    morning = cx.Timer(datetime(2024, 1, 2, 14, 30, tzinfo=timezone.utc), "tick")
    afternoon = cx.Timer(datetime(2024, 1, 2, 20, 50, tzinfo=timezone.utc), "tick")
    assert book.select(morning) == morning and book.select(afternoon) is None
    assert book.apply(cx.Subscribe(second, "pm"), morning.time).status == "active"
    restored = SubscriptionBook(contract, "owner", book.dump())
    assert restored.select(afternoon) == afternoon
    restored.apply(cx.Unsubscribe(subscription_id(first)), morning.time)
    assert restored.select(morning) is None


def test_schedule_validation_and_legacy_clock_compatibility():
    with pytest.raises(TypeError, match="datetime.time"):
        cx.ClockSchedule("enter", cast(Any, "15:50"))
    with pytest.raises(ValueError, match="local wall-clock"):
        cx.ClockSchedule("enter", time(15, 50, tzinfo=timezone.utc))
    with pytest.raises(ValueError, match="require"):
        cx.EventSubscription(cx.Timer)
    with pytest.raises(ValueError, match="Only Timer"):
        cx.EventSubscription(cx.FillEvent, schedule=cx.ClockSchedule("enter"))
    schedule = cx.ClockSchedule("enter", time(15, 50))
    with pytest.raises(ValueError, match="unique"):
        cx.StrategyContract(
            (),
            (
                cx.EventSubscription(cx.Timer, schedule=schedule),
                cx.EventSubscription(
                    cx.Timer, schedule=cx.ClockSchedule("enter", time(9, 30))
                ),
            ),
        )
    assert (
        cx.ClockSubscription("enter", "15:50").materialize(2024).timers
        == schedule.materialize(2024).timers
    )
