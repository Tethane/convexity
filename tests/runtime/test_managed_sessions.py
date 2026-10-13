"""Continuous deployments renew schedules and bound memory without new plumbing."""
import asyncio
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace
from typing import cast

import pytest

from convexity.runtime import PaperRuntime, SessionSchedule, Strategy, Timer
from convexity.runtime.runner import ScheduleSpec, StockSubscription
from tests.runtime.test_runner import app, now


def test_recurring_monthly_start_uses_full_month_and_daily_early_close():
    pytest.importorskip("exchange_calendars")
    monthly = SessionSchedule("monthly", "rebalance", date(2024, 10, 9), frequency="monthly", at="open")
    assert [timer.time.date() for timer in monthly.materialize(2024).timers] == [date(2024, 11, 1), date(2024, 12, 2)]
    exits = SessionSchedule("exits", "exit", date(2024, 11, 28), end=date(2024, 11, 29),
                            offset=-timedelta(minutes=15), policy="catch_up")
    timers = exits.materialize(2024).timers
    assert len(timers) == 1 and timers[0].time == datetime(2024, 11, 29, 17, 45, tzinfo=timezone.utc)


def test_runner_renews_recurring_year_without_rewriting_old_schedule(tmp_path):
    pytest.importorskip("exchange_calendars")
    declaration = SessionSchedule("monthly", "rebalance", date(2025, 1, 1), frequency="monthly")
    runner, _ = app(tmp_path, schedules=[declaration])
    started = datetime(2025, 12, 31, tzinfo=timezone.utc)
    schedules = []
    try:
        runner._sync_schedules(schedules, started, started)
        assert [runner._schedule_owners[schedule.namespace] for schedule in schedules] == [("session:monthly:2025", "alpha"), ("session:monthly:2026", "alpha")]
        first = schedules[0]
        runner._sync_schedules(schedules, started, datetime(2026, 1, 1, tzinfo=timezone.utc))
        runner._sync_schedules(schedules, started, datetime(2026, 1, 2, tzinfo=timezone.utc))
        assert [runner._schedule_owners[schedule.namespace] for schedule in schedules] == [("session:monthly:2025", "alpha"), ("session:monthly:2026", "alpha"), ("session:monthly:2027", "alpha")]
        assert schedules[0] is first
        assert len(schedules[1].timers) == 12
    finally:
        for schedule in schedules:
            schedule.close()


def test_reconnect_skips_missed_entry_but_delivers_overdue_exit(tmp_path):
    instances = []
    class RecordTimers(Strategy):
        def __init__(self):
            self.timers = []
            instances.append(self)
        def on_event(self, event, context):
            if isinstance(event, Timer):
                self.timers.append(event.name)
    async def run():
        due = now() + timedelta(seconds=1)
        runner, brokers = app(tmp_path, RecordTimers, reconnect_delay=1.2,
            schedules=[ScheduleSpec("entry", (Timer(due, "entry"),)),
                       ScheduleSpec("exit", (Timer(due, "exit"),), "catch_up")])
        task = await runner.start()
        brokers[0].fail = True
        async with asyncio.timeout(5):
            while len(instances) < 2 or "exit" not in instances[-1].timers:
                if task.done():
                    await task
                await asyncio.sleep(.01)
        await runner.stop()
        assert all("entry" not in strategy.timers for strategy in instances)
        assert instances[-1].timers == ["exit"]
        assert not any(broker.sent for broker in brokers)
    asyncio.run(run())


def test_runner_retains_configured_working_history_and_recent_events(tmp_path):
    runner, _ = app(tmp_path, subscriptions={"A": StockSubscription(history_seconds=180, max_bars=3)},
                    event_history_limit=4)
    runtime = SimpleNamespace(events=list(range(20)), errors=list(range(8)), history={"A": list(range(10))})
    runner._retain(cast(PaperRuntime, runtime))
    assert runtime.history["A"] == [7, 8, 9]
    assert runtime.events == [16, 17, 18, 19]
    assert runtime.errors == [4, 5, 6, 7]


def test_subscription_retention_cannot_discard_required_initial_window():
    with pytest.raises(ValueError, match="max_bars"):
        StockSubscription(history_seconds=1800, max_bars=3)


def test_recurring_schedule_rejects_offsets_outside_adjacent_year_horizon():
    with pytest.raises(ValueError, match="less than one day"):
        SessionSchedule("bad", "entry", date(2026, 1, 1), offset=-timedelta(days=30))
