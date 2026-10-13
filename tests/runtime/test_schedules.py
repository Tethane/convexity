from datetime import datetime, timedelta, timezone

import pytest

from convexity.runtime import DurableSchedule, Timer

def fail_delivery(run) -> None:
    raise ZeroDivisionError("interrupted delivery")


NOW = datetime(2026, 10, 9, 14, tzinfo=timezone.utc)


def test_restart_skips_missed_entry_and_never_duplicates(tmp_path):
    path = tmp_path / "schedule.sqlite"
    timers = (Timer(NOW, "monthly"), Timer(NOW + timedelta(days=1), "monthly"))
    with DurableSchedule(path, "stock", timers, started_at=NOW) as schedule:
        assert len(schedule.poll(NOW, lambda run: None)) == 1
        assert schedule.poll(NOW, lambda run: pytest.fail("duplicate")) == ()
    with DurableSchedule(path, "stock", timers, started_at=NOW + timedelta(days=2)) as schedule:
        assert schedule.poll(NOW + timedelta(days=2), lambda run: pytest.fail("missed entry")) == ()
        assert [r["status"] for r in schedule.status()] == ["completed", "skipped"]


def test_overdue_exit_is_delivered_with_original_key(tmp_path):
    timer = Timer(NOW, "exit")
    later = NOW + timedelta(hours=1)
    with DurableSchedule(tmp_path / "s.sqlite", "exit", (timer,), started_at=later, policy="catch_up") as schedule:
        run, = schedule.poll(later, lambda run: None)
        assert run.timer == timer
        assert run.delivered_at == later


def test_interrupted_callback_requires_explicit_reconciliation(tmp_path):
    path = tmp_path / "s.sqlite"
    timer = Timer(NOW, "entry")
    with DurableSchedule(path, "entry", (timer,), started_at=NOW) as schedule:
        with pytest.raises(ZeroDivisionError):
            schedule.poll(NOW, fail_delivery)
    with DurableSchedule(path, "entry", (timer,), started_at=NOW) as schedule:
        with pytest.raises(RuntimeError, match="reconciliation"):
            schedule.poll(NOW, lambda run: pytest.fail("blind retry"))
        schedule.resolve(timer, completed=True, detail="Order journal and broker confirmed intent.")
        assert schedule.poll(NOW, lambda run: pytest.fail("duplicate")) == ()


def test_run_keys_distinguish_delimiters_in_namespace_and_timer(tmp_path):
    keys = []
    for namespace, name in (("a/b", "c"), ("a", "b/c")):
        with DurableSchedule(tmp_path / "keys.sqlite", namespace, (Timer(NOW, name),), started_at=NOW) as schedule:
            run, = schedule.poll(NOW, lambda run: None)
            keys.append(run.key)
    assert keys[0] != keys[1]


def test_delivery_transaction_detects_competing_interrupted_claim(tmp_path, monkeypatch):
    path = tmp_path / "race.sqlite"
    timers = (Timer(NOW, "a"), Timer(NOW, "b"))
    with DurableSchedule(path, "shared", timers, started_at=NOW) as first, \
            DurableSchedule(path, "shared", timers, started_at=NOW) as competitor:
        read_status = first.status

        def snapshot_before_competing_claim():
            snapshot = read_status()
            # Another process can claim after the initial status read, before
            # this process starts its delivery transaction.
            with pytest.raises(ZeroDivisionError):
                competitor.poll(NOW, fail_delivery)
            return snapshot

        monkeypatch.setattr(first, "status", snapshot_before_competing_claim)
        with pytest.raises(RuntimeError, match="reconciliation"):
            first.poll(NOW, lambda run: pytest.fail("delivered after unresolved competing claim"))
        assert [row["status"] for row in read_status()] == ["uncertain", "pending"]


def test_async_callback_commits_completion_only_after_awaited_work(tmp_path):
    import asyncio
    path = tmp_path / "async.sqlite"
    timer = Timer(NOW, "entry")

    async def run():
        entered, finish = asyncio.Event(), asyncio.Event()
        with DurableSchedule(path, "async", (timer,), started_at=NOW) as schedule:
            async def callback(run):
                assert schedule.status()[0]["status"] == "uncertain"
                entered.set()
                await finish.wait()
            task = asyncio.create_task(schedule.poll_async(NOW, callback))
            await entered.wait()
            with DurableSchedule(path, "async", (timer,), started_at=NOW) as observer:
                assert observer.status()[0]["status"] == "uncertain"
            assert not task.done()
            finish.set()
            delivered = await task
            assert len(delivered) == 1
            assert schedule.status()[0]["status"] == "completed"
            assert await schedule.poll_async(NOW, callback) == ()
    asyncio.run(run())


def test_async_callback_failure_retains_uncertainty_across_reopen(tmp_path):
    import asyncio
    path = tmp_path / "async-failure.sqlite"
    timers = (Timer(NOW, "first"), Timer(NOW, "second"))

    async def fail(run):
        await asyncio.sleep(0)
        raise RuntimeError("callback failed after awaited work")

    async def run():
        with DurableSchedule(path, "async", timers, started_at=NOW) as schedule:
            with pytest.raises(RuntimeError, match="callback failed"):
                await schedule.poll_async(NOW, fail)
            assert [row["status"] for row in schedule.status()] == ["uncertain", "pending"]
        with DurableSchedule(path, "async", timers, started_at=NOW) as schedule:
            with pytest.raises(RuntimeError, match="reconciliation"):
                await schedule.poll_async(NOW, fail)
    asyncio.run(run())


def test_cancelled_async_callback_is_not_marked_completed(tmp_path):
    import asyncio
    path = tmp_path / "async-cancel.sqlite"
    timer = Timer(NOW, "entry")

    async def run():
        entered = asyncio.Event()
        with DurableSchedule(path, "async", (timer,), started_at=NOW) as schedule:
            async def callback(run):
                entered.set()
                await asyncio.Event().wait()
            task = asyncio.create_task(schedule.poll_async(NOW, callback))
            await entered.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert schedule.status()[0]["status"] == "uncertain"
        with DurableSchedule(path, "async", (timer,), started_at=NOW) as schedule:
            with pytest.raises(RuntimeError, match="reconciliation"):
                await schedule.poll_async(NOW, callback)
    asyncio.run(run())


def test_async_poll_limit_counts_skipped_rows_before_later_deliveries(tmp_path):
    import asyncio

    async def run():
        timers = (Timer(NOW - timedelta(seconds=1), "missed"), Timer(NOW + timedelta(seconds=2), "later"))
        with DurableSchedule(tmp_path / "limited.sqlite", "a", timers, started_at=NOW) as schedule:
            async def callback(run):
                assert run.timer.name == "later"
            assert await schedule.poll_async(NOW + timedelta(seconds=3), callback, limit=1) == ()
            assert [row["status"] for row in schedule.status()] == ["skipped", "pending"]
            assert len(await schedule.poll_async(NOW + timedelta(seconds=3), callback, limit=1)) == 1
            for value in (0, -1, True):
                with pytest.raises(ValueError, match="positive"):
                    await schedule.poll_async(NOW + timedelta(seconds=3), callback, limit=value)
    asyncio.run(run())
