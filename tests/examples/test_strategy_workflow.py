"""The user-facing example crosses both execution paths without an application pump."""
import asyncio
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal as D

import convexity as cx
from convexity.brokers.ibkr import Position as BrokerPosition
from examples.strategy_workflow import Momentum
from tests.runtime.test_runner import Broker, CONTRACT, Feed, app


def test_same_custom_strategy_runs_offline_and_in_managed_paper(tmp_path, monkeypatch):
    # Use a real exchange session and advance observation time deterministically.
    clock = [datetime(2025, 1, 2, 14, 33, 30, tzinfo=timezone.utc)]
    class ClockDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return clock[0].astimezone(tz) if tz else clock[0].replace(tzinfo=None)
    monkeypatch.setattr("tests.runtime.test_runner.now", lambda: clock[0])
    monkeypatch.setattr("convexity.runtime.runner._now", lambda: clock[0])
    monkeypatch.setattr("convexity.runtime.paper.datetime", ClockDateTime)
    def strategy():
        return Momentum("A", 2, 1, warmup_sessions=0)
    feed = Feed()
    feed.values[-1] = replace(feed.values[-1], high=D(101), close=D(101))
    final = feed.values[-1]
    simulation = cx.HistoricalData(tuple(feed.values) + (
        replace(final, start=final.end, end=final.end + timedelta(minutes=1), open=D(101)),))
    result = cx.Backtest(data=simulation, strategies={"momentum_A": strategy}).run()
    assert result.status == "completed"
    assert len(result.fills) == 1 and result.positions["momentum_A"]["A"].quantity == 1

    class FilledBroker(Broker):
        async def submit(self, request, *, submission_guard=None):
            if submission_guard is not None:
                submission_guard()
            await super().submit(request)
            self.fill()
            return self.journal.get(request.client_order_id)

        async def snapshot(self):
            snapshot = await super().snapshot()
            quantity = self.journal.attributed_position(self.config.account, "momentum_A", 123)
            return replace(snapshot, positions=(BrokerPosition(CONTRACT, quantity, D(100)),))

    async def run():
        brokers = []
        def factory(config, journal):
            broker = FilledBroker(config, journal)
            brokers.append(broker)
            return broker
        newest = feed.values.pop()
        clock[0] -= timedelta(minutes=1)
        runner, _ = app(tmp_path, strategy, strategies=(strategy,), subscriptions=None,
                        budgets={"momentum_A": D(1000)}, data_source=feed,
                        broker_factory=factory, decision_max_age=180)
        await runner.start()
        assert not brokers[0].sent  # Initial history is warmup, not live decisions.
        clock[0] += timedelta(minutes=1)
        feed.values.append(newest)
        async with asyncio.timeout(2):
            while not runner.status().recent_fills:
                await asyncio.sleep(.01)
        assert runner.runtime is not None
        runner.runtime.events.clear()  # Observation retention cannot hide durable fills.
        await runner.stop()
        status = runner.status()
        assert status.positions["momentum_A"]["A"].quantity == 1
        assert status.recent_fills[0].quantity == result.fills[0].quantity
        assert status.outstanding_orders == ()
        assert status.strategy_states == {"momentum_A": "stopped"}
        assert brokers[0].closed
    asyncio.run(run())


def test_root_api_exports_match_public_modules():
    assert cx.Strategy is cx.runtime.Strategy
    assert cx.Backtest is cx.backtest.Backtest
    assert cx.PaperTrading is cx.runtime.PaperTrading
    assert cx.StockSubscription is cx.runtime.StockSubscription
    assert cx.ScheduleSpec is cx.runtime.ScheduleSpec
