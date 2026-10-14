from datetime import date, timedelta
from pathlib import Path
from decimal import Decimal

import pytest

from convexity.backtest import Research
from convexity.backtest.history import HistoricalDataset, HistoryError
from convexity.backtest.data import HistoricalData, SessionCalendar
from convexity.data.universe import UniverseStore
from convexity.runtime import (
    Bar,
    StrategyContract,
    StrategyMetadata,
    UniverseSubscription,
    EventSubscription,
    Timer,
    SessionSchedule,
    SubmitOrder,
)
from convexity.strategies import TargetWeightStrategy, equal_weights


class Provider:
    def __init__(self, missing=(), native=False):
        self.calls = []
        self.missing = missing
        self.native = native

    async def fetch(self, request, *, refresh=False):
        self.calls.append(request)
        if request.symbols[0] in self.missing:
            raise HistoryError("delisted history unavailable")
        calendar = SessionCalendar.from_exchange(
            request.calendar, request.start, request.end
        )
        data = HistoricalData(
            tuple(
                Bar(
                    symbol,
                    s.open,
                    s.close,
                    Decimal(100),
                    Decimal(100),
                    Decimal(100),
                    Decimal(100),
                    100000,
                )
                for s in calendar.sessions
                for symbol in request.symbols
            ),
            source="Alpaca:test:native-daily" if self.native else "fixture",
        )
        return HistoricalDataset(data, calendar, {}, Path("/fixture"))


def store_at(tmp_path):
    path = tmp_path / "universe.sqlite3"
    with UniverseStore(path) as store:
        for identity, symbol in [("first", "AAA"), ("second", "BBB")]:
            store.add_security(identity)
            store.add_symbol(identity, symbol, "2023-01-01", known_at="2023-01-01")
        store.add_membership(
            "SP500", "first", "2023-01-01", "2024-01-10", known_at="2023-01-01"
        )
        store.add_membership("SP500", "second", "2024-01-10", known_at="2024-01-09")
    return path


def strategy_at(path, snapshots):
    class Portfolio(TargetWeightStrategy):
        metadata = StrategyMetadata("portfolio")
        contract = StrategyContract(
            universes=(UniverseSubscription("SP500", path),),
            subscriptions=(
                EventSubscription(
                    Timer,
                    schedule=SessionSchedule(
                        "daily", "rebalance", date(2024, 1, 8), at="open"
                    ),
                ),
            ),
            warmup_sessions=2,
        )

        def __init__(self):
            super().__init__(self.select, cash_buffer=0.1)

        def select(self, context):
            members = context.universe("SP500").security_ids
            snapshots.append((context.time, members, dict(context.history)))
            assert all(
                bar.end <= context.time
                for history in context.history.values()
                for bar in history
            )
            return equal_weights(members)

    return Portfolio


def test_dynamic_membership_and_normal_orders_without_static_constituents(tmp_path):
    path = store_at(tmp_path)
    snapshots, provider = [], Provider()
    result = Research("2024-01-08", "2024-01-12").run(
        strategy_at(path, snapshots),
        provider=provider,
        cache=None,
        execution="frictionless",
        initial_cash=1000,
    )
    assert result.status == "completed"
    assert snapshots[0][1] == ("first",)
    assert snapshots[-1][1] == ("second",)
    assert "second" not in snapshots[0][2]
    assert result.positions["portfolio"]["second"].quantity == 9
    assert result.positions["portfolio"]["first"].quantity == 0
    assert {fill.symbol for fill in result.fills} == {"first", "second"}
    assert {r.symbols for r in provider.calls} == {("AAA",), ("BBB",)}


def test_missing_data_preserves_eligible_and_reports_exclusion(tmp_path):
    path = store_at(tmp_path)
    result = Research("2024-01-08", "2024-01-12").run(
        strategy_at(path, []),
        provider=Provider(("BBB",)),
        cache=None,
        missing_data="exclude",
        execution="frictionless",
    )
    coverage = result.config["metadata"]["research"]["coverage"]
    assert coverage[-1]["eligible"] == 1
    assert coverage[-1]["observable"] == 0
    assert coverage[-1]["missing"] == ["second"]
    assert result.config["metadata"]["research"]["provider_failures"]
    with pytest.raises(HistoryError, match="delisted"):
        Research("2024-01-08", "2024-01-12").run(
            strategy_at(path, []), provider=Provider(("BBB",)), cache=None
        )


def test_native_daily_observations_are_delayed_to_next_open(tmp_path):
    path = store_at(tmp_path)
    snapshots = []
    result = Research("2024-01-08", "2024-01-12").run(
        strategy_at(path, snapshots),
        provider=Provider(native=True),
        cache=None,
        execution="frictionless",
        initial_cash=1000,
    )
    from convexity.runtime import Bars

    observations = [e for e in result.events if isinstance(e, Bars)]
    assert observations and all(e.delayed for e in observations)
    assert all(
        bar.end < event.time for event in observations for bar in event.bars.values()
    )
    assert all(
        fill.occurred_at >= result.orders[0].accepted_at for fill in result.fills
    )


def test_future_member_orders_cannot_bypass_dynamic_contract(tmp_path):
    from convexity.backtest import BacktestFailure
    from convexity.runtime import Strategy

    path = store_at(tmp_path)

    class Lookahead(Strategy):
        metadata = StrategyMetadata("bad")
        contract = StrategyContract(universes=(UniverseSubscription("SP500", path),))

        def on_start(self, context):
            # Internally prefetched bars must never confer trading eligibility.
            return SubmitOrder("second", "BUY", 1)

    with pytest.raises(BacktestFailure, match="outside the strategy contract"):
        Research("2024-01-08", "2024-01-12").run(
            Lookahead, provider=Provider(), cache=None
        )


def test_renamed_ticker_history_preserves_one_security_position(tmp_path):
    path = tmp_path / "rename.sqlite3"
    with UniverseStore(path) as store:
        store.add_security("stable")
        store.add_symbol(
            "stable", "OLD", "2023-01-01", "2024-01-10", known_at="2023-01-01"
        )
        store.add_symbol("stable", "NEW", "2024-01-10", known_at="2024-01-09")
        store.add_membership("SP500", "stable", "2023-01-01", known_at="2023-01-01")
    snapshots, provider = [], Provider()
    result = Research("2024-01-08", "2024-01-12").run(
        strategy_at(path, snapshots),
        provider=provider,
        cache=None,
        execution="frictionless",
        initial_cash=1000,
    )
    assert {fill.symbol for fill in result.fills} == {"stable"}
    assert result.positions["portfolio"]["stable"].quantity == 9
    assert {r.symbols[0] for r in provider.calls} == {"OLD", "NEW"}
    assert len(snapshots[-1][2]["stable"]) >= 5


def test_default_daily_provider_reuses_native_prefetch_cache(tmp_path, monkeypatch):
    from convexity.backtest.alpaca import AlpacaCredentials
    from convexity.backtest.alpaca_daily import AlpacaDailyHistoricalData
    from zoneinfo import ZoneInfo
    from datetime import datetime, time

    path, cache = store_at(tmp_path), tmp_path / "history.sqlite3"
    source = AlpacaDailyHistoricalData(
        AlpacaCredentials("fixture", "fixture"), request_interval=0
    )

    def page(params, credentials):
        calendar = SessionCalendar.from_exchange(
            "XNYS", params["start"][:10], params["end"][:10]
        )
        rows = [
            dict(
                t=datetime.combine(
                    s.label, time(), ZoneInfo("America/New_York")
                ).isoformat(),
                o=100,
                h=100,
                l=100,
                c=100,
                v=100000,
            )
            for s in calendar.sessions
        ]
        return dict(bars={params["symbols"]: rows}, next_page_token=None)

    monkeypatch.setattr(source, "_request_page", page)
    Research("2024-01-08", "2024-01-12").run(
        strategy_at(path, []), provider=source, cache=cache, execution="frictionless"
    )

    def forbidden():
        raise AssertionError(
            "Cache replay must not request Alpaca credentials or the network"
        )

    monkeypatch.setattr(AlpacaCredentials, "load", forbidden)
    result = Research("2024-01-08", "2024-01-12").run(
        strategy_at(path, []), cache=cache, execution="frictionless"
    )
    assert result.status == "completed"
    assert (
        result.config["metadata"]["research"]["observation_availability"]
        == "next session open"
    )


def test_native_daily_rejects_close_time_decisions():
    from datetime import time
    from convexity.runtime import (
        Strategy,
        BarSubscription,
        EventSubscription,
        ClockSchedule,
    )

    class CloseDecision(Strategy):
        metadata = StrategyMetadata("close-decision")
        contract = StrategyContract(
            ("AAA",),
            (
                BarSubscription(("AAA",)),
                EventSubscription(
                    Timer, schedule=ClockSchedule("rebalance", at=time(16))
                ),
            ),
        )

    with pytest.raises(ValueError, match="requires session-open decisions"):
        Research("2024-01-08", "2024-01-12").run(
            CloseDecision, provider=Provider(native=True), cache=None
        )


def _single_member_store(tmp_path):
    path = tmp_path / "single.sqlite3"
    with UniverseStore(path) as store:
        store.add_security("first")
        store.add_symbol("first", "AAA", "2023-01-01", known_at="2023-01-01")
        store.add_membership("SP500", "first", "2023-01-01", known_at="2023-01-01")
    return path


def test_exclusion_salvages_sessions_around_one_missing_day(tmp_path):
    class GapProvider(Provider):
        async def fetch(self, request, *, refresh=False):
            if request.start <= date(2024, 1, 10) <= request.end:
                self.calls.append(request)
                raise HistoryError(
                    "Incomplete historical coverage: missing daily bar on 2024-01-10"
                )
            return await super().fetch(request, refresh=refresh)

    provider, snapshots = GapProvider(), []
    result = Research("2024-01-08", "2024-01-12").run(
        strategy_at(_single_member_store(tmp_path), snapshots),
        provider=provider,
        cache=tmp_path / "cache.sqlite3",
        missing_data="exclude",
        execution="frictionless",
    )
    coverage = result.config["metadata"]["research"]["coverage"]
    assert [(row["date"], row["observable"]) for row in coverage] == [
        ("2024-01-08", 1),
        ("2024-01-09", 1),
        ("2024-01-10", 0),
        ("2024-01-11", 1),
        ("2024-01-12", 1),
    ]
    assert {bar.end.date() for bar in snapshots[-1][2]["first"]} >= {
        date(2024, 1, 9),
        date(2024, 1, 11),
    }
    assert len(provider.calls) < 20
    assert any(
        row["start"] == row["end"] == "2024-01-10"
        for row in result.config["metadata"]["research"]["provider_failures"]
    )


def test_exclusion_recovers_existing_cache_without_retrying_credentials(tmp_path):
    import asyncio
    from convexity.backtest import CachedHistoricalProvider, HistoryRequest

    class OfflineProvider(Provider):
        unavailable = False

        async def fetch(self, request, *, refresh=False):
            if self.unavailable:
                self.calls.append(request)
                raise HistoryError(
                    "Credentials unavailable; do not retry this infrastructure failure"
                )
            return await super().fetch(request, refresh=refresh)

    source = OfflineProvider()
    cached = CachedHistoricalProvider(source, tmp_path / "cache.sqlite3")

    async def seed():
        await cached.fetch(HistoryRequest(("AAA",), "2024-01-04", "2024-01-09"))
        await cached.fetch(HistoryRequest(("AAA",), "2024-01-11", "2024-01-12"))

    asyncio.run(seed())
    source.unavailable = True
    result = Research("2024-01-08", "2024-01-12").run(
        strategy_at(_single_member_store(tmp_path), []),
        provider=cached,
        cache=None,
        missing_data="exclude",
        execution="frictionless",
    )
    coverage = result.config["metadata"]["research"]["coverage"]
    assert [row["observable"] for row in coverage] == [1, 1, 0, 1, 1]
    assert (
        len(source.calls) == 3
    )  # two seed requests and only one unavailable gap request
