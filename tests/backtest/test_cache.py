from decimal import Decimal
import asyncio
from dataclasses import replace
from pathlib import Path
import sqlite3

import pytest

from convexity.backtest import (
    CachedHistoricalProvider,
    HistoricalData,
    HistoricalDataset,
    HistoryError,
    HistoryRequest,
    SessionCalendar,
)
from convexity.backtest.bar_size import intervals
from convexity.runtime import Bar


class Provider:
    source = "fixture-feed"
    price_basis = "unadjusted_no_corporate_actions"

    def __init__(self):
        self.requests = []
        self.price = 100

    async def fetch(self, request, *, refresh=False):
        self.requests.append((request, refresh))
        calendar = SessionCalendar.from_exchange(
            request.calendar, request.start, request.end
        )
        data = HistoricalData(
            tuple(
                Bar(
                    symbol,
                    start,
                    end,
                    Decimal(self.price),
                    Decimal(self.price) + 1,
                    Decimal(self.price) - 1,
                    Decimal(self.price),
                    10,
                )
                for start, end in intervals(calendar.sessions, request.bar_size)
                for symbol in request.symbols
            ),
            self.source,
            self.price_basis,
        )
        return HistoricalDataset(
            data,
            calendar,
            {"provider": self.source, "request": request.to_dict()},
            Path("fixture"),
        )


def fetch(
    cache,
    symbols=("SPY",),
    start="2024-11-27",
    end="2024-11-29",
    size="1 day",
    refresh=False,
):
    return asyncio.run(
        cache.fetch(HistoryRequest(symbols, start, end, size), refresh=refresh)
    )


def test_sqlite_reopens_without_provider_calls(tmp_path):
    provider = Provider()
    cache = CachedHistoricalProvider(provider, tmp_path / "history.sqlite3")
    first = fetch(cache)
    other = Provider()
    second = fetch(CachedHistoricalProvider(other, cache.path))
    assert second.cache_hit and not other.requests
    assert first.data == second.data and first.manifest == second.manifest
    assert cache.path.read_bytes().startswith(b"SQLite format 3")


def test_overlap_only_fetches_missing_sessions_and_symbols(tmp_path):
    provider = Provider()
    cache = CachedHistoricalProvider(provider, tmp_path / "history.sqlite3")
    fetch(cache)
    result = fetch(cache, ("SPY", "QQQ"), "2024-11-25", "2024-12-02")
    requests = [r for r, _ in provider.requests]
    assert [(r.symbols, str(r.start), str(r.end)) for r in requests[1:]] == [
        (("SPY",), "2024-11-25", "2024-11-26"),
        (("QQQ",), "2024-11-25", "2024-12-02"),
        (("SPY",), "2024-12-02", "2024-12-02"),
    ]
    assert len(result.data.bars) == 10
    assert fetch(cache, ("QQQ",), "2024-11-26", "2024-11-29").cache_hit
    assert len(provider.requests) == 4


def test_refresh_replaces_only_requested_coverage(tmp_path):
    provider = Provider()
    cache = CachedHistoricalProvider(provider, tmp_path / "history.sqlite3")
    fetch(cache, ("SPY", "QQQ"))
    provider.price = 110
    fresh = fetch(cache, start="2024-11-29", refresh=True)
    assert fresh.data.bars[0].close == 110
    assert provider.requests[-1][1]
    assert fetch(cache, ("QQQ",)).data.bars[0].close == 100
    assert [bar.close for bar in fetch(cache).data.bars] == [100, 110]


def test_corrupt_observation_fails_closed_and_refresh_repairs(tmp_path):
    provider = Provider()
    cache = CachedHistoricalProvider(provider, tmp_path / "history.sqlite3")
    fetch(cache)
    with sqlite3.connect(cache.path) as db:
        db.execute("UPDATE history_sessions SET payload='{}'")
    with pytest.raises(HistoryError, match="invalid"):
        fetch(cache)
    assert len(provider.requests) == 1
    assert not fetch(cache, refresh=True).cache_hit
    assert fetch(cache).cache_hit


def test_partition_source_resolution_calendar_and_basis(tmp_path):
    provider = Provider()
    path = tmp_path / "history.sqlite3"
    cache = CachedHistoricalProvider(provider, path)
    fetch(cache)
    fetch(cache, size="1 hour")
    fetch(CachedHistoricalProvider(provider, path, source="different-feed"))
    provider.price_basis = "split_adjusted_price_return"
    fetch(CachedHistoricalProvider(provider, path))
    provider.price_basis = "unadjusted_no_corporate_actions"
    asyncio.run(
        cache.fetch(
            HistoryRequest(("SPY",), "2024-11-27", "2024-11-29", calendar="XNAS")
        )
    )
    assert len(provider.requests) == 5


def test_multisession_aggregation_reuses_daily_and_realigns_boundaries(tmp_path):
    provider = Provider()
    cache = CachedHistoricalProvider(provider, tmp_path / "history.sqlite3")
    first = fetch(cache, start="2024-11-25", end="2024-12-02", size="2 days")
    second = fetch(cache, start="2024-11-26", end="2024-12-02", size="2 days")
    assert len(provider.requests) == 1
    assert provider.requests[0][0].bar_size == "1 day"
    assert second.cache_hit
    assert first.data.bars[0].end != second.data.bars[0].end


def test_incomplete_provider_calendar_never_committed(tmp_path):
    class Incomplete(Provider):
        async def fetch(self, request, *, refresh=False):
            result = await super().fetch(request, refresh=refresh)
            return replace(
                result, calendar=SessionCalendar(result.calendar.sessions[:-1])
            )

    cache = CachedHistoricalProvider(Incomplete(), tmp_path / "history.sqlite3")
    with pytest.raises(HistoryError, match="incomplete"):
        fetch(cache)
    with sqlite3.connect(cache.path) as db:
        assert db.execute("SELECT COUNT(*) FROM history_sessions").fetchone()[0] == 0


def test_no_partial_commit_when_a_later_acquisition_fails(tmp_path):
    class Fails(Provider):
        async def fetch(self, request, *, refresh=False):
            if len(self.requests) == 2:
                raise HistoryError("network failed")
            return await super().fetch(request, refresh=refresh)

    provider = Fails()
    cache = CachedHistoricalProvider(provider, tmp_path / "history.sqlite3")
    fetch(cache)
    with pytest.raises(HistoryError, match="network failed"):
        fetch(cache, ("QQQ", "SPY"), "2024-11-25", "2024-12-02")
    with sqlite3.connect(cache.path) as db:
        assert db.execute("SELECT COUNT(*) FROM history_sessions").fetchone()[0] == 2


def test_provenance_corruption_fails_closed_and_explicit_refresh_repairs(tmp_path):
    provider = Provider()
    cache = CachedHistoricalProvider(provider, tmp_path / "history.sqlite3")
    fetch(cache)
    with sqlite3.connect(cache.path) as db:
        db.execute("UPDATE history_acquisitions SET payload='{}'")
    with pytest.raises(HistoryError, match="invalid"):
        fetch(cache)
    assert len(provider.requests) == 1
    fetch(cache, refresh=True)
    assert fetch(cache).cache_hit


def test_future_and_empty_session_requests_never_fetch(tmp_path):
    provider = Provider()
    cache = CachedHistoricalProvider(provider, tmp_path / "history.sqlite3")
    with pytest.raises(HistoryError, match="future session"):
        fetch(cache, start="2030-01-02", end="2030-01-03")
    with pytest.raises(HistoryError, match="no exchange sessions"):
        fetch(cache, start="2024-11-28", end="2024-11-28")
    assert not provider.requests
    assert not cache.path.exists()
