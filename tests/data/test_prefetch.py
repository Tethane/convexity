import asyncio
from dataclasses import replace
from datetime import date
import sqlite3
import json

import pandas as pd
import pytest

from convexity.backtest import HistoryError
from convexity.data.coverage import build_coverage, enforce_coverage
from convexity.data.prefetch import (
    DailyPrefetcher,
    PrefetchConfig,
    PermanentHistoryError,
)
from convexity.data.universe import HistoricalMember
from tests.backtest.test_cache import Provider


def member(
    sid="security-a",
    symbol="AAA",
    start="2024-11-25",
    end="2024-12-03",
    universes=("SP500",),
):
    start_at = pd.Timestamp(start, tz="UTC")
    end_at = pd.Timestamp(end, tz="UTC")
    assert isinstance(start_at, pd.Timestamp) and isinstance(end_at, pd.Timestamp)
    return HistoricalMember(
        sid,
        symbol,
        start_at,
        end_at,
        universes,
    )


def config(**kwargs):
    return replace(
        PrefetchConfig(retries=0, primary_interval=0, fallback_interval=0, backoff=0),
        **kwargs,
    )


def run(downloader, members=None, **kwargs):
    return asyncio.run(
        downloader.run(members or (member(),), "2024-11-25", "2024-12-02", **kwargs)
    )


def test_chunks_persist_and_resume_only_missing_dates(tmp_path):
    source = Provider()
    fetcher = DailyPrefetcher(
        source, path=tmp_path / "cache.db", config=config(chunk_sessions=2)
    )
    first = run(fetcher)
    assert first["complete"] and len(source.requests) == 3
    with sqlite3.connect(fetcher.path) as db:
        assert db.execute("SELECT COUNT(*) FROM history_sessions").fetchone()[0] == 5
        db.execute("DELETE FROM history_sessions WHERE session='2024-11-26'")
    run(DailyPrefetcher(source, path=fetcher.path, config=config()))
    assert len(source.requests) == 4
    request = source.requests[-1][0]
    assert request.start == request.end == date(2024, 11, 26)


def test_renames_reuse_and_union_are_identity_safe(tmp_path):
    source = Provider()
    fetcher = DailyPrefetcher(source, path=tmp_path / "cache.db", config=config())
    members = (
        member(end="2024-11-27"),
        member(symbol="NEW", start="2024-11-27"),
        member(sid="security-b", start="2024-11-27", universes=("NASDAQ100",)),
        member(end="2024-11-27", universes=("NASDAQ100",)),
    )
    report = run(fetcher, members)
    assert report["complete"] and len(source.requests) == 3
    rows = {(row["date"], row["universe"]): row for row in report["by_date"]}
    assert rows["2024-11-25", "UNION"]["eligible"] == 1
    assert rows["2024-11-27", "UNION"]["eligible"] == 2
    with sqlite3.connect(fetcher.path) as db:
        assert db.execute(
            "SELECT DISTINCT symbol FROM prefetch_observations WHERE security_id='security-a'"
        ).fetchall() == [("AAA",), ("NEW",)]


def test_fallback_cached_without_repeating_primary_failure(tmp_path):
    class Missing(Provider):
        source = "missing-primary"

        async def fetch(self, request, *, refresh=False):
            self.requests.append((request, refresh))
            raise HistoryError("history currently unavailable")

    primary, fallback = Missing(), Provider()
    fetcher = DailyPrefetcher(
        primary, fallback, path=tmp_path / "cache.db", config=config()
    )
    report = run(fetcher)
    assert report["complete"] and len(primary.requests) == len(fallback.requests) == 1
    assert report["failures"] and all(
        not row["permanent"] for row in report["failures"]
    )
    run(DailyPrefetcher(primary, fallback, path=fetcher.path, config=config()))
    assert len(primary.requests) == len(fallback.requests) == 1


def test_partial_missing_chunk_keeps_eligible_and_recovers_accessible_sessions(
    tmp_path,
):
    class Gap(Provider):
        async def fetch(self, request, *, refresh=False):
            if request.start <= date(2024, 11, 26) <= request.end:
                raise HistoryError(
                    "provider gap; not evidence of permanent unavailability"
                )
            return await super().fetch(request, refresh=refresh)

    source = Gap()
    fetcher = DailyPrefetcher(source, path=tmp_path / "cache.db", config=config())
    report = run(fetcher)
    assert not report["complete"]
    missing = [row for row in report["by_date"] if row["missing"]]
    assert {row["date"] for row in missing} == {"2024-11-26"}
    assert all(row["eligible"] == 1 for row in report["by_date"])
    assert len({str(r.start) for r, _ in source.requests}) >= 2
    with pytest.raises(HistoryError, match="incomplete"):
        enforce_coverage(report, "fail")


def test_permanent_requires_evidence_and_skips_repeated_provider_calls(tmp_path):
    class Unavailable(Provider):
        async def fetch(self, request, *, refresh=False):
            self.requests.append((request, refresh))
            raise PermanentHistoryError(
                "coverage not licensed",
                evidence="fixture provider inventory explicitly denies requested period",
            )

    source = Unavailable()
    fetcher = DailyPrefetcher(source, path=tmp_path / "cache.db", config=config())
    assert not run(fetcher)["complete"]
    assert len(source.requests) == 1
    run(fetcher)
    assert len(source.requests) == 1
    run(fetcher, retry_permanent=True)
    assert len(source.requests) == 2
    with pytest.raises(ValueError, match="evidence"):
        PermanentHistoryError("unknown", evidence="")


def test_retry_then_success_and_transient_failures_retry_next_run(tmp_path):
    class Flaky(Provider):
        failures = 1

        async def fetch(self, request, *, refresh=False):
            if self.failures:
                self.failures -= 1
                raise HistoryError("timeout")
            return await super().fetch(request, refresh=refresh)

    source = Flaky()
    fetcher = DailyPrefetcher(
        source, path=tmp_path / "cache.db", config=config(retries=1)
    )
    assert run(fetcher)["complete"]
    assert len(source.requests) == 1


def test_cancellation_preserves_completed_chunks(tmp_path):
    class Blocking(Provider):
        block = True

        async def fetch(self, request, *, refresh=False):
            if self.requests and self.block:
                assert blocked is not None
                blocked.set()
                await asyncio.sleep(60)
            return await super().fetch(request, refresh=refresh)

    async def interrupted():
        nonlocal blocked
        blocked = asyncio.Event()
        fetcher = DailyPrefetcher(
            Blocking(), path=tmp_path / "cache.db", config=config(chunk_sessions=1)
        )
        task = asyncio.create_task(fetcher.run((member(),), "2024-11-25", "2024-12-02"))
        await blocked.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return fetcher.path

    blocked = None
    path = asyncio.run(interrupted())
    source = Blocking()
    source.block = False
    assert run(DailyPrefetcher(source, path=path, config=config()))["complete"]
    assert source.requests[0][0].start == date(2024, 11, 26)


def test_candidate_coverage_distinct_from_observation_coverage():
    result = build_coverage(
        {"2020-01-02": {"SP500": {"a", "b", "c"}}},
        {"2020-01-02": {"a", "b"}},
        candidates={"2020-01-02": {"a"}},
        selected={"2020-01-02": set()},
    )
    row = result["by_date"][0]
    assert row["eligible"] == 3 and row["observable"] == 2 and row["candidates"] == 1
    assert row["coverage_ratio"] == 2 / 3 and row["candidate_coverage_ratio"] == 1 / 3
    assert row["selected"] == 0 and row["missing"] == ["c"]


def test_lazy_intraday_cache_uses_only_missing_sessions(tmp_path):
    from convexity.backtest import CachedHistoricalProvider, HistoryRequest

    source = Provider()
    cache = CachedHistoricalProvider(source, tmp_path / "cache.db")
    asyncio.run(
        cache.fetch(HistoryRequest(("AAA",), "2024-11-25", "2024-11-26", "1 hour"))
    )
    asyncio.run(
        cache.fetch(HistoryRequest(("AAA",), "2024-11-26", "2024-11-27", "1 hour"))
    )
    assert len(source.requests) == 2
    assert (
        source.requests[-1][0].start == source.requests[-1][0].end == date(2024, 11, 27)
    )


def test_acquisition_can_include_verified_pre_membership_lookback(tmp_path):
    source = Provider()
    fetcher = DailyPrefetcher(source, path=tmp_path / "cache.db", config=config())
    report = run(
        fetcher, (member(start="2024-11-27"),), acquisition_members=(member(),)
    )
    assert source.requests[0][0].start == date(2024, 11, 25)
    assert min(row["date"] for row in report["by_date"]) == "2024-11-27"


def test_provider_request_pacing_applies_across_symbols(tmp_path):
    class Timed(Provider):
        def __init__(self):
            super().__init__()
            self.times = []

        async def fetch(self, request, *, refresh=False):
            self.times.append(asyncio.get_running_loop().time())
            return await super().fetch(request, refresh=refresh)

    source = Timed()
    fetcher = DailyPrefetcher(
        source, path=tmp_path / "cache.db", config=config(primary_interval=0.02)
    )
    run(fetcher, (member(), member(sid="other", symbol="BBB")))
    assert len(source.times) == 2 and source.times[1] - source.times[0] >= 0.019


def test_cli_uses_union_identity_intervals_and_writes_report(tmp_path, monkeypatch):
    from convexity.cli import main
    from convexity.data.universe import UniverseStore
    from convexity.backtest import alpaca_daily

    path = tmp_path / "universes.db"
    with UniverseStore(path) as store:
        store.add_security("stable-a")
        store.add_symbol("stable-a", "AAA", "2024-01-01", known_at="2023-12-01")
        store.add_membership("SP500", "stable-a", "2024-11-27", known_at="2024-11-01")
    source = Provider()
    monkeypatch.setattr(alpaca_daily, "AlpacaDailyHistoricalData", lambda: source)
    report = tmp_path / "coverage.json"
    argv = [
        "data",
        "prefetch",
        "--universe",
        "SP500",
        "--start",
        "2024-11-25",
        "--end",
        "2024-12-02",
        "--store",
        str(path),
        "--cache",
        str(tmp_path / "cache.db"),
        "--report",
        str(report),
        "--no-ibkr",
        "--resume",
    ]
    assert main(argv) == 0
    assert source.requests[0][0].start == date(2024, 11, 25)
    payload = json.loads(report.read_text())
    assert payload["complete"]
    assert min(row["date"] for row in payload["by_date"]) == "2024-11-27"
    assert main(argv) == 0 and len(source.requests) == 1


def test_resume_does_not_reuse_a_different_feed_namespace(tmp_path):
    source = Provider()
    fetcher = DailyPrefetcher(source, path=tmp_path / "cache.db", config=config())
    run(fetcher)
    other = Provider()
    other.source = "different-feed"
    assert run(DailyPrefetcher(other, path=fetcher.path, config=config()))["complete"]
    assert len(other.requests) == 1


def test_corrected_ticker_mapping_cannot_inherit_other_symbol_coverage(tmp_path):
    class MappingProvider(Provider):
        async def fetch(self, request, *, refresh=False):
            if "NEW" in request.symbols:
                raise HistoryError("corrected historical symbol unavailable")
            return await super().fetch(request, refresh=refresh)

    source = MappingProvider()
    fetcher = DailyPrefetcher(source, path=tmp_path / "cache.db", config=config())
    assert run(fetcher)["complete"]
    report = run(fetcher, (member(symbol="NEW"),))
    assert not report["complete"]
    assert all(row["observable"] == 0 for row in report["by_date"])
