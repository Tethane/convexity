import argparse
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from convexity.backtest import HistoricalData, HistoricalDataset, SessionCalendar
from convexity.data import UniverseStore, UniverseMember, UniverseSnapshot
from convexity.runtime import Bar
from examples.monthly_momentum import MonthlyMomentum, six_months_before, run_example

T = datetime(2024, 7, 1, 13, 30, tzinfo=timezone.utc)


def bar(symbol, day, close):
    end = datetime.fromisoformat(day).replace(hour=20, tzinfo=timezone.utc)
    return Bar(
        symbol, end - timedelta(hours=6), end, close, close, close, close, 100000
    )


def test_six_calendar_month_momentum_excludes_future_and_reports_missing():
    members = tuple(UniverseMember(s, s) for s in ("A", "B", "GONE"))
    as_of = pd.Timestamp(T)
    assert isinstance(as_of, pd.Timestamp)
    snapshot = UniverseSnapshot("SP500", as_of, members)
    history = {
        "A": (
            bar("A", "2023-12-29", 100),
            bar("A", "2024-06-28", 120),
            bar("A", "2024-07-02", 1),
        ),
        "B": (
            bar("B", "2023-12-29", 100),
            bar("B", "2024-06-28", 110),
            bar("B", "2024-07-02", 999),
        ),
    }
    context = SimpleNamespace(time=T, history=history, universe=lambda name: snapshot)
    strategy = MonthlyMomentum(start=date(2024, 7, 1), count=1)
    assert strategy.rank(context) == {"A": Decimal(1)}
    report = strategy.candidate_coverage[0]
    assert (report["eligible"], report["observable"], report["selected"]) == (3, 2, 1)
    assert report["missing"][0]["security_id"] == "GONE"
    assert six_months_before(date(2024, 8, 31)) == date(2024, 2, 29)


class FixtureProvider:
    source = "offline-daily-example-test"
    price_basis = "split_adjusted_price_return"

    async def fetch(self, request, *, refresh=False):
        calendar = SessionCalendar.from_exchange(
            request.calendar, request.start, request.end
        )
        bars = []
        for session in calendar.sessions:
            for symbol in request.symbols:
                rate = Decimal(".02") if symbol == "A" else Decimal(".01")
                price = Decimal(100) + (session.label - date(2023, 1, 1)).days * rate
                bars.append(
                    Bar(
                        symbol,
                        session.open,
                        session.close,
                        price,
                        price,
                        price,
                        price,
                        1000000,
                    )
                )
        data = HistoricalData(tuple(bars), self.source, self.price_basis)
        return HistoricalDataset(
            data,
            calendar,
            {"source": self.source, "price_basis": self.price_basis},
            Path("."),
        )


def test_monthly_example_end_to_end_uses_real_runtime_and_reports_coverage(tmp_path):
    pytest.importorskip("exchange_calendars")
    store_path = tmp_path / "universe.sqlite"
    with UniverseStore(store_path) as store:
        for identity, symbol in (("security-a", "A"), ("security-b", "B")):
            store.add_security(identity)
            store.add_symbol(identity, symbol, "2020-01-01", known_at="2019-01-01")
            store.add_membership("SP500", identity, "2020-01-01", known_at="2019-01-01")
    args = argparse.Namespace(
        start=date(2024, 1, 2),
        end=date(2024, 3, 8),
        store=str(store_path),
        allow_reconstructed=False,
        cache=None,
        missing_data="fail",
        output=str(tmp_path / "report"),
    )
    import asyncio

    result = asyncio.run(run_example(args, provider=FixtureProvider()))
    coverage = result["candidate_coverage"]
    portfolio = result["portfolio"]
    benchmark = result["spy"]
    assert isinstance(coverage, list)
    assert isinstance(portfolio, dict) and isinstance(benchmark, dict)
    assert len(coverage) == 3
    for row in coverage:
        assert isinstance(row, dict)
        assert row.get("selected") == 2
    assert portfolio["fees"] > 0
    assert portfolio["two_way_turnover"] > 0
    assert benchmark["fees"] > 0
    assert not result["provider_failures"]
    assert (tmp_path / "report" / "summary.json").exists()
