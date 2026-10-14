"""The high-level API must replay from SQLite with no account or network access."""

from datetime import date
import json

from convexity import Research
from tests.backtest.test_research import BuyAtStart, FixtureProvider


def test_research_reuses_persistent_database_across_instances(tmp_path):
    provider = FixtureProvider()
    database = tmp_path / "history.sqlite3"
    first = Research("2024-01-08", "2024-01-10").run(
        BuyAtStart, provider=provider, cache=database, output=tmp_path / "first"
    )
    assert len(provider.calls) == 1
    assert database.read_bytes().startswith(b"SQLite format 3")

    async def offline(*args, **kwargs):
        raise AssertionError("A covered research run must not fetch history")

    provider.fetch = offline
    second = Research("2024-01-08", "2024-01-10").run(
        BuyAtStart, provider=provider, cache=database, output=tmp_path / "second"
    )
    assert first.to_dict() == second.to_dict()
    assert json.loads((tmp_path / "first/result.json").read_text()) == json.loads(
        (tmp_path / "second/result.json").read_text()
    )


def test_expanding_research_fetches_only_new_sessions(tmp_path):
    provider = FixtureProvider()
    database = tmp_path / "history.sqlite3"
    Research("2024-01-08", "2024-01-10").run(
        BuyAtStart, provider=provider, cache=database
    )
    Research("2024-01-09", "2024-01-12").run(
        BuyAtStart, provider=provider, cache=database
    )
    assert len(provider.calls) == 2
    assert provider.calls[1][0].start == date(2024, 1, 11)
    assert provider.calls[1][0].end == date(2024, 1, 12)
