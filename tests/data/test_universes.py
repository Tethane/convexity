import json

import pandas as pd
import pytest

from convexity.data import (
    Security,
    UniverseStore,
    UniverseError,
    KnowledgeUnavailableError,
    import_snapshot_csv,
    universe,
)
from convexity.data.universe_sources import _parse_n100_yaml, fetch_candidate


def seed(store):
    store.add_security(Security("meta-a", "meta", "Meta Class A"))
    store.add_security(Security("old-xyz", status="delisted"))
    store.add_security(Security("new-xyz"))
    store.add_symbol("meta-a", "FB", "2012-05-18", "2022-06-09", known_at="2012-05-17")
    store.add_symbol("meta-a", "META", "2022-06-09", known_at="2022-05-31")
    store.add_symbol(
        "old-xyz", "XYZ", "2010-01-01", "2020-01-01", known_at="2010-01-01"
    )
    store.add_symbol("new-xyz", "XYZ", "2021-01-01", known_at="2021-01-01")
    store.add_membership("SP500", "meta-a", "2013-12-23", known_at="2013-12-13")
    store.add_membership("NASDAQ100", "meta-a", "2012-12-12", known_at="2012-12-05")
    store.add_membership(
        "SP500", "old-xyz", "2015-01-01", "2019-12-01", known_at="2014-12-30"
    )
    store.add_membership("SP500", "new-xyz", "2021-01-01", known_at="2020-12-30")


def test_security_identity_reuse_rename_boundaries_persistence(tmp_path):
    path = tmp_path / "universes.sqlite3"
    with UniverseStore(path) as store:
        seed(store)
        assert store.resolve_symbol("XYZ", "2018").security_id == "old-xyz"
        assert store.resolve_symbol("XYZ", "2022").security_id == "new-xyz"
        with pytest.raises(UniverseError):
            store.resolve_symbol("XYZ", "2020-06-01")
        assert store.symbol_at("meta-a", "2022-06-08") == "FB"
        assert store.symbol_at("meta-a", "2022-06-09") == "META"
        assert len(store.symbol_history("meta-a", "2022-06-08", "2022-06-10")) == 2
    assert universe("SP500", "2015-06-01", store=path).symbols == ("FB", "XYZ")


def test_snapshots_union_intersection_and_historical_intervals():
    with UniverseStore(":memory:") as store:
        seed(store)
        sp = store.resolve("SP500", "2015-06-01")
        nq = store.resolve("NASDAQ100", "2015-06-01")
        assert len(sp | nq) == 2
        assert (sp & nq).security_ids == ("meta-a",)
        assert "old-xyz" not in store.resolve("SP500", "2019-12-01").security_ids
        with pytest.raises(UniverseError):
            _ = sp | store.resolve("SP500", "2015-06-02")
        history = store.historical_members(
            ["SP500", "NASDAQ100"], "2022-06-08", "2022-06-10"
        )
        meta = [r for r in history if r.security_id == "meta-a"]
        assert [r.symbol for r in meta] == ["FB", "META"]
        assert meta[0].end == pd.Timestamp("2022-06-09", tz="UTC")
        assert meta[0].universes == ("NASDAQ100", "SP500")


def test_known_after_decision_fails_instead_of_silently_excluding():
    with UniverseStore(":memory:") as store:
        store.add_security("a")
        store.add_symbol("a", "AAA", "2015-01-01", known_at="2020-01-01")
        store.add_membership("SP500", "a", "2015-01-01", known_at="2020-01-01")
        with pytest.raises(KnowledgeUnavailableError):
            store.resolve("SP500", "2016-01-01")
        with pytest.raises(KnowledgeUnavailableError):
            store.resolve("SP500", "2016-01-01", known_at="2020-01-01")
        snapshot = store.resolve("SP500", "2016-01-01", allow_reconstructed=True)
        assert snapshot.symbols == ("AAA",)
        assert snapshot.warnings


def test_conflicting_symbol_or_membership_rejected():
    with UniverseStore(":memory:") as store:
        seed(store)
        with pytest.raises(UniverseError):
            store.add_symbol("new-xyz", "XYZ", "2018-01-01", known_at="2017-01-01")
        with pytest.raises(UniverseError):
            store.add_membership("SP500", "meta-a", "2015", known_at="2015")


CSV = (
    b'date,tickers\n2015-01-01,"AAA,OLD"\n2015-02-01,"AAA,NEW"\n2015-03-01,"AAA,OLD"\n'
)


def test_import_requires_identity_or_explicit_provisional_and_preserves_raw():
    with UniverseStore(":memory:") as store:
        with pytest.raises(UniverseError, match="identity"):
            import_snapshot_csv(
                store,
                CSV,
                name="SP500",
                source="test",
                known_at="2026-01-01",
                coverage_end="2015-04-01",
            )
        result = import_snapshot_csv(
            store,
            CSV,
            name="SP500",
            source="test",
            known_at="2026-01-01",
            coverage_end="2015-04-01",
            allow_provisional=True,
        )
        assert result["securities"] == 4
        assert (
            bytes(
                store.connection.execute("SELECT raw FROM universe_sources").fetchone()[
                    0
                ]
            )
            == CSV
        )
        with pytest.raises(KnowledgeUnavailableError):
            store.resolve("SP500", "2015-01-10")
        january = store.resolve("SP500", "2015-01-10", allow_reconstructed=True)
        march = store.resolve("SP500", "2015-03-10", allow_reconstructed=True)
        assert january.symbols == march.symbols
        assert january.security_ids != march.security_ids
        with pytest.raises(UniverseError, match="coverage"):
            store.resolve("SP500", "2015-04-01", allow_reconstructed=True)


def test_verified_rename_preserves_identity_and_repeated_symbol_requires_episode_map():
    raw = b"date,tickers\n2022-06-08,FB\n2022-06-09,META\n"
    with UniverseStore(":memory:") as store:
        import_snapshot_csv(
            store,
            raw,
            name="NASDAQ100",
            source="test",
            known_at="2026-01-01",
            coverage_end="2022-06-10",
            identity_map={"FB": "meta-a", "META": "meta-a"},
        )
        rows = store.historical_members(
            "NASDAQ100", "2022-06-08", "2022-06-10", allow_reconstructed=True
        )
        assert len(rows) == 2
        assert {r.security_id for r in rows} == {"meta-a"}
    with UniverseStore(":memory:") as store:
        with pytest.raises(UniverseError, match="OLD@"):
            import_snapshot_csv(
                store,
                CSV,
                name="SP500",
                source="test",
                known_at="2026-01-01",
                coverage_end="2015-04-01",
                identity_map={"AAA": "a", "OLD": "old", "NEW": "new"},
            )


def test_import_atomic_on_bad_identity_and_archives_conflicting_update():
    with UniverseStore(":memory:") as store:
        with pytest.raises(UniverseError):
            import_snapshot_csv(
                store,
                CSV,
                name="SP500",
                source="test",
                known_at="2026-01-01",
                coverage_end="2015-04-01",
                identity_map={
                    "AAA": "same",
                    "OLD@2015-01-01": "same",
                    "OLD@2015-03-01": "old",
                    "NEW": "new",
                },
            )
        assert not store.connection.execute("SELECT * FROM securities").fetchall()
        import_snapshot_csv(
            store,
            CSV,
            name="SP500",
            source="test",
            known_at="2026-01-01",
            coverage_end="2015-04-01",
            allow_provisional=True,
        )
        with pytest.raises(UniverseError, match="archived"):
            import_snapshot_csv(
                store,
                CSV.replace(b"NEW", b"CHANGED"),
                name="SP500",
                source="test",
                known_at="2026-02-01",
                coverage_end="2015-04-01",
                allow_provisional=True,
            )
        assert len(store.sources("SP500")) == 2
        assert (
            "NEW"
            in store.resolve("SP500", "2015-02-01", allow_reconstructed=True).symbols
        )


def test_import_duplicate_and_out_of_order_rows_fail():
    with UniverseStore(":memory:") as store:
        for raw in (
            b'date,tickers\n2015-01-01,"A,A"',
            b"date,tickers\n2015-02-01,A\n2015-01-01,B",
        ):
            with pytest.raises(UniverseError):
                import_snapshot_csv(
                    store,
                    raw,
                    name="SP500",
                    source="test",
                    known_at="2026",
                    coverage_end="2016",
                    allow_provisional=True,
                )


def test_missing_symbol_interval_does_not_drop_member():
    with UniverseStore(":memory:") as store:
        store.add_security("delisted")
        store.add_membership("SP500", "delisted", "2015", "2016", known_at="2014")
        with pytest.raises(UniverseError, match="historical symbol"):
            store.historical_members("SP500", "2015", "2016")


def test_source_license_gate_without_network():
    with UniverseStore(":memory:") as store:
        with pytest.raises(UniverseError, match="license"):
            fetch_candidate(store, "thuningxu/sp500nq100", "NASDAQ100")


def test_n100_yaml_strict_parser_and_verified_effective_date():
    raw = b"---\nyear: 2022\ntickers_on_Jan_1:\n  - FB\nchanges:\n  '2022-06-09':\n    difference:\n      - FB\n    union:\n      - META\n"
    snapshots = _parse_n100_yaml(raw)
    assert snapshots[-1] == (pd.Timestamp("2022-06-09", tz="UTC"), {"META"})
    with pytest.raises(UniverseError):
        _parse_n100_yaml(raw.replace(b"      - FB", b"      - UNKNOWN"))


def test_subsecond_half_open_boundary():
    with UniverseStore(":memory:") as store:
        store.add_security("s")
        store.add_symbol("s", "S", "2020-01-01", "2020-01-02", known_at="2019")
        store.add_membership("SP500", "s", "2020-01-01", "2020-01-02", known_at="2019")
        store.set_coverage("SP500", "2020-01-01", "2020-01-02", known_at="2019")
        assert len(store.resolve("SP500", "2020-01-01T23:59:59.999999999Z")) == 1
        with pytest.raises(UniverseError):
            store.resolve("SP500", "2020-01-02")


def test_removal_knowledge_is_independent_of_addition_knowledge():
    with UniverseStore(":memory:") as store:
        store.add_security("s")
        store.add_symbol("s", "S", "2014", known_at="2014")
        store.add_membership(
            "SP500", "s", "2015", "2016", known_at="2014", end_known_at="2016-01-03"
        )
        assert len(store.resolve("SP500", "2015-06-01")) == 1
        with pytest.raises(KnowledgeUnavailableError, match="Removal"):
            store.resolve("SP500", "2016-01-02")
        assert len(store.resolve("SP500", "2016-01-03")) == 0
        with pytest.raises(KnowledgeUnavailableError, match="Removal"):
            store.historical_members("SP500", "2015", "2017")


def test_verified_identity_cross_index_import_deduplicates_different_start_dates():
    with UniverseStore(":memory:") as store:
        for name, start in [("SP500", "2015-01-01"), ("NASDAQ100", "2015-02-01")]:
            import_snapshot_csv(
                store,
                f"date,tickers\n{start},AAPL\n".encode(),
                name=name,
                source="test",
                known_at="2026",
                coverage_end="2016",
                identity_map={"AAPL": "apple-common"},
            )
        union = store.resolve(
            "SP500", "2015-06-01", allow_reconstructed=True
        ) | store.resolve("NASDAQ100", "2015-06-01", allow_reconstructed=True)
        assert union.security_ids == ("apple-common",)


def test_historical_union_splits_overlaps_without_duplicate_identity_intervals():
    with UniverseStore(":memory:") as store:
        store.add_security("s")
        store.add_symbol("s", "S", "2010", known_at="2009")
        store.add_membership("SP500", "s", "2015-01-01", "2015-04-01", known_at="2014")
        store.add_membership(
            "NASDAQ100", "s", "2015-02-01", "2015-05-01", known_at="2014"
        )
        rows = store.historical_members(
            ["SP500", "NASDAQ100"], "2015-01-01", "2015-05-01"
        )
        assert len(rows) == 3
        assert [r.universes for r in rows] == [
            ("SP500",),
            ("NASDAQ100", "SP500"),
            ("NASDAQ100",),
        ]
        assert all(a.end == b.start for a, b in zip(rows, rows[1:]))


def test_fetch_candidate_pins_commit_and_retains_original_bytes_and_license(
    monkeypatch,
):
    import convexity.data.universe_sources as sources

    requested = []

    def fake_read(url):
        requested.append(url)
        if "/commits/" in url:
            return json.dumps(
                {
                    "sha": "deadbeef",
                    "commit": {"committer": {"date": "2026-01-01T00:00:00Z"}},
                }
            ).encode()
        assert "/deadbeef/" in url
        if url.endswith("README.md"):
            return b"Audited fixture source"
        if url.endswith("LICENSE"):
            return b"MIT License fixture"
        return CSV

    monkeypatch.setattr(sources, "_read", fake_read)
    with UniverseStore(":memory:") as store:
        result = fetch_candidate(
            store, "fja05680/sp500", "SP500", allow_provisional=True
        )
        assert result["snapshots"] == 3
        provenance = store.sources("SP500")[0]
        assert provenance["metadata"]["commit"] == "deadbeef"
        assert provenance["metadata"]["license_text"] == "MIT License fixture"
        assert (
            store.connection.execute("SELECT raw FROM universe_sources").fetchone()[0]
            == CSV
        )
        assert len(requested) == 4
