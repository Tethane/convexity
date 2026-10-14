# Historical universes and security identity

`cx.data.universe` resolves eligible membership independently of price availability.
No constituent is deleted because Alpaca or IBKR cannot retrieve its prices.
The persistent default database is `.convexity/universes.sqlite3`; pass `store=`
to use another database or an open `UniverseStore`.

```python
import convexity as cx
from convexity.data import UniverseStore

with UniverseStore("reference.sqlite3") as reference:
    sp500 = cx.data.universe("SP500", as_of="2015-06-01", store=reference)
    nasdaq = cx.data.universe("NASDAQ100", as_of="2015-06-01", store=reference)
    combined = sp500 | nasdaq
    common = sp500 & nasdaq
    print(combined.security_ids, combined.symbols)
```

An empty database raises an explicit error. Import reference data first. Every
interval is `[start, end)`, using UTC timestamps. Date-only inputs mean midnight
UTC. Public date-only lists establish a session's membership, not the precise
intraday time of an announcement. Snapshots at different decision timestamps
cannot be combined. Union/intersection deduplicate by **security ID**, never by
symbol or issuer. Distinct share classes remain distinct securities.

## Verified reference records

```python
from convexity.data import Security, UniverseStore

with UniverseStore("reference.sqlite3") as store:
    store.add_security(Security("verified:meta-class-a", issuer_id="verified:meta"))
    store.add_symbol("verified:meta-class-a", "FB", "2012-05-18", "2022-06-09",
                     known_at="2012-05-17")
    store.add_symbol("verified:meta-class-a", "META", "2022-06-09",
                     known_at="2022-05-31")
    store.add_membership("NASDAQ100", "verified:meta-class-a", "2012-12-12",
                         known_at="2012-12-05")
```

IDs are caller-supplied verified identifiers, independent of ticker spellings.
`issuer_id` can connect distinct share classes without merging them. `status`
records labels such as delisted, bankrupt, or acquired; it never invents a
liquidation price. Corporate-action accounting is separate from reference data.
`resolve_symbol(symbol, as_of)` protects against ticker reuse;
`symbol_history(security_id, start, end)` provides historical provider query
symbols. Provider-specific mappings use `provider=`. Conflicting intervals fail.

`known_at` is the time the reference fact became available. Resolution rejects
facts learned after the decision timestamp, including incomplete symbol history.
Membership removals may supply a separate `end_known_at`; if a removal took
effect before it was known, strict resolution fails rather than silently removing
the constituent. When omitted, `known_at` attests both endpoints. A future
announced removal does not affect membership before its effective date.
`set_coverage` records the exact supported interval of a complete source.

## Importing retrospective public sources

Public candidate lists lack verified publication timestamps and permanent IDs.
Imports therefore **always** mark them as retrospective reconstructions. Strict
resolution rejects them even when their recorded membership dates appear correct.
An explicit `allow_reconstructed=True` permits research, attaches warnings, and
does not claim publication-time correctness.

```python
from convexity.data import UniverseStore, fetch_candidate

with UniverseStore("reference.sqlite3") as store:
    report = fetch_candidate(store, "fja05680/sp500", "SP500",
                             allow_provisional=True)
    snapshot = store.resolve("SP500", "2015-06-01", allow_reconstructed=True)
    print(report, snapshot.warnings)
```

For Nasdaq, prefer the licensed `jmccarrell/n100tickers` candidate after reviewing
its limitations. `thuningxu/sp500nq100` has no declared license; automatic import
requires `allow_unlicensed=True`, an explicit acknowledgment that the caller has
appropriate usage rights. Convexity does not redistribute these datasets.

`fetch_candidate` pins the download to a Git commit and retains original source
bytes, SHA-256, repository URL, retrieval time, commit publication time, README,
license text, coverage assumptions, and unresolved discrepancies in SQLite.
Annual Nasdaq YAML is parsed as a restricted data format, never executed. Annual
baselines and changes are validated for continuity. Imports are transactional.
Reimporting identical bytes is idempotent. A competing source or changed upstream
history is archived and rejected pending explicit reconciliation; automatic
reference-history updates are intentionally unsupported.

Default coverage stops after the last observed snapshot for CSV sources, or at
the documented coverage bound for the annual Nasdaq source. `coverage_end` is
exclusive. Extending it is an explicit caller assumption, retained as a warning;
it must not be mistaken for verified current membership.

Provisional IDs are unique per source, universe, ticker, and contiguous membership
episode. They do **not** assert that two occurrences of a ticker identify the same
security. Renames and cross-index duplicates remain unresolved. Accordingly,
provisional unions may overcount and are unsuitable for claiming stable-identity
portfolio results. Provide verified mappings for research that needs identity
continuity:

```python
from convexity.data import import_snapshot_csv

# CSV format: date,tickers with comma-separated symbols quoted in the second cell.
import_snapshot_csv(
    store, "authorized-history.csv", name="NASDAQ100", source="licensed vendor",
    known_at="2026-10-13", coverage_end="2026-10-14",
    identity_map={"FB": "verified:meta-class-a", "META": "verified:meta-class-a"},
)
```

A symbol appearing in multiple episodes requires keys such as
`XYZ@2015-01-01`; a simple `XYZ` key is deliberately insufficient. Verified
mappings to one ID cannot overlap for different share classes. Imported symbol
history only covers observed membership episodes: pre-entry momentum lookback
requires independently verified symbol mappings. `historical_members` returns
all historical members with membership-clipped ticker intervals; it never fills
unknown identity intervals or substitutes another issuer's prices.

The public sources disagree on some historical members and lack publication-time
security identities. Source/license details and unresolved discrepancies stay in
the import report; review that report before choosing a research population.
Agreement between related reconstructions is not independent verification.
