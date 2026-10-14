# Daily acquisition and coverage

Install the optional calendar, data and IBKR dependencies in your environment:

```bash
python -m pip install -e '.[data,calendars,ibkr]'
python -m convexity data prefetch --help
```

`convexity data ...` and `python -m convexity data ...` use the same CLI. No
network downloads happen at import time.

## Import reference data first

The downloader requires populated, audited `UniverseStore` membership and symbol
intervals; it never substitutes today's constituents. See
[point-in-time universes](universes.md) for source review and identity limitations.
For example, the licensed candidate imports are:

```bash
python -m convexity data import-universe \
  --source fja05680/sp500 --name SP500 --allow-provisional
python -m convexity data import-universe \
  --source jmccarrell/n100tickers --name NASDAQ100 --allow-provisional
```

`--allow-provisional` explicitly accepts source-scoped, unverified security IDs.
Use `--identity-map curated-identities.json` instead for a verified mapping;
symbols with multiple membership episodes require `SYMBOL@episode-start-date`
keys. Different sources' provisional identities must not be treated as verified
cross-index identities. `--allow-unlicensed` is separately required for sources
without a verified license. No option removes source provenance or disagreements.

Imports retain original bytes, source commit and retrieval time. Their default
coverage ends at the source's verified bound. `--coverage-end` is an exclusive
bound asserted by the caller, not a way to infer recent membership. Requests
beyond known coverage fail instead of assuming unchanged constituents.

## Resume daily history

After checking that both imported sources cover the requested interval:

```bash
python -m convexity data prefetch \
  --universe SP500,NASDAQ100 --start 2015-01-01 --end 2026-10-13 \
  --resolution 1d --resume --allow-reconstructed
```

The example end date is not a claim that those public sources are current through
that date. Shorten the request or update the reference store if it is beyond the
verified source bounds. `--allow-reconstructed` explicitly chooses retrospective
membership research; it does not turn recently published reconstructions into
information known to a historical trader.

The historical union determines security IDs. Acquisition then spans their
verified historical symbol mappings across the requested period, including
known mappings outside index membership. Membership-only source data does not
establish a symbol's listing history outside its membership episode; supply
curated symbol intervals to acquire that additional lookback safely.

The downloader fetches daily bars through `AlpacaDailyHistoricalData` and uses
IBKR only after a primary chunk fails. It requests Alpaca `1Day`, follows
pagination, uses split adjustment and `asof=-` to preserve the requested historical
symbol. It never bulk-downloads intraday bars. Alpaca credentials follow the
existing configuration script/environment handling. `--no-ibkr` disables fallback.

Acquisitions are bounded by `--concurrency` and `--chunk-sessions` (default 63
exchange sessions); each provider is serialized and paced, including IBKR client
connections. Retries use exponential backoff. If a chunk cannot be acquired, the
request is bisected to recover available dates instead of dropping all preceding
history around a listing, delisting or provider gap. Individual failures can make
sparse or unavailable historical periods slow to investigate.

Each validated chunk is committed through the existing `CachedHistoricalProvider`
SQLite tables immediately. Security/session mappings and provider failures live
alongside those tables. Ctrl-C cancels pending work; completed chunks survive.
`--resume` is the default and skips completed observations, including cached fallback
observations. `--refresh` replaces requested coverage. Neither mode deletes other
instruments or resolutions.

Empty responses, timeouts, missing credentials and delisting status do not prove
permanent unavailability. Built-in adapters report these conservatively as
retryable. Integrations with authoritative provider evidence can raise
`PermanentHistoryError(reason, evidence=...)`; those intervals are skipped for
that provider on subsequent runs, unless `--retry-permanent` is supplied. Another
provider may still retrieve them.

## Bar semantics and reuse

Native daily aggregation has different trade-condition/volume semantics from
minute-derived RTH history. The feeds have separate cache namespaces; a native
daily bar is never silently presented as a minute-derived RTH bar. Its manifest
records `aggregation=native_daily`, split adjustment, original symbol, source,
acquisition timestamps, page hashes and conservative availability at the next
New York midnight. Session open/close timestamps identify the underlying session;
they do not override the availability metadata. Dynamic research delays native
daily observations to the following session's open to avoid same-close lookahead.
See [Alpaca's historical bars reference](https://docs.alpaca.markets/us/reference/stockbars)
and [trade-condition rules](https://docs.alpaca.markets/us/docs/market-data-faq).

Adjusted prices reflect acquisition-time adjustments, not executable historical
share prices. Providers can revise older bars after later splits. Acquisition
timestamps are preserved, but the cache does not certify a single adjustment
vintage across incremental downloads. After corporate-action revisions, use
`--refresh` across **all** history required by the study; refreshing only the
latest dates can mix incompatible historical price scales. This limitation also
applies to the existing split-adjusted intraday cache.

Use the same provider/cache in daily research:

```python
from convexity.data import daily_provider
provider = daily_provider(cache=".convexity/history.sqlite3")
```

Existing intraday requests continue to use `CachedHistoricalProvider` with the
existing RTH adapters. It validates cached session panels, fetches only missing
session runs, and reuses them on later backtests. Intraday requests remain limited
to instruments and dates actually requested by the strategy.

## Coverage and failures

Progress is emitted as JSON lines on stderr; the final report defaults to
`.convexity/prefetch-coverage.json` (`--report` overrides it). Reports retain:

- Eligible/observable counts, ratios and missing stable IDs by session and index,
  with a deduplicated union.
- Provider names, failed symbol/date intervals, reasons, attempted times and any
  evidence supporting permanent unavailability.
- Separate candidate and selected populations. These are `null` during ingestion;
  bar availability alone does not establish a six-month momentum feature.
- Source and survivorship/selection-bias limitations.

`--missing-policy fail` writes the report and exits with code 2 for incomplete
coverage. `exclude` keeps the diagnostics and permits research over the observable
subset. Neither policy alters membership, fabricates prices, or substitutes a
different security. A correct historical eligible set with incomplete observable
prices is not a fully survivorship-bias-free dataset.

Python callers can use `DailyPrefetcher.run(...)`, `build_coverage(...)` and
`enforce_coverage(report, policy)` directly.
