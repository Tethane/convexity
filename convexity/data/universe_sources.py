"""Auditable, opt-in imports of third-party index membership reconstructions.

No third-party dataset ships with Convexity. Downloads are pinned to a Git
commit and original bytes are retained in the local reference database.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
import re
from urllib.parse import quote
from urllib.request import Request, urlopen

import pandas as pd

from .core import timestamp
from .security import Security, UniverseError, _iso
from .universe import UniverseStore

CANDIDATE_SOURCES = {
    "fja05680/sp500": {"license": "MIT", "names": ("SP500",), "branch": "master"},
    "thuningxu/sp500nq100": {
        "license": None,
        "names": ("SP500", "NASDAQ100"),
        "branch": "main",
    },
    "jmccarrell/n100tickers": {
        "license": "MIT",
        "names": ("NASDAQ100",),
        "branch": "main",
    },
}


def _read(url):
    with urlopen(
        Request(url, headers={"User-Agent": "convexity-universe-import/1"}), timeout=60
    ) as response:
        return response.read()


def _parse_snapshots(raw: bytes):
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
    if reader.fieldnames != ["date", "tickers"]:
        raise UniverseError("Expected CSV header date,tickers")
    result = []
    for line, row in enumerate(reader, 2):
        if None in row or row["tickers"] is None:
            raise UniverseError(f"Malformed snapshot CSV row {line}")
        date = timestamp(row["date"])
        symbols = tuple(x.strip() for x in row["tickers"].split(","))
        if not symbols or any(
            not s or not re.fullmatch(r"[A-Za-z0-9.^_/-]+", s) for s in symbols
        ):
            raise UniverseError(f"Invalid ticker on row {line}")
        if len(set(symbols)) != len(symbols):
            raise UniverseError(f"Duplicate ticker on row {line}")
        if result and date <= result[-1][0]:
            raise UniverseError("Snapshot dates must be strictly increasing")
        result.append((date, set(symbols)))
    if not result:
        raise UniverseError("Snapshot source is empty")
    return result


def import_snapshot_csv(
    store: UniverseStore,
    path: str | Path | bytes,
    *,
    name: str,
    source: str,
    known_at,
    coverage_end,
    identity_map: dict | None = None,
    allow_provisional: bool = False,
    metadata: dict | None = None,
    original_raw: bytes | None = None,
    discrepancies=(),
) -> dict:
    """Import date,tickers snapshots as a retrospective reconstruction.

    ``identity_map`` maps ``SYMBOL@episode-start-date`` to a verified security ID.
    A simple SYMBOL key is accepted only for a symbol occurring in one episode;
    repeats require episode keys, preventing silent ticker-reuse merges. Multiple
    symbols may map to one ID only if their effective intervals do not overlap.
    ``coverage_end`` is exclusive and must be verified by the caller. Imported
    knowledge is never backdated to the effective date.
    """
    raw = path if isinstance(path, bytes) else Path(path).read_bytes()
    snapshots = _parse_snapshots(raw)
    name, cutoff = name.upper(), timestamp(coverage_end)
    if cutoff <= snapshots[-1][0]:
        raise UniverseError("coverage_end must be after the final effective snapshot")
    knowledge = timestamp(known_at)
    original = raw if original_raw is None else original_raw
    digest = hashlib.sha256(original).hexdigest()
    source_id = hashlib.sha256(f"{name}:{source}:{digest}".encode()).hexdigest()
    if store.connection.execute(
        "SELECT 1 FROM universe_sources WHERE source_id=?", (source_id,)
    ).fetchone():
        return {"name": name, "source_id": source_id, "already_imported": True}
    # A conflicting update is archived for reconciliation rather than silently
    # overwriting history, publication times, or identity assignments.
    provenance = dict(
        metadata or {},
        coverage_end=_iso(cutoff),
        known_at=_iso(knowledge),
        reconstructed=True,
        identity_policy="verified map"
        if not allow_provisional
        else "explicit provisional episodes",
    )
    source_record = (
        source_id,
        name,
        source,
        _iso(pd.Timestamp.now(tz="UTC")),
        digest,
        original,
        json.dumps(provenance, sort_keys=True),
    )
    if store.connection.execute(
        "SELECT 1 FROM universe_coverage WHERE name=?", (name,)
    ).fetchone():
        store.connection.execute(
            "INSERT INTO universe_sources VALUES (?,?,?,?,?,?,?)", source_record
        )
        store.connection.execute(
            "INSERT INTO universe_discrepancies VALUES (?,?,?)",
            (
                name,
                source_id,
                "Additional source archived but not applied: reconcile competing history explicitly.",
            ),
        )
        store.connection.commit()
        raise UniverseError(
            f"{name} already has a source; new raw data archived for explicit reconciliation"
        )
    active, episodes = {}, []
    for date, symbols in [*snapshots, (cutoff, set())]:
        for symbol in sorted(active.keys() - symbols):
            episodes.append((symbol, active.pop(symbol), date))
        for symbol in sorted(symbols - active.keys()):
            active[symbol] = date
    counts = {}
    for symbol, _, _ in episodes:
        counts[symbol] = counts.get(symbol, 0) + 1
    mapping, assignments, provisional = identity_map or {}, [], False
    for symbol, a, b in episodes:
        key = f"{symbol}@{a.date().isoformat()}"
        sid = mapping.get(key)
        if sid is None and counts[symbol] == 1:
            sid = mapping.get(symbol)
        if sid is None:
            if not allow_provisional:
                raise UniverseError(
                    f"Verified security identity missing for {key}; supply identity_map or explicitly allow_provisional"
                )
            sid = f"provisional:{source}:{name}:{key}"
            provisional = True
        assignments.append((str(sid), symbol, a, b))
    # Group adjacent membership episodes of a verified identity, e.g. a rename.
    grouped = {}
    for sid, symbol, a, b in assignments:
        grouped.setdefault(sid, []).append((a, b))
    notes = list(discrepancies)
    notes.append(
        "Historical publication timestamps are unavailable; this source is a retrospective reconstruction."
    )
    if provisional:
        notes.append(
            "Provisional security identities: ticker reuse, renames and cross-index identity are unresolved; union counts may overcount. Import verified identity mappings before portfolio research."
        )
    store._defer_commit = True
    try:
        store.connection.execute("BEGIN")
        for sid, symbol, a, b in assignments:
            if not store.connection.execute(
                "SELECT 1 FROM securities WHERE security_id=?", (sid,)
            ).fetchone():
                store.add_security(Security(sid))
            # Verified mappings may already exist across another index import.
            existing = store.connection.execute(
                """SELECT * FROM symbols WHERE security_id=? AND symbol=? AND provider=''
                AND start<=? AND (end IS NULL OR end>=?)""",
                (sid, symbol, _iso(a), _iso(b)),
            ).fetchone()
            if not existing and not sid.startswith("provisional:"):
                overlaps = store.connection.execute(
                    """SELECT * FROM symbols WHERE security_id=? AND symbol=? AND provider=''
                    AND start<=? AND (end IS NULL OR end>=?)""",
                    (sid, symbol, _iso(b), _iso(a)),
                ).fetchall()
                if overlaps:
                    merged_a = min([_iso(a), *[r["start"] for r in overlaps]])
                    merged_b = (
                        None
                        if any(r["end"] is None for r in overlaps)
                        else max([_iso(b), *[r["end"] for r in overlaps]])
                    )
                    merged_known = max(
                        [_iso(knowledge), *[r["known_at"] for r in overlaps]]
                    )
                    for row in overlaps:
                        store.connection.execute(
                            "DELETE FROM symbols WHERE security_id=? AND symbol=? AND start=? AND provider=''",
                            (sid, symbol, row["start"]),
                        )
                    store.add_symbol(
                        sid, symbol, merged_a, merged_b, known_at=merged_known
                    )
                    existing = True
            if not existing:
                store.add_symbol(
                    sid,
                    symbol,
                    a,
                    b,
                    known_at=knowledge,
                    allow_ambiguous=sid.startswith("provisional:"),
                )
        for sid, intervals in grouped.items():
            merged = []
            for a, b in sorted(intervals):
                if merged and a < merged[-1][1]:
                    raise UniverseError(
                        f"Overlapping share classes mapped to one security: {sid}"
                    )
                if merged and a == merged[-1][1]:
                    merged[-1] = (merged[-1][0], b)
                else:
                    merged.append((a, b))
            for a, b in merged:
                store.add_membership(
                    name, sid, a, b, known_at=knowledge, source=source_id
                )
        store.set_coverage(
            name, snapshots[0][0], cutoff, known_at=knowledge, reconstructed=True
        )
        store.connection.execute(
            "INSERT INTO universe_sources VALUES (?,?,?,?,?,?,?)", source_record
        )
        store.connection.executemany(
            "INSERT OR IGNORE INTO universe_discrepancies VALUES (?,?,?)",
            [(name, source_id, n) for n in notes],
        )
        store.connection.commit()
    except BaseException:
        store.connection.rollback()
        raise
    finally:
        store._defer_commit = False
    return {
        "name": name,
        "source_id": source_id,
        "snapshots": len(snapshots),
        "securities": len(grouped),
        "coverage_start": str(snapshots[0][0]),
        "coverage_end": str(cutoff),
        "warnings": notes,
    }


def _parse_n100_yaml(raw: bytes):
    """Parse only the upstream's audited scalar/list YAML subset; never execute code."""
    year, baseline, changes, date, operation = None, [], {}, None, None
    for original in raw.decode("utf-8").splitlines():
        line = original.split("#", 1)[0].rstrip()
        if not line.strip() or line == "---":
            continue
        if re.fullmatch(r"year: \d{4}", line):
            year = int(line.split(":")[1])
        elif line == "tickers_on_Jan_1:":
            operation = "baseline"
        elif line == "changes:":
            operation = None
        elif re.fullmatch(r"  ['\"]?\d{4}-\d{2}-\d{2}['\"]?:", line):
            date = line.strip().rstrip(":").strip("'\"")
            if date in changes:
                raise UniverseError(f"Duplicate YAML change date {date}")
            changes[date] = {}
            operation = None
        elif line in ("    union:", "    difference:") and date:
            operation = line.strip().rstrip(":")
            changes[date][operation] = []
        elif re.fullmatch(r"\s+- [A-Za-z0-9.^_/-]+", line) and operation:
            symbol = line.strip()[2:]
            target = baseline if operation == "baseline" else changes[date][operation]
            if symbol in target:
                raise UniverseError("Duplicate YAML ticker")
            target.append(symbol)
        else:
            raise UniverseError(f"Unsupported upstream YAML syntax: {original!r}")
    if year is None or not baseline:
        raise UniverseError("Missing annual baseline")
    state, snapshots = set(baseline), [(timestamp(f"{year}-01-01"), set(baseline))]
    for date, ops in sorted(changes.items()):
        additions, removals = set(ops.get("union", [])), set(ops.get("difference", []))
        if not removals <= state or additions & (state - removals):
            raise UniverseError(f"Inconsistent membership change at {date}")
        state = (state - removals) | additions
        snapshots.append((timestamp(date), state.copy()))
    return snapshots


def fetch_candidate(
    store: UniverseStore,
    source: str,
    name: str,
    *,
    allow_unlicensed=False,
    allow_provisional=False,
    coverage_end=None,
    identity_map=None,
) -> dict:
    """Download a pinned candidate to local storage, retaining license/provenance.

    No unpublished announcement timestamps or permanent identifiers are inferred.
    A source lacking a license requires the caller to confirm their own rights.
    """
    name = name.upper()
    if (
        source not in CANDIDATE_SOURCES
        or name not in CANDIDATE_SOURCES[source]["names"]
    ):
        raise UniverseError(f"Unsupported source/index pair: {source}/{name}")
    spec = CANDIDATE_SOURCES[source]
    if spec["license"] is None and not allow_unlicensed:
        raise UniverseError(
            "Source has no published license; supply authorized local data or explicitly acknowledge your usage rights with allow_unlicensed=True"
        )
    commit = json.loads(
        _read(f"https://api.github.com/repos/{source}/commits/{spec['branch']}")
    )
    sha, known = commit["sha"], commit["commit"]["committer"]["date"]
    base = f"https://raw.githubusercontent.com/{source}/{sha}/"
    readme = _read(base + "README.md").decode()
    license_text = _read(base + "LICENSE").decode() if spec["license"] else None
    notes = []
    if name == "NASDAQ100":
        notes.append(
            "Cross-source audit 2026-10-13: Nasdaq 2015-06-01 differs (jmccarrell 105 vs thuningxu 106): jmccarrell-only QVCA/SYMC/WFM; thuningxu-only NDOI/NLOK/QRTEA/WFMI. Unresolved; do not automatically merge conflicting sources."
        )
    if source == "jmccarrell/n100tickers":
        tree = json.loads(
            _read(f"https://api.github.com/repos/{source}/git/trees/{sha}?recursive=1")
        )
        paths = sorted(
            x["path"]
            for x in tree["tree"]
            if re.fullmatch(
                r"src/nasdaq_100_ticker_history/n100-ticker-changes-\d{4}.yaml",
                x["path"],
            )
        )
        if not paths:
            raise UniverseError("No annual Nasdaq history found")
        originals, snapshots, previous = {}, [], None
        for path in paths:
            data = _read(base + quote(path))
            originals[path] = data.decode()
            annual = _parse_n100_yaml(data)
            if previous is not None and previous != annual[0][1]:
                raise UniverseError(f"Annual baseline discontinuity in {path}")
            snapshots.extend(annual)
            previous = annual[-1][1]
        snapshots = [(d, s) for d, s in snapshots if d >= timestamp("2007-02-01")]
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["date", "tickers"])
        for d, s in snapshots:
            writer.writerow([d.date().isoformat(), ",".join(sorted(s))])
        raw = output.getvalue().encode()
        original = json.dumps(originals, sort_keys=True).encode()
        match = re.search(
            r"coverage is provided from .*?through at least ([A-Za-z]+ \d{1,2}, \d{4})",
            readme,
            re.I | re.S,
        )
        verified_end = (
            timestamp(match.group(1)) + pd.Timedelta(days=1)
            if match
            else snapshots[-1][0] + pd.Timedelta(days=1)
        )
        default_end = verified_end
        if coverage_end is not None and timestamp(coverage_end) > verified_end:
            notes.append(
                "Caller extends coverage beyond upstream documented coverage; latest composition is assumed unchanged."
            )
    else:
        path = (
            "S&P 500 Historical Components & Changes (Updated).csv"
            if source == "fja05680/sp500"
            else (
                "sp500_components_history.csv"
                if name == "SP500"
                else "nasdaq100_components_history.csv"
            )
        )
        raw = _read(base + quote(path))
        original = raw
        default_end = _parse_snapshots(raw)[-1][0] + pd.Timedelta(days=1)
        if source == "fja05680/sp500":
            notes.append(
                "Original pre-2001 S&P history has documented missing constituents; ticker-only source does not establish permanent security identity."
            )
        elif name == "NASDAQ100":
            notes.append(
                "Upstream reports missing ENDP and CMCSK intervals before 2017, and some historical ticker inaccuracies."
            )
        if coverage_end is not None and timestamp(coverage_end) > default_end:
            notes.append(
                "Caller extends coverage beyond the final recorded snapshot; latest composition is assumed unchanged."
            )
    return import_snapshot_csv(
        store,
        raw,
        name=name,
        source=source,
        known_at=known,
        coverage_end=coverage_end or default_end,
        identity_map=identity_map,
        allow_provisional=allow_provisional,
        original_raw=original,
        discrepancies=notes,
        metadata={
            "repository": f"https://github.com/{source}",
            "commit": sha,
            "license": spec["license"],
            "license_text": license_text,
            "readme": readme,
        },
    )
