"""Small command line interface for explicit universe import and daily ingestion."""

from __future__ import annotations

import argparse
import asyncio
from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path
import sys


def parser():
    root = argparse.ArgumentParser(prog="convexity")
    commands = root.add_subparsers(dest="command", required=True)
    data = commands.add_parser("data", help="Historical universe and price acquisition")
    actions = data.add_subparsers(dest="action", required=True)
    acquire = actions.add_parser(
        "prefetch", help="Resume native daily OHLCV acquisition"
    )
    acquire.add_argument("--universe", default="SP500,NASDAQ100")
    acquire.add_argument("--start", required=True, type=date.fromisoformat)
    acquire.add_argument("--end", required=True, type=date.fromisoformat)
    acquire.add_argument("--resolution", choices=("1d", "1day", "1 day"), default="1d")
    acquire.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Fetch missing observations (default)",
    )
    acquire.add_argument(
        "--refresh",
        action="store_true",
        help="Explicitly replace requested observations",
    )
    acquire.add_argument("--retry-permanent", action="store_true")
    acquire.add_argument(
        "--cache", type=Path, default=Path(".convexity/history.sqlite3")
    )
    acquire.add_argument(
        "--store", type=Path, default=Path(".convexity/universes.sqlite3")
    )
    acquire.add_argument(
        "--report", type=Path, default=Path(".convexity/prefetch-coverage.json")
    )
    acquire.add_argument(
        "--allow-reconstructed",
        action="store_true",
        help="Explicitly accept retrospective membership research",
    )
    acquire.add_argument(
        "--no-ibkr", action="store_true", help="Disable selective IBKR fallback"
    )
    acquire.add_argument("--concurrency", type=int, default=4)
    acquire.add_argument("--chunk-sessions", type=int, default=63)
    acquire.add_argument("--retries", type=int, default=2)
    acquire.add_argument(
        "--missing-policy", choices=("exclude", "fail"), default="exclude"
    )
    acquire.add_argument("--calendar", default="XNYS")
    source = actions.add_parser(
        "import-universe", help="Preserve and import a reviewed source snapshot dataset"
    )
    source.add_argument(
        "--source",
        required=True,
        choices=("fja05680/sp500", "thuningxu/sp500nq100", "jmccarrell/n100tickers"),
    )
    source.add_argument("--name", required=True, choices=("SP500", "NASDAQ100"))
    source.add_argument(
        "--store", type=Path, default=Path(".convexity/universes.sqlite3")
    )
    source.add_argument("--coverage-end")
    source.add_argument("--allow-unlicensed", action="store_true")
    source.add_argument("--allow-provisional", action="store_true")
    source.add_argument(
        "--identity-map",
        type=Path,
        help="JSON object mapping historical symbols to curated stable security IDs",
    )
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    from convexity.data.universe import UniverseStore

    store = None
    try:
        store = UniverseStore(args.store)
        if args.action == "import-universe":
            from convexity.data.universe_sources import fetch_candidate

            result = fetch_candidate(
                store,
                args.source,
                args.name,
                allow_unlicensed=args.allow_unlicensed,
                allow_provisional=args.allow_provisional,
                coverage_end=args.coverage_end,
                identity_map=json.loads(args.identity_map.read_text())
                if args.identity_map
                else None,
            )
            print(json.dumps(result, indent=2, default=str))
            return 0
        from convexity.backtest.history import IBKRHistoricalData, _atomic
        from convexity.data.prefetch import DailyPrefetcher, PrefetchConfig

        start = datetime.combine(args.start, datetime.min.time(), timezone.utc)
        end = datetime.combine(
            args.end + timedelta(days=1), datetime.min.time(), timezone.utc
        )
        names = tuple(
            name.strip().upper() for name in args.universe.split(",") if name.strip()
        )
        members = store.historical_members(
            names, start, end, allow_reconstructed=args.allow_reconstructed
        )
        from convexity.data.universe import HistoricalMember

        acquisitions = []
        for sid in sorted({member.security_id for member in members}):
            for mapping in store.symbol_history(
                sid, start, end, allow_reconstructed=args.allow_reconstructed
            ):
                # symbol_history clips open-ended mappings to the request end.
                assert mapping.end is not None
                acquisitions.append(
                    HistoricalMember(
                        sid, mapping.symbol, mapping.start, mapping.end, ()
                    )
                )
        config = PrefetchConfig(
            chunk_sessions=args.chunk_sessions,
            concurrency=args.concurrency,
            retries=args.retries,
        )
        downloader = DailyPrefetcher(
            fallback=None if args.no_ibkr else IBKRHistoricalData(),
            path=args.cache,
            config=config,
            progress=lambda row: print(json.dumps(row), file=sys.stderr, flush=True),
        )
        report = asyncio.run(
            downloader.run(
                members,
                args.start,
                args.end,
                calendar=args.calendar,
                resume=not args.refresh,
                retry_permanent=args.retry_permanent,
                acquisition_members=acquisitions,
            )
        )
        report["membership_mode"] = (
            "retrospective_reconstruction"
            if args.allow_reconstructed
            else "publication_time_safe"
        )
        _atomic(args.report, report)
        print(
            json.dumps(
                dict(
                    complete=report["complete"],
                    report=str(args.report),
                    cache=str(args.cache),
                )
            )
        )
        return 2 if args.missing_policy == "fail" and not report["complete"] else 0
    except KeyboardInterrupt:
        print(
            "Cancelled. Completed chunks are persisted; rerun with --resume.",
            file=sys.stderr,
        )
        return 130
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"convexity: {exc}", file=sys.stderr)
        return 2
    finally:
        if store is not None:
            store.close()
