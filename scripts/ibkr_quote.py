"""Resolve and quote one stock through Convexity's read-only paper API."""

import argparse
import asyncio
from dataclasses import asdict
import json
import logging
from pathlib import Path
import sys
from typing import cast, Literal

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from convexity.brokers.ibkr import (
    PaperAccountConfig,
    PaperAccountError,
    StockQuoteRequest,
    read_stock_quote,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--exchange", default="SMART")
    parser.add_argument("--currency", default="USD")
    parser.add_argument("--primary-exchange", default="")
    parser.add_argument(
        "--data-type",
        choices=("live", "frozen", "delayed", "delayed_frozen"),
        default="live",
    )
    parser.add_argument(
        "--max-age",
        type=float,
        default=30.0,
        help="Maximum local bid/ask receipt age in seconds.",
    )
    args = parser.parse_args()
    for name in ("ib_async", "ib_async.ib", "ib_async.client", "ib_async.wrapper"):
        logging.getLogger(name).disabled = True
    try:
        config = PaperAccountConfig.from_env()
        request = StockQuoteRequest(
            args.symbol,
            args.exchange,
            args.currency,
            args.primary_exchange,
            cast(
                Literal["live", "frozen", "delayed", "delayed_frozen"], args.data_type
            ),
            args.max_age,
        )
        quote = asyncio.run(read_stock_quote(config, request))
    except (OSError, ValueError, PaperAccountError) as exc:
        print(f"Stock quote failed: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Stock quote interrupted; disconnected.", file=sys.stderr)
        return 130
    print(json.dumps(asdict(quote), indent=2, default=str))
    return 0 if quote.status == "received" else 2


if __name__ == "__main__":
    raise SystemExit(main())
