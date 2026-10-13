"""Read option chains or quote one exact option; submits no orders."""
import argparse
import asyncio
from dataclasses import asdict
from datetime import date
from decimal import Decimal
import json
import logging
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from convexity.brokers.ibkr import PaperAccountConfig, PaperAccountError
from convexity.brokers.ibkr_options import OptionQuoteRequest, discover_options, read_option_quote


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", required=True)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("chain")
    quote = sub.add_parser("quote")
    quote.add_argument("--expiry", type=date.fromisoformat, required=True)
    quote.add_argument("--strike", type=Decimal, required=True)
    quote.add_argument("--right", choices=("C", "P"), required=True)
    quote.add_argument("--trading-class", required=True)
    args = parser.parse_args()
    for name in ("ib_async", "ib_async.ib", "ib_async.client", "ib_async.wrapper"):
        logging.getLogger(name).disabled = True
    try:
        config = PaperAccountConfig.from_env()
        if args.action == "chain":
            result = [asdict(c) for c in asyncio.run(discover_options(config, args.symbol))]
        else:
            result = asdict(asyncio.run(read_option_quote(config, OptionQuoteRequest(
                args.symbol, args.expiry, args.strike, args.right, args.trading_class))))
        print(json.dumps(result, indent=2, default=str))
        return 0 if args.action == "chain" or result["status"] == "received" else 2
    except (OSError, ValueError, PaperAccountError) as exc:
        print(f"Option read failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
