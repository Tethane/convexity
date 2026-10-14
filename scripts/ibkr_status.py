"""Show verified paper account, orders, executions, ownership and faults."""

import argparse
import asyncio
import logging
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from convexity.brokers.ibkr import PaperAccountConfig, PaperAccountError
from convexity.brokers.status import read_paper_status
from convexity.trading import OrderJournal


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journal", type=Path, default=Path("out/orders.sqlite"))
    args = parser.parse_args()
    for name in ("ib_async", "ib_async.ib", "ib_async.client", "ib_async.wrapper"):
        logging.getLogger(name).disabled = True
    try:
        if not args.journal.exists():
            raise ValueError(
                "Journal does not exist; pass the actual trading journal path."
            )
        with OrderJournal(args.journal) as journal:
            status = asyncio.run(
                read_paper_status(PaperAccountConfig.from_env(), journal)
            )
        print(status.format())
        return 2 if status.faults else 0
    except (OSError, ValueError, PaperAccountError) as exc:
        print(f"Status failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
