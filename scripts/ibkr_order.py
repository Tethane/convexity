"""Explicit paper stock submit/observe/cancel commands using Convexity APIs."""

import argparse
import asyncio
from dataclasses import asdict
from decimal import Decimal, InvalidOperation
import json
import logging
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from convexity.brokers.ibkr import PaperAccountConfig, PaperAccountError
from convexity.brokers.ibkr_trading import PaperBroker
from convexity.trading import OrderJournal, StockOrder


def emit(record):
    data = asdict(record)
    data["account"] = "[PAPER_ACCOUNT]"
    data["remaining_quantity"] = str(record.remaining_quantity)
    print(json.dumps(data, default=str), flush=True)


async def run(args):
    config = PaperAccountConfig.from_env()
    with OrderJournal(args.journal) as journal:
        async with PaperBroker(config, journal, on_update=emit) as broker:
            if args.action == "submit":
                request = StockOrder(args.id, args.strategy, args.symbol, args.side, args.quantity,
                                     Decimal(args.limit), args.session)
                await broker.submit(request)
            elif args.action == "cancel":
                await broker.cancel(args.id)
            row = await broker.observe(args.id, seconds=args.wait)
            if args.action == "submit" and args.cancel_after_wait and not row.terminal:
                await broker.cancel(args.id)
                row = await broker.observe(args.id, seconds=args.wait)
            positions = await broker.positions()
            row = journal.get(args.id)  # Include status/fees received during the position read.
            emit(row)
            matching = [asdict(p) for p in positions if p.contract.contract_id == row.contract_id]
            print(json.dumps(dict(broker_positions=matching,
                                  strategy_executed_position=str(journal.attributed_position(config.account, row.request.strategy_id, row.contract_id))), default=str))
            completed_cancel = args.action == "cancel" or (args.action == "submit" and args.cancel_after_wait)
            ok = row.acknowledged and not row.uncertain and row.status != "Rejected"
            return 0 if ok and (not completed_cancel or row.terminal) else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journal", type=Path, default=Path("out/orders.sqlite"))
    parser.add_argument("--wait", type=float, default=15)
    subs = parser.add_subparsers(dest="action", required=True)
    submit = subs.add_parser("submit")
    submit.add_argument("--id", required=True, help="Stable ID; reusing it never resubmits.")
    submit.add_argument("--strategy", required=True)
    submit.add_argument("--symbol", required=True)
    submit.add_argument("--side", choices=("BUY", "SELL"), required=True)
    submit.add_argument("--quantity", type=int, required=True)
    submit.add_argument("--limit", required=True)
    submit.add_argument("--session", choices=("regular", "extended", "overnight"), required=True)
    submit.add_argument("--cancel-after-wait", action="store_true")
    for action in ("observe", "cancel"):
        subs.add_parser(action).add_argument("--id", required=True)
    args = parser.parse_args()
    if not 0 < args.wait <= 300:
        parser.error("--wait must be in (0, 300].")
    for name in ("ib_async", "ib_async.ib", "ib_async.client", "ib_async.wrapper"):
        logging.getLogger(name).disabled = True
    try:
        return asyncio.run(run(args))
    except (PaperAccountError, OSError, ValueError, KeyError, TimeoutError, InvalidOperation) as exc:
        print(f"Paper order command failed: {exc}. Inspect the journal before retrying.", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted; disconnect does not cancel orders. Inspect journal and explicitly cancel owned orders.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
