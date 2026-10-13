"""Explicitly close a journal-owned paper vertical at a minimum credit, without market data."""
import argparse
import asyncio
from dataclasses import replace
from decimal import Decimal
import logging
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from convexity.brokers.ibkr import PaperAccountConfig
from convexity.brokers.ibkr_verticals import VerticalBroker
from convexity.trading import OrderJournal, VerticalOrder
from convexity.backtest.runner import serialize


def output(value):
    def redact(item):
        if isinstance(item, dict):
            return {key: "[PAPER_ACCOUNT]" if key == "account" else redact(value)
                    for key, value in item.items()}
        if isinstance(item, list):
            return [redact(value) for value in item]
        return item
    print(json.dumps(redact(serialize(value)), default=str), flush=True)


async def run(args):
    config = PaperAccountConfig.from_env()
    with OrderJournal(args.journal) as journal:
        entry = journal.get(args.entry_id)
        if entry.account != config.account or not isinstance(entry.request, VerticalOrder) or entry.request.side != "BUY":
            raise ValueError("Select an existing BUY vertical from the configured paper account.")
        request = replace(entry.request, client_order_id=args.id, side="SELL", quantity=args.quantity,
                          limit_price=args.minimum_credit)
        async with VerticalBroker(config, journal,
                strategy_budgets={request.strategy_id: request.width * 100 * request.quantity},
                max_contracts=args.quantity, on_update=output) as broker:
            await broker.close_at_limit(request)
            record = await broker.observe(args.id, seconds=30)
            if not record.terminal:
                await broker.cancel(args.id)
                await broker.observe(args.id, seconds=15)
            await asyncio.sleep(1)
            positions = await broker.positions()
            record = journal.get(args.id)
            owned = {leg.contract_id: journal.attributed_position(config.account, request.strategy_id, leg.contract_id)
                     for leg in (request.long_leg, request.short_leg)}
            output(dict(order=record, owned_positions=owned,
                        broker_positions=[p for p in positions if p.contract.contract_id in owned]))
            actual = {p.contract.contract_id: p.quantity for p in positions}
            owners = {r.request.strategy_id for r in journal.orders() if r.account == config.account}
            reconciled = all(actual.get(contract_id, Decimal(0)) == sum(
                (journal.attributed_position(config.account, owner, contract_id) for owner in owners), Decimal(0))
                for contract_id in owned)
            if (not record.terminal or record.uncertain or any(owned.values())
                    or record.filled_quantity > record.executed_quantity or not reconciled):
                raise RuntimeError("Close incomplete: inspect journal and broker positions; no automatic replacement.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journal", type=Path, default=Path("out/orders.sqlite"))
    parser.add_argument("--entry-id", required=True)
    parser.add_argument("--id", required=True)
    parser.add_argument("--minimum-credit", type=Decimal, required=True)
    parser.add_argument("--quantity", type=int, default=1)
    args = parser.parse_args()
    for name in ("ib_async", "ib_async.ib", "ib_async.client", "ib_async.wrapper"):
        logging.getLogger(name).disabled = True
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
