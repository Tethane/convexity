"""Actually open and close one configured paper debit vertical, with bounded waits."""

import argparse
import asyncio
from dataclasses import asdict, replace
from datetime import date
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
import json
import logging
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from convexity.brokers.ibkr import PaperAccountConfig
from convexity.brokers.ibkr_options import OptionQuoteRequest, read_option_quote
from convexity.brokers.ibkr_verticals import VerticalBroker
from convexity.trading import OptionIdentity, OrderJournal, VerticalOrder


def emit(record):
    data = asdict(record)
    data["account"] = "[PAPER_ACCOUNT]"
    print(json.dumps(data, default=str), flush=True)


async def run(args) -> int:
    config = PaperAccountConfig.from_env()
    identities = []
    for strike in (args.long_strike, args.short_strike):
        quote = await read_option_quote(
            config,
            OptionQuoteRequest(
                args.symbol, args.expiry, strike, args.right, args.trading_class
            ),
        )
        if quote.status != "received":
            raise RuntimeError(
                f"Option quote {strike} is {quote.status}; no new entry."
            )
        c = quote.contract
        identities.append(
            OptionIdentity(
                c.contract_id,
                c.symbol,
                args.expiry,
                c.strike,
                args.right,
                c.trading_class,
            )
        )
        print(json.dumps(asdict(quote), default=str), flush=True)
    template = VerticalOrder(
        args.id + "-entry",
        args.strategy,
        identities[0],
        identities[1],
        "BUY",
        1,
        Decimal(".01"),
    )
    with OrderJournal(args.journal) as journal:
        async with VerticalBroker(
            config,
            journal,
            strategy_budgets={args.strategy: args.budget},
            on_update=emit,
        ) as broker:
            try:
                entry = journal.get(template.client_order_id)
                if not isinstance(entry.request, VerticalOrder) or (
                    entry.request.long_leg,
                    entry.request.short_leg,
                ) != tuple(identities):
                    raise RuntimeError("Run ID belongs to different contracts.")
            except KeyError:
                a, b = await broker.vertical_quotes(template)
                if a.status != "received" or b.status != "received":
                    raise RuntimeError("Entry quotes unavailable.")
                assert a.ask is not None and b.bid is not None
                price = min(
                    template.width - Decimal(".01"), a.ask - b.bid + Decimal(".05")
                )
                entry = await broker.submit(
                    replace(
                        template,
                        limit_price=price.quantize(
                            Decimal(".01"), rounding=ROUND_CEILING
                        ),
                    )
                )
            entry = await broker.observe(entry.request.client_order_id, seconds=20)
            if not entry.terminal:
                await broker.cancel(entry.request.client_order_id)
                entry = await broker.observe(entry.request.client_order_id, seconds=15)
            await asyncio.sleep(1)
            entry = journal.get(entry.request.client_order_id)
            emit(entry)
            if (
                not entry.terminal
                or entry.uncertain
                or entry.executed_quantity != entry.filled_quantity
            ):
                raise RuntimeError("Entry unresolved; reconcile before proceeding.")
            if not entry.executed_quantity:
                print("Entry did not fill; no test position opened.", flush=True)
                return 2
            await asyncio.sleep(args.hold_seconds)
            for attempt in range(2):
                owned = [
                    journal.attributed_position(
                        config.account, args.strategy, c.contract_id
                    )
                    for c in identities
                ]
                if owned == [0, 0]:
                    break
                if owned != [1, -1]:
                    raise RuntimeError(
                        f"Unexpected leg holdings {owned}; reconcile, do not auto-leg."
                    )
                close_id = f"{args.id}-exit{attempt + 1}"
                try:
                    exit_order = journal.get(close_id)
                except KeyError:
                    a, b = await broker.vertical_quotes(template)
                    if a.status != "received" or b.status != "received":
                        raise RuntimeError(
                            "Exit quotes unavailable; position remains open."
                        )
                    assert a.bid is not None and b.ask is not None
                    price = max(
                        Decimal(".01"),
                        min(
                            template.width,
                            a.bid - b.ask - Decimal(".05") * (attempt + 1),
                        ),
                    )
                    exit_order = await broker.submit(
                        replace(
                            template,
                            client_order_id=close_id,
                            side="SELL",
                            limit_price=price.quantize(
                                Decimal(".01"), rounding=ROUND_FLOOR
                            ),
                        )
                    )
                exit_order = await broker.observe(close_id, seconds=20)
                if not exit_order.terminal:
                    await broker.cancel(close_id)
                    exit_order = await broker.observe(close_id, seconds=15)
                await asyncio.sleep(1)
                emit(journal.get(close_id))
                if not exit_order.terminal or exit_order.uncertain:
                    raise RuntimeError("Exit outcome unresolved; no replacement sent.")
            positions = await broker.positions()
            remaining = {
                c.contract_id: journal.attributed_position(
                    config.account, args.strategy, c.contract_id
                )
                for c in identities
            }
            broker_positions = {
                p.contract.contract_id: p.quantity
                for p in positions
                if p.contract.contract_id in remaining
            }
            print(
                json.dumps(
                    dict(owned_positions=remaining, broker_positions=broker_positions),
                    default=str,
                ),
                flush=True,
            )
            if any(remaining.values()) or any(broker_positions.values()):
                raise RuntimeError(
                    "Exit retries exhausted; position remains open and acceptance failed."
                )
            return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journal", type=Path, default=Path("out/orders.sqlite"))
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--expiry", type=date.fromisoformat, required=True)
    parser.add_argument("--long-strike", type=Decimal, required=True)
    parser.add_argument("--short-strike", type=Decimal, required=True)
    parser.add_argument("--right", choices=("C", "P"), default="C")
    parser.add_argument("--trading-class", required=True)
    parser.add_argument("--id", required=True)
    parser.add_argument("--strategy", default="spread-smoke")
    parser.add_argument("--budget", type=Decimal, default=Decimal(100))
    parser.add_argument("--hold-seconds", type=float, default=1)
    args = parser.parse_args()
    if not 0 <= args.hold_seconds <= 60:
        parser.error("hold-seconds must be between 0 and 60")
    for name in ("ib_async", "ib_async.ib", "ib_async.client", "ib_async.wrapper"):
        logging.getLogger(name).disabled = True
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
