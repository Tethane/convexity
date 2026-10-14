"""Run stock, spread and SPY/QQQ 0-DTE strategies in one actual paper runtime.

Explicitly submits orders. The short smoke session uses labeled one-shot entry
and exit timers. --session instead holds until 30 minutes before broker close.
Keep the same output directory to restart the saved run without duplicate entries.
"""

import argparse
import asyncio
from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
import json
import logging
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from convexity.backtest import HistoricalData, SessionCalendar
from convexity.backtest.runner import serialize
from convexity.brokers.ibkr import PaperAccountConfig, StockQuoteRequest
from convexity.brokers.ibkr_data import read_stock_history, read_stock_sessions
from convexity.brokers.ibkr_options import (
    OptionQuoteRequest,
    discover_options,
    read_option_quote,
)
from convexity.brokers.ibkr_verticals import VerticalBroker
from convexity.runtime import (
    Bar,
    DurableSchedule,
    Instrument,
    OptionQuotes,
    PaperRuntime,
    Timer,
)
from convexity.strategies import (
    MonthlyLongShort,
    ScheduledVertical,
    StockRule,
    VerticalRule,
    ZeroDTE,
)
from convexity.trading import OptionIdentity, OrderJournal, StockOrder, VerticalOrder


def output(value):
    def redact(item):
        if isinstance(item, dict):
            return {
                k: "[PAPER_ACCOUNT]" if k == "account" else redact(v)
                for k, v in item.items()
            }
        if isinstance(item, list):
            return [redact(v) for v in item]
        return item

    print(json.dumps(redact(serialize(value)), default=str), flush=True)


async def resolve_pair(config, symbol, expiry, spot):
    chains = await discover_options(config, symbol)
    candidates = [
        c
        for c in chains
        if c.exchange == "SMART"
        and c.multiplier == "100"
        and c.trading_class == symbol
        and expiry in c.expirations
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"Missing/ambiguous standard SMART chain for {symbol} {expiry}."
        )
    strikes = candidates[0].strikes
    long_strike = min(strikes, key=lambda s: (abs(s - spot), s))
    short_strike = long_strike + 1
    if short_strike not in strikes:
        raise RuntimeError(
            "Configured one-dollar vertical strikes are unavailable; no fallback."
        )
    identities = []
    for strike in (long_strike, short_strike):
        q = await read_option_quote(
            config, OptionQuoteRequest(symbol, expiry, strike, "C", symbol)
        )
        if q.status != "received":
            raise RuntimeError(
                f"{symbol} {strike} quote is {q.status}; skip session entry."
            )
        c = q.contract
        identities.append(
            asdict(
                OptionIdentity(
                    c.contract_id, symbol, expiry, strike, "C", c.trading_class
                )
            )
        )
    return identities


async def prepare(config, journal, args, owners):
    settings = {}  # Customize this example in Python; connection settings come from the environment.
    if not set(settings.get("paper_stock", {})) <= {
        "universe",
        "lookback",
        "leg_weight",
        "shortable",
        "exit_retry_seconds",
    }:
        raise ValueError(
            "Unknown paper_stock setting; session timer names are fixed by this runner."
        )
    stock_rule = StockRule(
        **(
            {"universe": ("IWM", "QQQ"), "leg_weight": ".0008", "exit_timer": "exit"}
            | settings.get("paper_stock", {})
        )
    )
    limits = {
        "stock_budget": "2000",
        "spread_budget": "200",
        "zero_dte_budget": "200",
        "order_cap": "1000",
        "collar_bps": "25",
    } | settings.get("paper_limits", {})
    if set(limits) != {
        "stock_budget",
        "spread_budget",
        "zero_dte_budget",
        "order_cap",
        "collar_bps",
    }:
        raise ValueError("Unknown paper_limits setting.")
    vertical_settings = {
        "quantity": 1,
        "max_entry_debit": ".90",
        "max_leg_spread": ".10",
        "price_collar": ".05",
        "max_trades_per_day": 1,
        "max_exit_attempts": 2,
    }
    supplied = settings.get("paper_vertical", {})
    if not set(supplied) <= set(vertical_settings):
        raise ValueError("Unknown paper_vertical setting.")
    vertical_settings.update(supplied)
    budgets = {
        owners[1]: Decimal(str(limits["spread_budget"])),
        owners[2]: Decimal(str(limits["zero_dte_budget"])),
    }
    async with VerticalBroker(config, journal, strategy_budgets=budgets) as broker:
        calendar = await read_stock_sessions(broker, "SPY")
        now = datetime.now(timezone.utc)
        session = next((s for s in calendar.sessions if s.open <= now < s.close), None)
        if session is None:
            raise RuntimeError("No currently open broker liquid session; no entry.")
        stock_data = [
            await read_stock_history(broker, symbol) for symbol in stock_rule.universe
        ]
        common = set.intersection(
            *(set(b.end for b in data.bars) for data in stock_data)
        )
        common = set(sorted(common)[-stock_rule.lookback :])
        if len(common) < stock_rule.lookback:
            raise RuntimeError(
                "Not enough aligned observed minute bars for the stock lookback."
            )
        bars = [b for data in stock_data for b in data.bars if b.end in common]
        spots = {}
        for symbol in ("SPY", "QQQ"):
            q = await broker.quote(StockQuoteRequest(symbol))
            if q.status != "received" or q.bid is None or q.ask is None:
                raise RuntimeError(f"No live {symbol} quote.")
            spots[symbol] = (q.bid + q.ask) / 2
    # Preflight broker is closed: these read-only requests reuse the configured
    # client ID sequentially, never collide with the runtime connection.
    spy_chains = await discover_options(config, "SPY")
    later = sorted(
        {
            e
            for c in spy_chains
            if c.exchange == "SMART" and c.trading_class == "SPY"
            for e in c.expirations
            if session.label + timedelta(days=7)
            <= e
            <= session.label + timedelta(days=14)
        }
    )
    if not later:
        raise RuntimeError("No expiry 7–14 days away for the spread rule.")
    legs = {
        "spread": await resolve_pair(config, "SPY", later[0], spots["SPY"]),
        "dte_spy": await resolve_pair(config, "SPY", session.label, spots["SPY"]),
        "dte_qqq": await resolve_pair(config, "QQQ", session.label, spots["QQQ"]),
    }
    started = datetime.now(timezone.utc)
    exit_at = (
        session.close - timedelta(minutes=30)
        if args.session
        else started + timedelta(seconds=args.seconds)
    )
    if exit_at >= session.close or exit_at <= started + timedelta(seconds=60):
        raise RuntimeError(
            "Need at least 60 seconds before the configured pre-close exit."
        )
    monthly = SessionCalendar.from_exchange(
        "XNYS", date(session.label.year, 1, 1), date(session.label.year, 12, 31)
    )
    return dict(
        run_id=args.run_id,
        owners=owners,
        created_at=started,
        entry_at=started + timedelta(seconds=5),
        entry_attempt_offsets=[0, 45, 90],
        exit_at=exit_at,
        session_close=session.close,
        legs=legs,
        history=bars,
        session=session,
        stock_rule=asdict(stock_rule),
        vertical_settings=vertical_settings,
        budgets={
            owners[0]: limits["stock_budget"],
            owners[1]: limits["spread_budget"],
            owners[2]: limits["zero_dte_budget"],
        },
        order_cap=limits["order_cap"],
        collar_bps=limits["collar_bps"],
        monthly_schedule=monthly.monthly(
            "rebalance", at="open", offset=timedelta(minutes=5)
        ),
        notice="Actual paper quotes/bars/orders. One-shot schedule identities are distinct from the production monthly schedule.",
    )


def identity(data):
    return OptionIdentity(
        data["contract_id"],
        data["symbol"],
        date.fromisoformat(data["expiry"]),
        Decimal(data["strike"]),
        data["right"],
        data["trading_class"],
    )


def build_strategies(plan):
    stock, spread, dte = plan["owners"]
    start = datetime.fromisoformat(plan["created_at"])
    exit_at = datetime.fromisoformat(plan["exit_at"])
    rules = {}
    for name, legs in plan["legs"].items():
        rules[name] = VerticalRule(
            identity(legs[0]),
            identity(legs[1]),
            start,
            exit_at - timedelta(seconds=15),
            exit_at,
            entry_timer="entry-" + ("spread" if name == "spread" else "dte"),
            exit_timer="exit",
            same_day=name != "spread",
            **plan.get("vertical_settings", {"max_entry_debit": ".90"}),
        )
    strategies = {
        stock: MonthlyLongShort(
            StockRule(**({"exit_timer": "exit"} | plan["stock_rule"]))
        ),
        spread: ScheduledVertical(rules["spread"]),
        dte: ZeroDTE((rules["dte_spy"], rules["dte_qqq"])),
    }
    return strategies, rules


def entry_faults(plan, records):
    """Require filled opening intents before the example deliberately restarts."""
    stock, spread, dte = plan["owners"]
    first = {}
    for record in records:
        if (
            record.request.strategy_id == stock
            and isinstance(record.request, StockOrder)
            and record.executed_quantity
        ):
            first.setdefault(record.request.symbol, record)
    faults = []
    if (
        len(first) != 2
        or not set(first) <= set(plan["stock_rule"]["universe"])
        or {r.request.side for r in first.values()} != {"BUY", "SELL"}
        or any(r.executed_quantity != r.request.quantity for r in first.values())
    ):
        faults.append(
            "Monthly stock strategy requires filled long and short opening intents for its ranked pair."
        )
    for owner, key in ((spread, "spread"), (dte, "dte_spy"), (dte, "dte_qqq")):
        pair = tuple(leg["contract_id"] for leg in plan["legs"][key])
        if not any(
            r.request.strategy_id == owner
            and isinstance(r.request, VerticalOrder)
            and (r.request.long_leg.contract_id, r.request.short_leg.contract_id)
            == pair
            and r.request.side == "BUY"
            and r.executed_quantity == r.request.quantity
            for r in records
        ):
            faults.append(
                f"{owner} {plan['legs'][key][0]['symbol']} requires a fully executed entry."
            )
    return tuple(faults)


def acceptance_faults(
    plan,
    records,
    journal,
    account,
    broker_positions,
    paused,
    *,
    require_restart=False,
    restarted=False,
    require_trades=True,
):
    """Require execution evidence for every leg of the demonstrated workflow."""
    owners = plan["owners"]
    records = [
        record
        for record in records
        if record.account == account and record.request.strategy_id in owners
    ]
    faults = []
    if paused:
        faults.append("One or more strategies are paused.")
    if require_restart and not restarted:
        faults.append("The requested runtime reconnect/restart was not exercised.")
    if any(not record.terminal for record in records):
        faults.append("An order remains working.")
    if any(
        record.uncertain or record.filled_quantity > record.executed_quantity
        for record in records
    ):
        faults.append("An order outcome or execution attribution remains unresolved.")
    if require_trades:
        stock, spread, dte = owners
        faults.extend(entry_faults(plan, records))
        for owner, key in ((spread, "spread"), (dte, "dte_spy"), (dte, "dte_qqq")):
            symbol = plan["legs"][key][0]["symbol"]
            expected_pair = tuple(leg["contract_id"] for leg in plan["legs"][key])
            sides = {
                record.request.side
                for record in records
                if record.request.strategy_id == owner
                and isinstance(record.request, VerticalOrder)
                and record.request.symbol == symbol
                and (
                    record.request.long_leg.contract_id,
                    record.request.short_leg.contract_id,
                )
                == expected_pair
                and record.executed_quantity > 0
            }
            if sides != {"BUY", "SELL"}:
                faults.append(
                    f"{owner} {symbol} requires actual BUY and SELL combo leg executions."
                )
    contracts = {
        contract
        for record in records
        for contract in (
            (record.request.long_leg.contract_id, record.request.short_leg.contract_id)
            if isinstance(record.request, VerticalOrder)
            else (record.contract_id,)
        )
    }
    for owner in owners:
        if any(
            journal.attributed_position(account, owner, contract)
            for contract in contracts
        ):
            faults.append(f"{owner} still owns an open position.")
    actual = {}
    for position in broker_positions:
        contract = position.contract.contract_id
        actual[contract] = actual.get(contract, Decimal(0)) + position.quantity
    all_owners = {
        record.request.strategy_id
        for record in journal.orders()
        if record.account == account
    }
    for contract in contracts:
        expected = sum(
            (
                journal.attributed_position(account, owner, contract)
                for owner in all_owners
            ),
            Decimal(0),
        )
        if actual.get(contract, Decimal(0)) != expected:
            faults.append(
                f"Broker position for contract {contract} differs from execution-derived ownership."
            )
    return tuple(faults)


async def run(args):
    config = PaperAccountConfig.from_env()
    args.output.mkdir(parents=True, exist_ok=True)
    plan_path = args.output / "plan.json"
    fresh = not plan_path.exists()
    if fresh and args.close_only:
        raise ValueError("--close-only requires an existing saved session plan.")
    with OrderJournal(args.journal) as journal:
        if fresh:
            owners = [
                f"monthly-{args.run_id}",
                f"spread-{args.run_id}",
                f"zero-dte-{args.run_id}",
            ]
            plan = serialize(await prepare(config, journal, args, owners))
            plan_path.write_text(json.dumps(plan, indent=2) + "\n")
        else:
            plan = json.loads(plan_path.read_text())
            if plan["run_id"] != args.run_id:
                raise ValueError("Output directory belongs to another run ID.")
        budgets = {k: Decimal(v) for k, v in plan["budgets"].items()}
        entry = datetime.fromisoformat(plan["entry_at"])
        exit_at = (
            datetime.now(timezone.utc)
            if args.close_only
            else datetime.fromisoformat(plan["exit_at"])
        )
        rows = []
        for data in plan["history"]:
            rows.append(
                Bar(
                    data["symbol"],
                    datetime.fromisoformat(data["start"]),
                    datetime.fromisoformat(data["end"]),
                    data["open"],
                    data["high"],
                    data["low"],
                    data["close"],
                    data["volume"],
                )
            )
        history = HistoricalData(tuple(rows), "IBKR preflight completed minute bars")
        restarted = False
        for generation in range(2):
            strategies, rules = build_strategies(plan)
            broker = VerticalBroker(
                config,
                journal,
                strategy_budgets={
                    k: v for k, v in budgets.items() if k != plan["owners"][0]
                },
                max_contracts=plan.get("vertical_settings", {}).get("quantity", 1),
                on_update=output,
            )
            runtime = PaperRuntime(
                broker,
                strategies,
                {s: Instrument(s) for s in plan["stock_rule"]["universe"]},
                budgets=budgets,
                max_order_notional=Decimal(plan["order_cap"]),
                market_collar_bps=Decimal(plan["collar_bps"]),
                allow_short=True,
                defer_start=True,
            )
            started = (
                datetime.fromisoformat(plan["created_at"])
                if fresh and generation == 0
                else datetime.now(timezone.utc)
            )
            with (
                DurableSchedule(
                    args.output / "schedules.sqlite",
                    args.run_id + "-entries",
                    (
                        Timer(entry + timedelta(seconds=offset), name)
                        for name in ("rebalance", "entry-spread", "entry-dte")
                        for offset in (
                            plan.get("entry_attempt_offsets", [0])
                            if name != "rebalance"
                            else [0]
                        )
                    ),
                    started_at=started,
                ) as entries,
                DurableSchedule(
                    args.output / "schedules.sqlite",
                    args.run_id + "-exits",
                    (Timer(datetime.fromisoformat(plan["exit_at"]), "exit"),),
                    started_at=started,
                    policy="catch_up",
                ) as exits,
                DurableSchedule(
                    args.output / "schedules.sqlite",
                    args.run_id + "-regular-monthly",
                    (
                        Timer(datetime.fromisoformat(t["time"]), "rebalance")
                        for t in plan["monthly_schedule"]
                    ),
                    started_at=started,
                ) as monthly,
            ):
                async with runtime:
                    await runtime.warmup(history.panels())
                    (await runtime.start()).raise_for_failure()
                    if args.close_only:
                        await runtime.step(Timer(datetime.now(timezone.utc), "exit"))

                    async def refresh_options(selected):
                        for rule in selected:
                            if args.close_only and not any(
                                journal.attributed_position(
                                    config.account, owner, rule.long_leg.contract_id
                                )
                                for owner in strategies
                            ):
                                continue
                            template = VerticalOrder(
                                "quote-only",
                                "quote-only",
                                rule.long_leg,
                                rule.short_leg,
                                "BUY",
                                1,
                                Decimal(".01"),
                            )
                            try:
                                quotes = await broker.vertical_quotes(template)
                            except Exception as error:
                                output(
                                    dict(
                                        time=datetime.now(timezone.utc),
                                        quote_fault=str(error),
                                        symbol=rule.long_leg.symbol,
                                    )
                                )
                                continue
                            await runtime.step(
                                OptionQuotes(datetime.now(timezone.utc), quotes)
                            )

                    async def deliver(run):
                        selected = [
                            rule
                            for rule in rules.values()
                            if rule.entry_timer == run.timer.name
                        ]
                        await refresh_options(selected)
                        await runtime.step(run.timer)

                    next_status = datetime.now(timezone.utc)
                    next_history = next_status + timedelta(seconds=60)
                    while datetime.now(timezone.utc) < exit_at + timedelta(seconds=120):
                        await refresh_options(rules.values())
                        now = datetime.now(timezone.utc)
                        if not args.close_only and now >= next_history:
                            datasets = [
                                await read_stock_history(broker, symbol, seconds=300)
                                for symbol in plan["stock_rule"]["universe"]
                            ]
                            common = set.intersection(
                                *(set(b.end for b in data.bars) for data in datasets)
                            )
                            observed = [
                                b
                                for data in datasets
                                for b in data.bars
                                if b.end in common
                                and b.start >= runtime.history[b.symbol][-1].end
                            ]
                            if observed:
                                for panel in HistoricalData(
                                    tuple(observed), "IBKR completed minute bars"
                                ).panels():
                                    await runtime.step(panel)
                            next_history = now + timedelta(seconds=60)
                        if not args.close_only:
                            await monthly.poll_async(
                                datetime.now(timezone.utc), deliver
                            )
                            await entries.poll_async(
                                datetime.now(timezone.utc), deliver
                            )
                            await exits.poll_async(datetime.now(timezone.utc), deliver)
                        now = datetime.now(timezone.utc)
                        if now >= exit_at:
                            # Repeated explicit cutoff checks cancel stalled exits
                            # before bounded replacement; never mark them filled.
                            await runtime.step(Timer(now, "exit"))
                        records = [
                            r
                            for r in journal.orders()
                            if r.request.strategy_id in strategies
                        ]
                        held = any(
                            journal.attributed_position(config.account, owner, contract)
                            for owner in strategies
                            for contract in {
                                e.contract_id or r.contract_id
                                for r in records
                                for e in r.executions
                            }
                        )
                        working = any(not r.terminal for r in records)
                        if now >= next_status:
                            output(
                                dict(
                                    time=now,
                                    generation=generation,
                                    paused=sorted(runtime.paused),
                                    errors=runtime.errors,
                                    decisions={
                                        k: s.last_decision
                                        for k, s in strategies.items()
                                    },
                                    schedules=entries.status()
                                    + exits.status()
                                    + monthly.status(),
                                    held=held,
                                    working=working,
                                )
                            )
                            next_status = now + timedelta(seconds=15)
                        if now >= exit_at and not held and not working:
                            break
                        if (
                            args.restart
                            and not args.close_only
                            and not restarted
                            and now >= entry + timedelta(seconds=30)
                            and not working
                            and held
                            and not entry_faults(plan, records)
                        ):
                            restarted = True
                            output(
                                dict(
                                    restart="disconnecting and reconstructing runtime from the same journal"
                                )
                            )
                            break
                        await asyncio.sleep(2)
                    positions = await broker.positions()
                    records = [
                        r
                        for r in journal.orders()
                        if r.account == config.account
                        and r.request.strategy_id in strategies
                    ]
                    restarting = restarted and generation == 0
                    faults = (
                        ()
                        if restarting
                        else acceptance_faults(
                            plan,
                            records,
                            journal,
                            config.account,
                            positions,
                            runtime.paused,
                            require_restart=args.restart and not args.close_only,
                            restarted=restarted,
                            require_trades=not args.close_only,
                        )
                    )
                    result = dict(
                        run_id=args.run_id,
                        restarted=restarted,
                        generation=generation,
                        status="restarting"
                        if restarting
                        else "incomplete"
                        if faults
                        else "closed"
                        if args.close_only
                        else "completed",
                        acceptance_faults=faults,
                        orders=records,
                        errors=runtime.errors,
                        paused=sorted(runtime.paused),
                        decisions={k: s.last_decision for k, s in strategies.items()},
                        positions=positions,
                        schedules=entries.status() + exits.status() + monthly.status(),
                    )
                    clean = serialize(result)
                    for row in clean["orders"]:
                        row["account"] = "[PAPER_ACCOUNT]"
                    (args.output / "result.json").write_text(
                        json.dumps(clean, indent=2) + "\n"
                    )
                    if not restarting:
                        if faults:
                            raise RuntimeError(
                                "Session incomplete: "
                                + "; ".join(faults)
                                + " Inspect result.json."
                            )
                        output(
                            dict(
                                status="closed" if args.close_only else "completed",
                                result=str(args.output / "result.json"),
                            )
                        )
                        return 0
        raise RuntimeError("Restarted session did not reach completion.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journal", type=Path, default=Path("out/orders.sqlite"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=600)
    parser.add_argument("--session", action="store_true")
    parser.add_argument("--restart", action="store_true")
    parser.add_argument(
        "--close-only",
        action="store_true",
        help="Dispatch explicit exit events for this saved run; never enter new positions.",
    )
    args = parser.parse_args()
    if args.seconds < 90:
        parser.error("Smoke duration must be at least 90 seconds.")
    for name in ("ib_async", "ib_async.ib", "ib_async.client", "ib_async.wrapper"):
        logging.getLogger(name).disabled = True
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
