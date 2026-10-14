"""Scheduled debit-vertical demonstration rules using real broker quotes/fills."""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_FLOOR
import hashlib
import json
from zoneinfo import ZoneInfo

from convexity.brokers.ibkr import Quote
from convexity.runtime.types import (
    CancelOrder,
    Context,
    Rejection,
    Strategy,
    Timer,
    aware,
    decimal,
)
from convexity.trading import VerticalOrder
from convexity.trading.options import OptionIdentity


@dataclass(frozen=True)
class VerticalRule:
    long_leg: OptionIdentity
    short_leg: OptionIdentity
    entry_start: datetime
    entry_end: datetime
    exit_at: datetime
    entry_timer: str = "entry"
    exit_timer: str = "exit"
    quantity: int = 1
    max_entry_debit: Decimal = Decimal(".90")
    max_leg_spread: Decimal = Decimal(".10")
    price_collar: Decimal = Decimal(".05")
    max_trades_per_day: int = 1
    max_exit_attempts: int = 2
    exit_retry_after: timedelta = timedelta(seconds=20)
    quote_max_age: timedelta = timedelta(seconds=30)
    same_day: bool = False

    def __post_init__(self):
        for name in ("entry_start", "entry_end", "exit_at"):
            aware(getattr(self, name))
            object.__setattr__(self, name, getattr(self, name).astimezone(timezone.utc))
        if not self.entry_start < self.entry_end <= self.exit_at:
            raise ValueError(
                "Require entry_start < entry_end <= exit_at from an explicit session."
            )
        for name in ("max_entry_debit", "max_leg_spread", "price_collar"):
            object.__setattr__(self, name, decimal(getattr(self, name)))
        VerticalOrder(
            "validate",
            "validate",
            self.long_leg,
            self.short_leg,
            "BUY",
            self.quantity,
            self.max_entry_debit,
        )
        if (
            not self.entry_timer
            or not self.exit_timer
            or self.entry_timer == self.exit_timer
            or self.max_leg_spread <= 0
            or self.price_collar < 0
            or self.quote_max_age <= timedelta()
            or self.exit_retry_after <= timedelta()
            or any(
                type(v) is not int or v < 1
                for v in (self.max_trades_per_day, self.max_exit_attempts)
            )
        ):
            raise ValueError("Invalid timer names, quote bounds or attempt limits.")
        if self.same_day:
            from zoneinfo import ZoneInfo

            if (
                self.long_leg.expiry
                != self.entry_start.astimezone(ZoneInfo("America/New_York")).date()
            ):
                raise ValueError(
                    "0-DTE rule expiry must match the supplied US equity session date."
                )


class ScheduledVertical(Strategy):
    """Enter once on a liquid, bounded debit quote; exit on the cutoff timer.

    This deliberately simple quote-based signal demonstrates execution, not an
    alpha claim. Persisted broker intents bound entries/retries across restarts.
    Only broker leg executions determine holdings. Unbalanced legs remain a
    reconciliation fault; the strategy never converts a combo into naked orders.
    """

    def __init__(self, config: VerticalRule):
        self.config = config
        self.quotes: dict[int, Quote] = {}
        self.last_decision = "waiting for entry schedule"
        self._exit_requested = False

    def dump_state(self):
        # Quotes must be observed again after reconnect; an explicit request to
        # exit remains in force even when it preceded the configured cutoff.
        return {
            "version": 1,
            "exit_requested": self._exit_requested,
            "last_decision": self.last_decision,
        }

    def load_state(self, state):
        if state is None:
            return
        if (
            not isinstance(state, dict)
            or state.get("version") != 1
            or type(state.get("exit_requested")) is not bool
            or not isinstance(state.get("last_decision"), str)
        ):
            raise ValueError("Invalid saved vertical strategy state.")
        self._exit_requested = state["exit_requested"]
        self.last_decision = state["last_decision"]
        self.quotes.clear()

    def _records(self, context):
        return tuple(
            r
            for r in context.broker_orders
            if isinstance(r.request, VerticalOrder)
            and r.request.strategy_id == context.strategy_id
            and (r.request.long_leg, r.request.short_leg)
            == (self.config.long_leg, self.config.short_leg)
        )

    def _holdings(self, records):
        return tuple(
            sum(
                (
                    e.quantity if e.side == "BUY" else -e.quantity
                    for r in records
                    for e in r.executions
                    if e.contract_id == leg.contract_id
                ),
                Decimal(0),
            )
            for leg in (self.config.long_leg, self.config.short_leg)
        )

    def _prices(self, context):
        values = tuple(
            self.quotes.get(leg.contract_id)
            for leg in (self.config.long_leg, self.config.short_leg)
        )
        for quote in values:
            if (
                quote is None
                or quote.status != "received"
                or quote.bid is None
                or quote.ask is None
                or quote.bid <= 0
                or quote.ask < quote.bid
                or any(
                    t is None
                    or not timedelta() <= context.time - t <= self.config.quote_max_age
                    for t in (quote.bid_received_at, quote.ask_received_at)
                )
            ):
                self.last_decision = "skipped: missing/stale/non-live option quote"
                return None
        return values

    def _intent(self, context, side, quantity, price, attempt):
        rule = self.config
        identity = (
            context.strategy_id,
            rule.long_leg.contract_id,
            rule.short_leg.contract_id,
            rule.entry_start.isoformat(),
            side,
            attempt,
        )
        if side == "SELL":
            entries = [
                r.request.client_order_id
                for r in self._records(context)
                if r.request.side == "BUY"
            ]
            identity += (entries[-1] if entries else "unassigned",)
        order_id = (
            "opt-" + hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:40]
        )
        return VerticalOrder(
            order_id,
            context.strategy_id,
            rule.long_leg,
            rule.short_leg,
            side,
            quantity,
            price,
        )

    def on_event(self, event, context: Context):
        # Import lazily so the broker-free math API remains independently usable.
        from convexity.runtime.types import BrokerOrderUpdate, OptionQuotes

        rule = self.config
        if isinstance(event, Rejection):
            self.last_decision = f"fault: order rejected: {event.reason}"
            return None
        if isinstance(event, OptionQuotes):
            for quote in event.quotes:
                self.quotes[quote.contract.contract_id] = quote
        records = self._records(context)
        if isinstance(event, Timer) and event.name == rule.exit_timer:
            self._exit_requested = True
        if context.time >= rule.exit_at:
            self._exit_requested = True
        if self._exit_requested:
            working = tuple(r for r in records if not r.terminal)
            if working:
                # Existing exits are observed until an explicit retry timer; an
                # order update by itself must not cancel a just-submitted exit.
                cancel = tuple(
                    r
                    for r in working
                    if not r.cancel_pending
                    and (
                        r.request.side == "BUY"
                        or isinstance(event, Timer)
                        and event.name == rule.exit_timer
                        and r.created_at is not None
                        and context.time - r.created_at >= rule.exit_retry_after
                    )
                )
                if cancel:
                    self.last_decision = "exit: cancel working order before retry"
                    return tuple(CancelOrder(r.request.client_order_id) for r in cancel)
                return None
            holdings = self._holdings(records)
            if holdings == (0, 0):
                self.last_decision = "flat: confirmed by recorded leg executions"
                return None
            if (
                holdings[0] <= 0
                or holdings[0] != -holdings[1]
                or holdings[0] != holdings[0].to_integral_value()
            ):
                self.last_decision = (
                    "fault: unbalanced option legs require reconciliation"
                )
                return None
            last_entry = max(
                (i for i, record in enumerate(records) if record.request.side == "BUY"),
                default=-1,
            )
            attempts = sum(r.request.side == "SELL" for r in records[last_entry + 1 :])
            if attempts >= rule.max_exit_attempts:
                self.last_decision = (
                    "fault: exit retries exhausted; position still open"
                )
                return None
            quotes = self._prices(context)
            if quotes is None:
                return None
            a, b = quotes
            assert (
                a is not None
                and b is not None
                and a.bid is not None
                and b.ask is not None
            )
            price = max(
                Decimal(".01"),
                min(
                    abs(rule.long_leg.strike - rule.short_leg.strike),
                    a.bid - b.ask - rule.price_collar * (attempts + 1),
                ),
            )
            self.last_decision = "exit: submitted bounded combo close"
            return self._intent(
                context,
                "SELL",
                int(holdings[0]),
                price.quantize(Decimal(".01"), rounding=ROUND_FLOOR),
                attempts,
            )
        if not isinstance(event, Timer) or event.name != rule.entry_timer:
            return None
        if not rule.entry_start <= context.time < rule.entry_end:
            self.last_decision = "skipped: outside entry window"
            return None
        session_date = context.time.astimezone(ZoneInfo("America/New_York")).date()
        daily_entries = sum(
            isinstance(r.request, VerticalOrder)
            and r.request.side == "BUY"
            and r.request.strategy_id == context.strategy_id
            and r.request.symbol == rule.long_leg.symbol
            and (
                r.created_at is None
                or r.created_at.astimezone(ZoneInfo("America/New_York")).date()
                == session_date
            )
            for r in context.broker_orders
        )
        if (
            daily_entries >= rule.max_trades_per_day
            or any(not r.terminal for r in records)
            or any(self._holdings(records))
        ):
            self.last_decision = "skipped: existing exposure/order or daily trade limit"
            return None
        quotes = self._prices(context)
        if quotes is None:
            return None
        a, b = quotes
        assert (
            a is not None
            and b is not None
            and a.ask is not None
            and a.bid is not None
            and b.ask is not None
            and b.bid is not None
        )
        debit = a.ask - b.bid
        if (
            max(a.ask - a.bid, b.ask - b.bid) > rule.max_leg_spread
            or not 0 < debit <= rule.max_entry_debit
        ):
            self.last_decision = "skipped: quote spread/debit signal not satisfied"
            return None
        price = min(rule.max_entry_debit, debit + rule.price_collar).quantize(
            Decimal(".01"), rounding=ROUND_FLOOR
        )
        if price <= 0:
            self.last_decision = "skipped: debit cap is below the supported cent tick"
            return None
        self.last_decision = "entry: liquid debit quote satisfies configured bounds"
        return self._intent(context, "BUY", rule.quantity, price, daily_entries)


class ZeroDTE(Strategy):
    """Run the same bounded intraday rule for several supplied underlyings."""

    def __init__(self, rules: tuple[VerticalRule, ...]):
        if (
            not rules
            or any(not r.same_day for r in rules)
            or len({r.long_leg.symbol for r in rules}) != len(rules)
        ):
            raise ValueError("Supply one same-day rule per distinct underlying.")
        self.config = tuple(rules)
        self.strategies = tuple(ScheduledVertical(r) for r in rules)

    def dump_state(self):
        return {
            "version": 1,
            "underlyings": [s.config.long_leg.symbol for s in self.strategies],
            "strategies": [s.dump_state() for s in self.strategies],
        }

    def load_state(self, state):
        if state is None:
            return
        if (
            not isinstance(state, dict)
            or state.get("version") != 1
            or state.get("underlyings")
            != [s.config.long_leg.symbol for s in self.strategies]
            or not isinstance(state.get("strategies"), list)
            or len(state["strategies"]) != len(self.strategies)
        ):
            raise ValueError(
                "Saved 0-DTE state does not match the configured underlyings."
            )
        for strategy, saved in zip(self.strategies, state["strategies"]):
            strategy.load_state(saved)

    @property
    def last_decision(self) -> dict[str, str]:
        return {s.config.long_leg.symbol: s.last_decision for s in self.strategies}

    def on_event(self, event, context):
        commands = []
        for strategy in self.strategies:
            output = strategy.on_event(event, context)
            if output is not None:
                commands.extend(output if isinstance(output, tuple) else (output,))
        return tuple(commands)
