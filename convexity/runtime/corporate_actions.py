"""Explicit, sourced corporate actions for raw-price event replay.

No provider data is inferred. Adjusted share-unit datasets cannot safely consume
nominal share entitlements; actions against those datasets fail explicitly.
"""

from dataclasses import dataclass, replace
from datetime import datetime
from decimal import Decimal
from typing import Literal

from .types import Position, ZERO, aware, decimal


class UnsupportedCorporateAction(ValueError):
    """An event cannot be accounted for without inventing missing terms."""


@dataclass(frozen=True)
class CorporateAction:
    action_id: str
    symbol: str
    kind: Literal["split", "dividend", "merger", "delisting", "bankruptcy", "spinoff"]
    effective_at: datetime
    known_at: datetime
    source: str
    ratio: Decimal = Decimal(1)
    cash_per_share: Decimal = ZERO
    target_symbol: str | None = None
    pay_at: datetime | None = None
    valuation_price: Decimal | None = None
    valuation_at: datetime | None = None

    def __post_init__(self):
        aware(self.effective_at)
        aware(self.known_at)
        if self.pay_at is not None:
            aware(self.pay_at)
        if self.valuation_at is not None:
            aware(self.valuation_at)
        if self.valuation_price is not None:
            object.__setattr__(self, "valuation_price", decimal(self.valuation_price))
            if self.valuation_price <= 0 or self.valuation_at != self.effective_at:
                raise ValueError(
                    "A sourced valuation must be positive and observed at the effective timestamp."
                )
        elif self.valuation_at is not None:
            raise ValueError("valuation_at requires a sourced valuation_price.")
        if not self.action_id or not self.symbol or not self.source:
            raise ValueError(
                "Corporate actions require identity, security symbol, and provenance."
            )
        if self.kind not in (
            "split",
            "dividend",
            "merger",
            "delisting",
            "bankruptcy",
            "spinoff",
        ):
            raise UnsupportedCorporateAction(
                f"Unsupported corporate action: {self.kind}."
            )
        object.__setattr__(self, "ratio", decimal(self.ratio))
        object.__setattr__(self, "cash_per_share", decimal(self.cash_per_share))
        if self.ratio <= 0 or self.cash_per_share < 0:
            raise ValueError(
                "Action ratio must be positive and cash consideration nonnegative."
            )
        if self.known_at > self.effective_at:
            raise ValueError(
                "Late-discovered actions require reconciliation, not a retroactive replay."
            )
        if self.pay_at is not None and (
            self.kind != "dividend" or self.pay_at < self.effective_at
        ):
            raise ValueError(
                "Only dividends accept pay_at, at or after the entitlement timestamp."
            )
        if self.kind == "dividend" and self.pay_at is None:
            raise ValueError("Dividends require an explicit sourced payment timestamp.")
        if self.kind == "dividend" and self.cash_per_share <= 0:
            raise ValueError("Dividends need a positive sourced per-share entitlement.")
        if self.kind == "merger" and not self.target_symbol and not self.cash_per_share:
            raise ValueError("A merger requires explicit cash or share consideration.")
        if self.kind == "split" and (
            self.cash_per_share
            or self.target_symbol
            or self.valuation_price is not None
        ):
            raise UnsupportedCorporateAction(
                "Split cash-in-lieu or other consideration requires explicit supported terms."
            )
        if self.kind == "dividend" and (self.ratio != 1 or self.target_symbol):
            raise UnsupportedCorporateAction(
                "Share dividends require explicit supported share-entitlement accounting."
            )
        if self.kind != "dividend" and self.valuation_price is not None:
            raise ValueError(
                "valuation_price is currently supported only for dividend ex-date observations."
            )
        if self.kind == "merger" and not self.target_symbol and self.ratio != 1:
            raise ValueError(
                "A cash-only merger cannot specify a share-exchange ratio."
            )
        if self.target_symbol == self.symbol:
            raise ValueError("Merger destination must be a different security.")


def apply_corporate_action(runtime, action: CorporateAction, price_basis: str) -> None:
    """Apply an entitlement atomically to existing Runtime accounting state.

    Working affected orders and fractional entitlements require reconciliation;
    they are never silently canceled, rounded, or priced. Dividend receivables
    contribute to equity from ex time and become spendable cash only at pay time.
    """
    if runtime.time != action.effective_at or action.known_at > runtime.time:
        raise ValueError("Apply corporate actions at their effective event timestamp.")
    applied = getattr(runtime, "_corporate_actions_applied", set())
    if action.action_id in applied:
        raise ValueError(f"Duplicate corporate action {action.action_id}.")
    if price_basis != "unadjusted_no_corporate_actions":
        raise UnsupportedCorporateAction(
            "Corporate actions require raw prices; adjusted share units risk double application."
        )
    if action.kind in ("delisting", "bankruptcy", "spinoff"):
        raise UnsupportedCorporateAction(
            f"{action.kind} for {action.symbol}: supply verified settlement/consideration; no price is invented."
        )
    if action.symbol not in runtime.instruments:
        raise ValueError(f"Unknown action security {action.symbol}.")
    affected = {action.symbol, action.target_symbol}
    if any(
        not order.terminal and order.request.symbol in affected
        for order in runtime.orders.values()
    ):
        raise UnsupportedCorporateAction(
            "Reconcile working orders before applying a corporate action."
        )
    positions = [
        (key, p)
        for key, p in runtime.positions.items()
        if key[1] == action.symbol and p.quantity
    ]
    updates = {}
    cash = ZERO
    mark = runtime.marks.get(action.symbol)
    if action.kind == "split":
        for key, p in positions:
            quantity = p.quantity * action.ratio
            if quantity != quantity.to_integral_value():
                raise UnsupportedCorporateAction(
                    "Fractional split entitlement requires verified cash-in-lieu terms."
                )
            updates[key] = replace(
                p, quantity=int(quantity), average_cost=p.average_cost / action.ratio
            )
        new_mark = mark / action.ratio if mark is not None else None
    elif action.kind == "dividend":
        cash = sum((p.quantity * action.cash_per_share for _, p in positions), ZERO)
        if positions and action.valuation_price is None:
            raise UnsupportedCorporateAction(
                "Dividend entitlement requires a sourced ex-date valuation; no market price is inferred."
            )
        new_mark = (
            action.valuation_price if action.valuation_price is not None else mark
        )
    else:
        target = action.target_symbol
        target_mark = None
        if target:
            if target not in runtime.instruments or target not in runtime.marks:
                raise UnsupportedCorporateAction(
                    "Share consideration requires an observed destination valuation."
                )
            if (
                runtime.instruments[target].currency
                != runtime.instruments[action.symbol].currency
            ):
                raise UnsupportedCorporateAction(
                    "Cross-currency mergers require explicit currency consideration."
                )
            target_mark = runtime.marks[target]
        for (owner, symbol), p in positions:
            quantity = p.quantity * action.ratio if target else ZERO
            if quantity != quantity.to_integral_value():
                raise UnsupportedCorporateAction(
                    "Fractional merger entitlement requires verified cash-in-lieu terms."
                )
            proceeds = p.quantity * action.cash_per_share
            cash += proceeds
            value = proceeds + quantity * (target_mark or ZERO)
            updates[(owner, symbol)] = replace(
                p,
                quantity=0,
                average_cost=ZERO,
                realized_pnl=p.realized_pnl + value - p.quantity * p.average_cost,
            )
            if target:
                assert target_mark is not None
                old = runtime.positions.get(
                    (owner, target), Position(currency=p.currency)
                )
                if old.quantity and old.quantity * quantity < 0:
                    raise UnsupportedCorporateAction(
                        "Merger consideration crossing an existing position requires reconciliation."
                    )
                total = old.quantity + int(quantity)
                cost = (
                    (old.quantity * old.average_cost + quantity * target_mark) / total
                    if total
                    else ZERO
                )
                updates[(owner, target)] = replace(
                    old, quantity=total, average_cost=cost
                )
        new_mark = mark
    # Commit only once every entitlement and valuation has passed validation.
    runtime.positions.update(updates)
    if new_mark is not None:
        runtime.marks[action.symbol] = new_mark
        if action.kind == "dividend" and action.valuation_price is not None:
            runtime.mark_times[action.symbol] = action.valuation_at
        sources = getattr(runtime, "_corporate_mark_sources", {})
        sources[action.symbol] = f"corporate_action:{action.source}"
        runtime._corporate_mark_sources = sources
        # Split changes units, not observation freshness. A supplied ex-date
        # valuation is a sourced observation with its own explicit timestamp.
    code = runtime.instruments[action.symbol].currency
    receivables = getattr(runtime, "_corporate_receivables", {})
    if (
        action.kind == "dividend"
        and action.pay_at is not None
        and action.pay_at > runtime.time
    ):
        receivables[action.action_id] = (action.symbol, cash)
        runtime._corporate_receivables = receivables
        kind = "dividend_receivable"
    else:
        runtime._cash_change(code, cash)
        kind = action.kind
    runtime.cash_flows.append(
        dict(
            time=runtime.time,
            kind=kind,
            amount=cash,
            currency=code,
            action_id=action.action_id,
            symbol=action.symbol,
            source=action.source,
        )
    )
    applied.add(action.action_id)
    runtime._corporate_actions_applied = applied


def settle_dividend(runtime, action: CorporateAction) -> None:
    if runtime.time != action.pay_at:
        raise ValueError("Settle dividends only at their payment timestamp.")
    receivables = getattr(runtime, "_corporate_receivables", {})
    if action.action_id not in receivables:
        raise ValueError(
            "No recorded dividend entitlement; payment cannot infer past holdings."
        )
    symbol, amount = receivables.pop(action.action_id)
    code = runtime.instruments[symbol].currency
    runtime._cash_change(code, amount)
    runtime.cash_flows.append(
        dict(
            time=runtime.time,
            kind="dividend_payment",
            amount=amount,
            currency=code,
            action_id=action.action_id,
            symbol=symbol,
            source=action.source,
        )
    )
