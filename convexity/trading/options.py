"""Explicit standard-option debit vertical intents; no automatic legging."""

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
import re
from typing import Literal


@dataclass(frozen=True)
class OptionIdentity:
    contract_id: int
    symbol: str
    expiry: date
    strike: Decimal
    right: Literal["C", "P"]
    trading_class: str
    multiplier: int = 100

    def __post_init__(self):
        if (
            type(self.contract_id) is not int
            or self.contract_id <= 0
            or not self.symbol
            or not self.trading_class
        ):
            raise ValueError(
                "Qualified option identity, symbol and trading class are required."
            )
        if (
            type(self.expiry) is not date
            or self.right not in ("C", "P")
            or self.multiplier != 100
        ):
            raise ValueError("Supply exact expiry, C/P and standard multiplier 100.")
        if (
            not isinstance(self.strike, Decimal)
            or not self.strike.is_finite()
            or self.strike <= 0
        ):
            raise ValueError("Option strike must be a positive finite Decimal.")


@dataclass(frozen=True)
class VerticalOrder:
    client_order_id: str
    strategy_id: str
    long_leg: OptionIdentity
    short_leg: OptionIdentity
    side: Literal["BUY", "SELL"]
    quantity: int
    limit_price: Decimal

    def __post_init__(self):
        if (
            not re.fullmatch(r"[A-Za-z0-9_-]{1,48}", self.client_order_id)
            or not self.strategy_id.strip()
        ):
            raise ValueError(
                "Supply a broker-safe order ID (1–48 characters) and strategy ID."
            )
        a, b = self.long_leg, self.short_leg
        if (a.symbol, a.expiry, a.right, a.trading_class, a.multiplier) != (
            b.symbol,
            b.expiry,
            b.right,
            b.trading_class,
            b.multiplier,
        ):
            raise ValueError(
                "Vertical legs must share underlying, expiry, right, class and multiplier."
            )
        if a.contract_id == b.contract_id or not (
            a.strike < b.strike if a.right == "C" else a.strike > b.strike
        ):
            raise ValueError(
                "Only debit call/put verticals with distinct qualified legs are supported."
            )
        if (
            self.side not in ("BUY", "SELL")
            or type(self.quantity) is not int
            or self.quantity <= 0
        ):
            raise ValueError(
                "Use BUY to open, SELL to close, and a positive contract count."
            )
        if (
            not isinstance(self.limit_price, Decimal)
            or not self.limit_price.is_finite()
            or not 0 < self.limit_price <= self.width
        ):
            raise ValueError(
                "Vertical limit must be a positive debit/credit no greater than its width."
            )

    @property
    def symbol(self) -> str:
        return self.long_leg.symbol

    @property
    def width(self) -> Decimal:
        return abs(self.long_leg.strike - self.short_leg.strike)

    @property
    def maximum_debit(self) -> Decimal:
        return self.quantity * self.limit_price * self.long_leg.multiplier

    def leg_side(self, contract_id: int) -> str:
        if contract_id == self.long_leg.contract_id:
            return self.side
        if contract_id == self.short_leg.contract_id:
            return "SELL" if self.side == "BUY" else "BUY"
        raise ValueError("Execution contract is not a leg of this vertical.")
