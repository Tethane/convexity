"""Currency-aware account snapshots and explicit, timestamped FX valuations."""
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from types import MappingProxyType
from typing import Mapping
import re


def currency(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Z]{3}", value):
        raise ValueError("Currency must be a three-letter uppercase code.")
    return value


def number(value) -> Decimal:
    result = Decimal(str(value))
    if not result.is_finite():
        raise ValueError("Account values must be finite.")
    return result


@dataclass(frozen=True)
class FXRate:
    """One unit of currency is worth rate units of base_currency, known at time."""
    currency: str
    base_currency: str
    rate: Decimal
    time: datetime

    def __post_init__(self):
        currency(self.currency)
        currency(self.base_currency)
        object.__setattr__(self, "rate", number(self.rate))
        if self.rate <= 0 or (self.currency == self.base_currency and self.rate != 1):
            raise ValueError("FX rates must be positive; identity rates must equal one.")
        if self.time.tzinfo is None or self.time.utcoffset() is None:
            raise ValueError("FX timestamps must be timezone-aware.")


@dataclass(frozen=True)
class AccountState:
    """Totals and margin are in base_currency; cash_balances retain native units.

    None means a broker did not supply a field. buying_power is broker/model
    buying power; sizing_power is conservative capacity after local reservations.
    Conversion changes valuation, never cash balances or currency exposure.
    """
    time: datetime
    base_currency: str
    cash_balances: Mapping[str, Decimal]
    fx_rates: Mapping[str, FXRate]
    cash: Decimal
    equity: Decimal
    source: str
    initial_margin: Decimal | None = None
    maintenance_margin: Decimal | None = None
    available_funds: Decimal | None = None
    excess_liquidity: Decimal | None = None
    buying_power: Decimal | None = None
    reserved: Decimal = Decimal(0)
    sizing_power: Decimal | None = None
    cash_balances_complete: bool = True

    def __post_init__(self):
        currency(self.base_currency)
        if self.time.tzinfo is None or self.time.utcoffset() is None:
            raise ValueError("Account timestamps must be timezone-aware.")
        balances = {currency(c): number(v) for c, v in self.cash_balances.items()}
        rates = dict(self.fx_rates)
        for c, rate in rates.items():
            if c != rate.currency or rate.base_currency != self.base_currency or rate.time > self.time:
                raise ValueError("FX rates must match the account currencies and cannot be from the future.")
        object.__setattr__(self, "cash_balances", MappingProxyType(balances))
        object.__setattr__(self, "fx_rates", MappingProxyType(rates))
        for key in ("cash", "equity", "initial_margin", "maintenance_margin", "available_funds",
                    "excess_liquidity", "buying_power", "reserved", "sizing_power"):
            value = getattr(self, key)
            if value is not None:
                object.__setattr__(self, key, number(value))

    def convert(self, amount, from_currency: str, to_currency: str | None = None,
                *, max_age: timedelta | None = None) -> Decimal:
        target = currency(to_currency or self.base_currency)
        source = currency(from_currency)
        if max_age is not None and max_age < timedelta(0):
            raise ValueError("max_age must be nonnegative.")
        def rate(c):
            if c == self.base_currency:
                return Decimal(1)
            quote = self.fx_rates.get(c)
            if quote is None:
                raise ValueError(f"No observed FX rate for {c}/{self.base_currency}.")
            if max_age is not None and self.time - quote.time > max_age:
                raise ValueError(f"Stale FX rate for {c}/{self.base_currency}.")
            return quote.rate
        return number(amount) if source == target else number(amount) * rate(source) / rate(target)

    def equity_in(self, currency: str, *, max_age: timedelta | None = None) -> Decimal:
        return self.convert(self.equity, self.base_currency, currency, max_age=max_age)


def broker_account(snapshot) -> AccountState:
    """Interpret IBKR account totals separately from per-currency ledger rows."""
    rows = {}
    for row in snapshot.summary:
        key = (row.tag.removeprefix("$LEDGER-"), row.currency)
        if key in rows and rows[key] != row.value:
            raise ValueError(f"Conflicting broker account value: {key}.")
        rows[key] = row.value
    bases = {c for (tag, c) in rows if tag == "NetLiquidation" and c not in ("", "BASE")}
    if len(bases) != 1:
        raise ValueError("A unique broker base currency for NetLiquidation is required.")
    base = bases.pop()
    def total(tag, required=False):
        value = rows.get((tag, base), rows.get((tag, "BASE")))
        if value is None and required:
            raise ValueError(f"Broker {tag} in {base} is required.")
        return None if value is None else number(value)
    rates = {c: FXRate(c, base, number(v), snapshot.completed_at)
             for (tag, c), v in rows.items() if tag == "ExchangeRate" and c not in ("", "BASE")}
    balances = {c: number(v) for (tag, c), v in rows.items()
                if tag == "CashBalance" and c not in ("", "BASE")}
    # A converted TotalCashValue is not evidence of native cash ownership.
    cash, equity = total("TotalCashValue", True), total("NetLiquidation", True)
    assert cash is not None and equity is not None
    return AccountState(snapshot.completed_at, base, balances, rates,
                        cash, equity, "broker",
                        initial_margin=total("InitMarginReq"), maintenance_margin=total("MaintMarginReq"),
                        available_funds=total("AvailableFunds"), excess_liquidity=total("ExcessLiquidity"),
                        buying_power=total("BuyingPower"), cash_balances_complete=bool(balances))
