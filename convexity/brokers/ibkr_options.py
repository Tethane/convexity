"""Read-only option discovery and explicitly identified option quotes."""
import asyncio
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Literal

from . import ibkr
from .ibkr import PaperAccountConfig, PaperAccountError, Quote, StockQuoteRequest


@dataclass(frozen=True)
class OptionChain:
    underlying_id: int
    symbol: str
    exchange: str
    trading_class: str
    multiplier: str
    expirations: tuple[date, ...]
    strikes: tuple[Decimal, ...]


@dataclass(frozen=True)
class OptionQuoteRequest:
    symbol: str
    expiry: date
    strike: Decimal
    right: Literal["C", "P"]
    trading_class: str
    multiplier: str = "100"
    exchange: str = "SMART"
    currency: str = "USD"
    market_data_type: Literal["live", "frozen", "delayed", "delayed_frozen"] = "live"
    max_age: float = 30.0

    def __post_init__(self):
        StockQuoteRequest(self.symbol, self.exchange, self.currency,
                          market_data_type=self.market_data_type, max_age=self.max_age)
        strike = Decimal(str(self.strike))
        object.__setattr__(self, "strike", strike)
        if type(self.expiry) is not date or not strike.is_finite() or strike <= 0:
            raise ValueError("Supply an exact expiry date and positive finite strike.")
        if self.right not in ("C", "P") or not self.trading_class or self.multiplier != "100" or self.currency != "USD":
            raise ValueError("MVP options require C/P, trading class, USD and standard multiplier 100.")


async def discover_options(config: PaperAccountConfig, symbol: str) -> tuple[OptionChain, ...]:
    """Return broker chain metadata; each candidate still needs qualification.

    Expiry/strike Cartesian products do not guarantee existing contracts.
    A missing chain returns an empty tuple; connection/resolution failures raise.
    """
    request = StockQuoteRequest(symbol)
    client = ibkr._new_client()
    try:
        from ib_async import Stock
        async with asyncio.timeout(config.timeout):
            await ibkr._connect_paper(client, config)
            details = await client.reqContractDetailsAsync(Stock(request.symbol, "SMART", "USD"))
            if len(details) != 1 or details[0].contract.secType != "STK" or details[0].contract.conId <= 0:
                raise PaperAccountError("Option underlying must resolve to one stock contract.")
            underlying = details[0].contract
            chains = await client.reqSecDefOptParamsAsync(symbol, "", "STK", underlying.conId)
            return tuple(sorted((OptionChain(int(c.underlyingConId), symbol, c.exchange, c.tradingClass,
                c.multiplier, tuple(sorted(date.fromisoformat(e) for e in c.expirations)),
                tuple(sorted(Decimal(str(s)) for s in c.strikes))) for c in chains),
                key=lambda c: (c.exchange, c.trading_class, c.multiplier)))
    except PaperAccountError:
        raise
    except Exception as exc:
        raise PaperAccountError(f"Option discovery failed ({type(exc).__name__}); check Gateway and contract access.") from exc
    finally:
        client.disconnect()


async def read_option_quote(config: PaperAccountConfig, request: OptionQuoteRequest) -> Quote:
    """Qualify exact option identity and quote it, preserving data-quality status."""
    from ib_async import Option
    quote = await ibkr._read_quote(config, request, lambda: Option(
        request.symbol, request.expiry.strftime("%Y%m%d"), float(request.strike), request.right,
        request.exchange, multiplier=request.multiplier, currency=request.currency,
        tradingClass=request.trading_class), "OPT")
    c = quote.contract
    if (c.symbol, c.expiry[:8], c.strike, c.right, c.multiplier, c.currency, c.trading_class) != (
            request.symbol, request.expiry.strftime("%Y%m%d"), request.strike,
            request.right, request.multiplier, request.currency, request.trading_class):
        raise PaperAccountError("Resolved option identity differs from the requested standard contract.")
    return quote
