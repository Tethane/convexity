import asyncio
from datetime import date
from decimal import Decimal
from types import SimpleNamespace as NS

import pytest

from convexity.brokers import ibkr
from convexity.brokers.ibkr_options import OptionQuoteRequest, discover_options, read_option_quote
from tests.brokers.test_ibkr import config, contract, pytestmark
from tests.brokers.test_ibkr_quote import QuoteClient


class OptionClient(QuoteClient):
    async def reqContractDetailsAsync(self, request):
        c = contract()
        if request.secType == "OPT":
            c.secType = "OPT"
            c.lastTradeDateOrContractMonth = request.lastTradeDateOrContractMonth
            c.strike, c.right, c.multiplier = request.strike, request.right, request.multiplier
            c.tradingClass = request.tradingClass
        return [NS(contract=c)]

    async def reqSecDefOptParamsAsync(self, symbol, exchange, kind, con_id):
        assert (symbol, exchange, kind, con_id) == ("SPY", "", "STK", 123)
        return [NS(underlyingConId=123, exchange="SMART", tradingClass="SPY", multiplier="100",
                   expirations={"20261009"}, strikes={500, 501})]


def test_exact_option_quote_and_chain(monkeypatch):
    client = OptionClient()
    monkeypatch.setattr(ibkr, "_new_client", lambda: client)
    chains = asyncio.run(discover_options(config(), "SPY"))
    assert chains[0].expirations == (date(2026, 10, 9),)
    assert chains[0].strikes == (Decimal(500), Decimal(501))
    quote = asyncio.run(read_option_quote(config(), OptionQuoteRequest(
        "SPY", date(2026, 10, 9), Decimal(500), "C", "SPY")))
    assert quote.status == "received"
    assert quote.contract.security_type == "OPT"
    assert quote.contract.strike == 500
    assert client.disconnected


def test_invalid_option_request():
    with pytest.raises(ValueError):
        OptionQuoteRequest("SPY", date(2026, 10, 9), Decimal("NaN"), "C", "SPY")
    with pytest.raises(ValueError):
        OptionQuoteRequest("SPY", date(2026, 10, 9), Decimal(500), "C", "SPY", multiplier="10")


def test_option_nonlive_quote_is_not_usable(monkeypatch):
    client = OptionClient()
    client.data_type = 3
    monkeypatch.setattr(ibkr, "_new_client", lambda: client)
    quote = asyncio.run(read_option_quote(config(), OptionQuoteRequest(
        "SPY", date(2026, 10, 9), Decimal(500), "C", "SPY")))
    assert quote.status == "non_live"
