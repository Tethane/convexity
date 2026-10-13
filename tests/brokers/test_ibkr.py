import asyncio
import importlib.util
from dataclasses import FrozenInstanceError
from decimal import Decimal
from types import SimpleNamespace as NS
from typing import Any

import pytest

from convexity.brokers import ibkr

pytestmark = pytest.mark.skipif(importlib.util.find_spec("ib_async") is None,
                                reason="Install the ibkr extra to test the broker integration")


def config(**kwargs):
    values: dict[str, Any] = dict(account="DU123", port=7497, paper_session_confirmed=True) | kwargs
    return ibkr.PaperAccountConfig(**values)


def contract():
    return NS(conId=123, symbol="SPY", secType="STK", localSymbol="SPY",
              currency="USD", exchange="SMART", lastTradeDateOrContractMonth="",
              strike=0.0, right="", multiplier="", comboLegs=[])


class FakeClient:
    def __init__(self):
        self.disconnected = False
        self.connected = True
        self.calls = []
        self.accounts = ["DU123", "DU999"]
        self.summary = [NS(account="DU123", tag="NetLiquidation", value="1234.56", currency="USD"),
                        NS(account="DU999", tag="NetLiquidation", value="secret", currency="USD")]
        self.positions = [NS(account="DU123", contract=contract(), position=-2.5, avgCost=123.45),
                          NS(account="DU999", contract=contract(), position=999, avgCost=0)]
        order = NS(account="DU123", orderId=12, permId=34, clientId=99, action="BUY",
                   totalQuantity=1.5, orderType="LMT", tif="DAY", lmtPrice=500.25,
                   auxPrice=float.fromhex("0x1.fffffffffffffp+1023"))
        self.orders = [NS(contract=contract(), order=order, orderStatus=NS(status="Submitted"))]

    async def connectAsync(self, host, port, **kwargs):
        self.calls.append("connect")
        assert kwargs["readonly"] is True
        assert kwargs["clientId"] != 0
        assert kwargs["raiseSyncErrors"] is True
        assert not kwargs["fetchFields"].value

    def managedAccounts(self):
        return self.accounts

    async def accountSummaryAsync(self, account):
        self.calls.append("summary")
        return self.summary

    async def reqPositionsAsync(self):
        self.calls.append("positions")
        return self.positions

    async def reqAllOpenOrdersAsync(self):
        self.calls.append("orders")
        return self.orders

    def isConnected(self):
        return self.connected

    def disconnect(self):
        self.disconnected = True


@pytest.fixture
def client(monkeypatch):
    client = FakeClient()
    monkeypatch.setattr(ibkr, "_new_client", lambda: client)
    return client


def test_snapshot_filters_accounts_and_preserves_precision(client):
    snapshot = asyncio.run(ibkr.read_paper_account(config()))
    assert client.disconnected
    assert snapshot.summary == (ibkr.AccountValue("NetLiquidation", "1234.56", "USD"),)
    assert len(snapshot.positions) == 1
    assert snapshot.positions[0].quantity == Decimal("-2.5")
    assert snapshot.open_orders[0].limit_price == Decimal("500.25")
    assert snapshot.open_orders[0].auxiliary_price is None
    assert snapshot.open_orders[0].client_id == 99
    assert snapshot.started_at <= snapshot.completed_at
    with pytest.raises(FrozenInstanceError):
        snapshot.account = "changed"  # pyright: ignore[reportAttributeAccessIssue] -- deliberately test frozen result


def test_empty_positions_and_orders_are_valid(client):
    client.positions = []
    client.orders = []
    snapshot = asyncio.run(ibkr.read_paper_account(config()))
    assert snapshot.positions == snapshot.open_orders == ()


def test_wrong_account_stops_before_reading(client):
    client.accounts = ["U999"]
    with pytest.raises(ibkr.PaperAccountError, match="not available"):
        asyncio.run(ibkr.read_paper_account(config()))
    assert client.calls == ["connect"]
    assert client.disconnected


def test_missing_summary_is_not_success(client):
    client.summary = []
    with pytest.raises(ibkr.PaperAccountError, match="incomplete"):
        asyncio.run(ibkr.read_paper_account(config()))
    assert client.disconnected


@pytest.mark.parametrize("stage", ["connectAsync", "accountSummaryAsync", "reqPositionsAsync", "reqAllOpenOrdersAsync"])
def test_failures_disconnect(client, stage):
    async def fail(*args, **kwargs):
        raise ConnectionError("private broker detail")
    setattr(client, stage, fail)
    with pytest.raises(ibkr.PaperAccountError, match="Failed while") as exc:
        asyncio.run(ibkr.read_paper_account(config()))
    assert "private broker detail" not in str(exc.value)
    assert client.disconnected


def test_timeout_does_not_return_partial_snapshot(client):
    async def hang():
        await asyncio.Event().wait()
    client.reqAllOpenOrdersAsync = hang
    with pytest.raises(ibkr.PaperAccountError, match="Timed out while reading open orders"):
        asyncio.run(ibkr.read_paper_account(config(timeout=0.01)))
    assert client.disconnected


def test_cancellation_disconnects(client):
    async def cancel():
        raise asyncio.CancelledError
    client.reqPositionsAsync = cancel
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(ibkr.read_paper_account(config()))
    assert client.disconnected


def test_disconnect_after_read_is_failure(client):
    client.connected = False
    with pytest.raises(ibkr.PaperAccountError, match="disconnected"):
        asyncio.run(ibkr.read_paper_account(config()))
    assert client.disconnected


@pytest.mark.parametrize("kwargs", [dict(account="U123"), dict(account="DU_REPLACE_ME"),
    dict(paper_session_confirmed=False), dict(client_id=0), dict(port=True),
    dict(port=65536), dict(timeout=float("nan")), dict(timeout=0), dict(host="")])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        config(**kwargs)


def test_toml_config(tmp_path):
    path = tmp_path / "connection.toml"
    path.write_text('[ibkr]\naccount="DU123"\nport=7497\npaper_session_confirmed=true\n')
    assert ibkr.PaperAccountConfig.from_toml(path) == config()
    path.write_text(path.read_text() + 'typo=true\n')
    with pytest.raises(ValueError, match="Invalid"):
        ibkr.PaperAccountConfig.from_toml(path)


def test_real_client_connection_refusal_disconnects():
    import socket
    # Reserve a non-listening local port: exercise the installed client without
    # contacting any broker or guessing the user's account settings.
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        with pytest.raises(ibkr.PaperAccountError, match="connecting"):
            asyncio.run(ibkr.read_paper_account(config(port=sock.getsockname()[1], timeout=2)))


def test_cli_invalid_config_is_actionable(monkeypatch):
    import subprocess
    import sys
    monkeypatch.setenv("IBKR_ACCOUNT", "U123")
    result = subprocess.run([sys.executable, "scripts/ibkr_probe.py"], capture_output=True, text=True)
    assert result.returncode == 1
    assert "expected individual paper account" in result.stderr
    assert result.stdout == ""
    assert "Traceback" not in result.stderr


def test_cli_redacts_and_uses_public_api(client, tmp_path, monkeypatch, capsys):
    import json
    from scripts import ibkr_probe
    monkeypatch.setenv("IBKR_ACCOUNT", "DU123")
    monkeypatch.setenv("IBKR_PORT", "7497")
    monkeypatch.setenv("IBKR_PAPER_CONFIRMED", "true")
    client.positions = []
    client.orders = []
    monkeypatch.setattr("sys.argv", ["ibkr_probe", "--redact"])
    assert ibkr_probe.main() == 0
    output = capsys.readouterr()
    data = json.loads(output.out)
    assert data["account"] == "DU[REDACTED]"
    assert data["positions"] == data["open_orders"] == []
    assert data["connection"] == "verified; disconnected cleanly"
    assert "DU123" not in output.out + output.err
    assert client.disconnected


def test_lettered_paper_account_id_matches_actual_broker_format(client):
    client.accounts = ["DUQ123456"]
    client.summary = [NS(account="DUQ123456", tag="NetLiquidation", value="100", currency="USD")]
    snapshot = asyncio.run(ibkr.read_paper_account(config(account="DUQ123456")))
    assert snapshot.account == "DUQ123456"
    assert client.disconnected


def test_shared_toml_keeps_broker_fields_strict(tmp_path):
    path = tmp_path / "shared.toml"
    path.write_text('[ibkr]\naccount="DU123"\nport=7497\npaper_session_confirmed=true\n[backtest]\ninitial_cash="1000"\n')
    assert ibkr.PaperAccountConfig.from_toml(path) == config()


def test_environment_configuration_requires_expected_account_and_confirmation(monkeypatch):
    monkeypatch.delenv("IBKR_ACCOUNT", raising=False)
    with pytest.raises(ValueError, match="IBKR_ACCOUNT"):
        ibkr.PaperAccountConfig.from_env()
    monkeypatch.setenv("IBKR_ACCOUNT", "DU123")
    monkeypatch.setenv("IBKR_PORT", "7497")
    monkeypatch.setenv("IBKR_CLIENT_ID", "71")
    monkeypatch.setenv("IBKR_HOST", "127.0.0.1")
    monkeypatch.setenv("IBKR_TIMEOUT", "15")
    monkeypatch.delenv("IBKR_PAPER_CONFIRMED", raising=False)
    with pytest.raises(ValueError, match="Confirm"):
        ibkr.PaperAccountConfig.from_env()
    monkeypatch.setenv("IBKR_PAPER_CONFIRMED", "true")
    assert ibkr.PaperAccountConfig.from_env() == config()
