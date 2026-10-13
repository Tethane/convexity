"""Read-only endpoint fallback, explicit overrides and WSL discovery."""
import asyncio
from pathlib import Path

import pytest

from convexity import _ibkr_connection as connection


class Client:
    def __init__(self, target=None):
        self.target = target
        self.calls = []
        self.disconnects = 0

    async def connectAsync(self, host, port, **kwargs):
        self.calls.append((host, port))
        assert kwargs["readonly"] is True
        if (host, port) != self.target:
            raise ConnectionRefusedError("refused")

    def disconnect(self):
        self.disconnects += 1


def connect(client, **kwargs):
    return asyncio.run(connection.connect_readonly(
        client, client_id=72, timeout=1, **kwargs))


def test_wsl_gateway_fallback(monkeypatch):
    monkeypatch.setattr(connection, "_default_hosts", lambda: ("127.0.0.1", "172.31.96.1"))
    client = Client(("172.31.96.1", 4002))
    assert connect(client, host=None, port=None) == client.target
    assert client.calls == [("127.0.0.1", 4002), client.target]
    assert client.disconnects == 1


def test_explicit_endpoint_does_not_fall_back():
    client = Client()
    with pytest.raises(ConnectionError, match="custom:1234"):
        connect(client, host="custom", port=1234)
    assert client.calls == [("custom", 1234)]


def test_explicit_host_still_discovers_tws():
    client = Client(("custom", 7497))
    connect(client, host="custom", port=None)
    assert client.calls == [("custom", 4002), ("custom", 7497)]


def test_explicit_port_still_discovers_host(monkeypatch):
    monkeypatch.setattr(connection, "_default_hosts", lambda: ("127.0.0.1", "172.31.96.1"))
    client = Client(("172.31.96.1", 1234))
    connect(client, host=None, port=1234)
    assert client.calls == [("127.0.0.1", 1234), client.target]


def test_failure_reports_every_endpoint():
    client = Client()
    with pytest.raises(ConnectionError) as error:
        connect(client, host="custom", port=None)
    for port in (4002, 7497, 4001, 7496):
        assert f"custom:{port}" in str(error.value)
    assert client.disconnects == 4


def test_timeout_reserves_time_for_fallback():
    class SlowClient(Client):
        async def connectAsync(self, host, port, **kwargs):
            if port == 4002:
                await asyncio.sleep(10)
            await super().connectAsync(host, port, **kwargs)
    client = SlowClient(("custom", 7497))
    connect(client, host="custom", port=None)
    assert client.calls == [client.target]
    assert client.disconnects == 1


def test_cancellation_is_not_retried():
    class CancelledClient(Client):
        async def connectAsync(self, *args, **kwargs):
            raise asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        connect(CancelledClient(), host=None, port=None)


def test_wsl_default_route_without_dns_server(monkeypatch):
    def read(path):
        if path.name == "osrelease":
            return "6.6-microsoft-standard-WSL2"
        assert str(path) == "/proc/net/route"
        return "Iface Destination Gateway Flags\neth0 00000000 01601FAC 0003\n"
    monkeypatch.setattr(Path, "read_text", read)
    assert connection._default_hosts() == ("127.0.0.1", "172.31.96.1")


def test_non_wsl_uses_localhost(monkeypatch):
    monkeypatch.delenv("WSL_DISTRO_NAME", raising=False)
    monkeypatch.setattr(Path, "read_text", lambda _: "generic-linux")
    assert connection._default_hosts() == ("127.0.0.1",)


@pytest.mark.parametrize("target", [("custom", 7497), None])
def test_attempt_errors_are_quiet_and_final_failure_is_reported(caplog, target):
    import logging

    logger = logging.getLogger("ib_async.client")
    original_filters = list(logger.filters)

    class LoggingClient(Client):
        async def connectAsync(self, host, port, **kwargs):
            try:
                await super().connectAsync(host, port, **kwargs)
            except ConnectionRefusedError:
                logger.error("API connection failed: ConnectionRefusedError")
                logger.error("Make sure API port on TWS/IBG is open")
                raise

    client = LoggingClient(target)
    if target is None:
        with pytest.raises(ConnectionError, match="Attempted:.*custom:4002"):
            connect(client, host="custom", port=None)
    else:
        assert connect(client, host="custom", port=None) == target
    assert not caplog.records
    assert logger.filters == original_filters
    logger.error("API connection failed: outside discovery")
    assert "outside discovery" in caplog.text


def test_filter_preserves_unrelated_and_concurrent_logs(caplog):
    import logging

    logger = logging.getLogger("ib_async.client")

    async def run():
        entered = asyncio.Event()
        emitted = asyncio.Event()

        async def other_connection():
            await entered.wait()
            logger.error("API connection failed: unrelated connection")
            emitted.set()

        other = asyncio.create_task(other_connection())
        with connection._quiet_connection_attempt():
            logger.error("An unrelated broker error")
            logger.error("API connection failed: suppressed attempt")
            entered.set()
            await emitted.wait()
        await other

    asyncio.run(run())
    assert "unrelated connection" in caplog.text
    assert "An unrelated broker error" in caplog.text
    assert "suppressed attempt" not in caplog.text
