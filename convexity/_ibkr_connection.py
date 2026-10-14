"""Endpoint discovery for read-only IBKR data connections."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
import logging
import os
from pathlib import Path
import socket
from typing import Any


_discovering: ContextVar[bool] = ContextVar("ibkr_endpoint_discovery", default=False)


class _ConnectionAttemptFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not (
            _discovering.get()
            and (
                message.startswith("API connection failed:")
                or message == "Make sure API port on TWS/IBG is open"
            )
        )


@contextmanager
def _quiet_connection_attempt():
    # Task-local suppression preserves other clients' logs, including concurrent
    # connections. The aggregate exception below reports exhausted discovery.
    logger = logging.getLogger("ib_async.client")
    attempt_filter = _ConnectionAttemptFilter()
    token = _discovering.set(True)
    logger.addFilter(attempt_filter)
    try:
        yield
    finally:
        logger.removeFilter(attempt_filter)
        _discovering.reset(token)


def _default_hosts() -> tuple[str, ...]:
    hosts = ["127.0.0.1"]
    try:
        release = Path("/proc/sys/kernel/osrelease").read_text().lower()
        if "microsoft" in release or "WSL_DISTRO_NAME" in os.environ:
            # Linux exposes the default gateway in little-endian IPv4 notation.
            # Unlike resolv.conf, this also works with WSL DNS tunneling enabled.
            for line in Path("/proc/net/route").read_text().splitlines()[1:]:
                fields = line.split()
                if (
                    len(fields) >= 4
                    and fields[1] == "00000000"
                    and int(fields[3], 16) & 2
                ):
                    host = socket.inet_ntoa(bytes.fromhex(fields[2])[::-1])
                    if host != "0.0.0.0":
                        hosts.append(host)
    except OSError, ValueError:
        pass
    return tuple(dict.fromkeys(hosts))


async def connect_readonly(
    client: Any, *, host: str | None, port: int | None, client_id: int, timeout: float
) -> tuple[str, int]:
    """Try discovered endpoints within one total timeout; explicit fields pin that axis."""
    from ib_async import StartupFetchNONE

    endpoints = [
        (h, p)
        for p in ((port,) if port is not None else (4002, 7497, 4001, 7496))
        for h in ((host,) if host is not None else _default_hosts())
    ]
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    failures = []
    client.RaiseRequestErrors = True
    for index, (address, socket_port) in enumerate(endpoints):
        remaining = deadline - loop.time()
        if remaining <= 0:
            break
        # Reserve time for every fallback even when an endpoint silently drops packets.
        budget = remaining / (len(endpoints) - index)
        try:
            with _quiet_connection_attempt():
                async with asyncio.timeout(budget):
                    await client.connectAsync(
                        address,
                        socket_port,
                        clientId=client_id,
                        timeout=budget,
                        readonly=True,
                        raiseSyncErrors=True,
                        fetchFields=StartupFetchNONE,
                    )
            return address, socket_port
        except (OSError, TimeoutError) as exc:
            failures.append(f"{address}:{socket_port} ({type(exc).__name__}: {exc})")
            client.disconnect()
    raise ConnectionError(
        "IBKR connection failed. Attempted: "
        + "; ".join(failures)
        + ". Set IBKR_HOST/IBKR_PORT for a custom endpoint; check Gateway/TWS API access, "
        "trusted IPs and Windows firewall settings for WSL."
    )
