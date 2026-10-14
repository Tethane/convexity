import asyncio
from datetime import datetime, timezone

from convexity.brokers.ibkr import _new_client
from tests.brokers.test_ibkr import pytestmark


def test_live_utc_wire_and_qualified_replay_have_identical_execution_times():
    async def run():
        client = _new_client()
        client.client.decoder.serverVersion = 178
        captured = []
        client.wrapper.execDetails = lambda req, contract, execution: captured.append(
            execution.time
        )
        fields = [
            "11",
            "1",
            "2",
            "123",
            "SPY",
            "STK",
            "",
            "0",
            "",
            "",
            "SMART",
            "USD",
            "SPY",
            "SPY",
            "execution",
            "20261009-17:19:56",
            "DU123",
            "SMART",
            "BOT",
            "1",
            "100",
            "1",
            "71",
            "0",
            "1",
            "100",
            "smoke",
            "",
            "0",
            "",
            "1",
            "0",
        ]
        client.client.decoder.handlers[11](fields)
        fields[15] = "20261009 13:19:56 US/Eastern"
        client.client.decoder.handlers[11](fields)
        assert captured == [datetime(2026, 10, 9, 17, 19, 56, tzinfo=timezone.utc)] * 2

    asyncio.run(run())
