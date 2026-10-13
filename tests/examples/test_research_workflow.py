"""CLI options configure strategy declarations; research owns only dates."""
from datetime import date
from decimal import Decimal
from types import SimpleNamespace

import pytest

from examples import strategy_workflow as workflow


def test_research_cli_requires_symbols_and_dates_before_loading_account(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Research must not load paper account configuration")
    monkeypatch.setattr(workflow.cx.PaperAccountConfig, "from_env", forbidden)
    for args in ([], ["--symbol", "SPY"], ["--symbol", "SPY", "--start", "2025-01-02"]):
        with pytest.raises(SystemExit) as error:
            workflow.main(args)
        assert error.value.code == 2


@pytest.mark.parametrize('bar_size', ['1 min', '7 mins', '90 secs', '1 week'])
def test_research_cli_routes_requirements_through_strategy_contract(monkeypatch, tmp_path, bar_size):
    captured = {}
    def forbidden(*args, **kwargs):
        raise AssertionError("Historical research must not load account configuration")
    monkeypatch.setattr(workflow.cx.PaperAccountConfig, "from_env", forbidden)
    class RecordingResearch:
        def __init__(self, start, end):
            captured.update(start=start, end=end)
        def run(self, strategies, **options):
            captured.update(options, instances=[factory() for factory in strategies])
            return SimpleNamespace(status="completed", fills=(), orders=(), open_orders=(),
                metrics={"final_equity": Decimal(100000)},
                config={"data_source": "IBKR:test routing only", "price_basis": "split_adjusted_price_return",
                        "data_identity": "routing-test"})
    monkeypatch.setattr(workflow.cx, "Research", RecordingResearch)
    assert workflow.main(["--symbols", "SPY", "QQQ", "--start", "2025-01-02", "--end", "2025-01-03",
                          "--cache", str(tmp_path), "--warmup-sessions", "2", "--refresh", "--bar-size", bar_size]) == 0
    assert captured["start"] == date(2025, 1, 2) and captured["end"] == date(2025, 1, 3)
    assert captured["cache"] == tmp_path and captured["refresh"] is True
    assert {strategy.metadata.name for strategy in captured["instances"]} == {"momentum_SPY", "momentum_QQQ"}
    for strategy in captured["instances"]:
        assert strategy.contract.symbols == (strategy.config["symbol"],)
        assert strategy.contract.warmup_sessions == 2
        assert strategy.contract.subscriptions[0].bar_size == bar_size


def test_paper_cli_rejects_daily_research_cadence_before_connecting(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("An incompatible paper cadence must never connect")
    monkeypatch.setattr(workflow.cx.PaperAccountConfig, "from_env", forbidden)
    with pytest.raises(SystemExit) as error:
        workflow.main(["--mode", "paper", "--symbol", "SPY", "--bar-size", "1 day"])
    assert error.value.code == 2
