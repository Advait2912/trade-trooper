"""Paper-runner orchestration tests (mocked trading API + fake pipeline)."""


import httpx
import pytest
import respx

import trading.runner as runner_mod
from schemas.decision import DecisionResult
from schemas.pipeline import FinalReport
from trading.runner import PaperRunner, is_market_open
from utils.config import Settings

PAPER = "https://paper-api.alpaca.markets"


def _settings(**kw) -> Settings:
    base = dict(alpaca_api_key="k", alpaca_api_secret="s", trading_enabled=True,
                trading_interval_min=1, max_open_positions=1, account_capital=100000.0,
                risk_per_trade_pct=0.01, max_position_pct=0.05, min_risk_reward=1.0,
                min_confidence=0.35, order_type="limit", limit_slippage_pct=0.0)
    base.update(kw)
    return Settings(**base)


def _report(decision="hold") -> FinalReport:
    from datetime import datetime, timezone

    return FinalReport(
        ticker="NVDA",
        timestamp=datetime.now(timezone.utc).isoformat(),
        decision=DecisionResult(
            trade_decision=decision,
            symbol="NVDA",
            entry_price=100.0,
            stop_loss=95.0,
            take_profit=110.0,
            position_shares=1.0 if decision == "long_equity" else 0.0,
            confidence_score=0.6,
            composite_bias="bullish" if decision != "hold" else "neutral",
            agreement_score=0.7,
            instrument="equity" if decision == "long_equity" else "none",
            rationale="test",
            summary="test",
        ),
    )


class FakePipeline:
    def __init__(self, *a, **kw):
        pass

    async def run(self, ticker):
        return FakePipeline._report


@pytest.fixture(autouse=True)
def _fake_pipeline(monkeypatch):
    monotonic = {"report": _report("hold")}

    class FP:
        def __init__(self, *a, **kw):
            self.settings = None

        async def run(self, ticker):
            return monotonic["report"]

    monkeypatch.setattr(runner_mod, "Pipeline", FP)
    return monotonic


@respx.mock
async def test_run_once_hold_journals_cycle_no_order(tmp_path):
    respx.get(f"{PAPER}/v2/account").mock(
        return_value=httpx.Response(200, json={"equity": "100000", "cash": "100000"})
    )
    respx.get(f"{PAPER}/v2/positions").mock(return_value=httpx.Response(200, json=[]))

    runner = PaperRunner(_settings(), "NVDA", journal_path=str(tmp_path / "j.db"))
    result = await runner.run_once()

    assert result["decision"] == "hold"
    assert result["equity"] == 100000.0
    cycles = runner.journal.cycles()
    assert cycles and cycles[0]["decision"] == "hold"
    assert runner.journal.orders() == []


@respx.mock
async def test_run_once_long_equity_submits_order(tmp_path, _fake_pipeline):
    respx.get(f"{PAPER}/v2/account").mock(
        return_value=httpx.Response(200, json={"equity": "100000", "cash": "100000"})
    )
    respx.get(f"{PAPER}/v2/positions").mock(return_value=httpx.Response(200, json=[]))
    respx.post(f"{PAPER}/v2/orders").mock(
        return_value=httpx.Response(200, json={"id": "ord1", "status": "accepted"})
    )

    _fake_pipeline["report"] = _report("long_equity")
    runner = PaperRunner(_settings(), "NVDA", journal_path=str(tmp_path / "j.db"))
    result = await runner.run_once()

    assert result["entry"]["action"] == "order"
    orders = runner.journal.orders()
    assert orders and orders[0]["side"] == "buy"
    assert orders[0]["client_order_id"].startswith("NVDA-")


@respx.mock
async def test_run_once_max_positions_blocks(tmp_path, _fake_pipeline):
    respx.get(f"{PAPER}/v2/account").mock(
        return_value=httpx.Response(200, json={"equity": "100000", "cash": "100000"})
    )
    respx.get(f"{PAPER}/v2/positions").mock(
        return_value=httpx.Response(200, json=[{"symbol": "NVDA", "qty": "1",
                                                "avg_entry_price": "100", "mark_price": "100"}])
    )

    _fake_pipeline["report"] = _report("long_equity")
    runner = PaperRunner(_settings(), "NVDA", journal_path=str(tmp_path / "j.db"))
    result = await runner.run_once()

    assert result["entry_blocked"] == "max_positions"


def test_is_market_open_weekend_closed():
    from datetime import datetime, timezone

    sat = datetime(2026, 9, 5, 15, 30, tzinfo=timezone.utc)  # Saturday, 11:30 ET
    assert is_market_open(sat) is False
