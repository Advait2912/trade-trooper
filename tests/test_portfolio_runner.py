import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from schemas.decision import DecisionResult
from schemas.pipeline import FinalReport
from trading.journal import TradeJournal
from trading.runner import PortfolioRunner
from utils.config import Settings
from utils.decision_logger import DecisionLogger
from weights_db import resolve_config


def test_resolve_config_clones_and_applies():
    base_settings = Settings(min_confidence=0.35, min_risk_reward=1.0)
    db = {
        "default": {"min_confidence": 0.50},
        "Technology": {"min_confidence": 0.42, "trade_horizon_days": 3, "equity_only": True},
    }
    s_nvda, t_nvda = resolve_config("NVDA", db, base_settings)
    assert s_nvda.min_confidence == 0.42
    assert s_nvda.trade_horizon_days == 3
    assert t_nvda.equity_only is True
    # Base settings untouched
    assert base_settings.min_confidence == 0.35

    s_unknown, t_unknown = resolve_config("XYZ", db, base_settings)
    assert s_unknown.min_confidence == 0.50


@pytest.mark.asyncio
async def test_decision_logger_roundtrip(tmp_path: Path):
    logger = DecisionLogger(tmp_path)
    report = FinalReport(
        ticker="AAPL",
        timestamp="2026-09-04T03:00:00Z",
        decision=DecisionResult(trade_decision="long_equity", confidence_score=0.75, rationale="Bullish"),
    )
    rec = await logger.log("AAPL", report, benchmark={"total_ms": 150.0, "industry": "Technology"})
    assert rec["ticker"] == "AAPL"
    assert rec["decision"] == "long_equity"
    assert rec["benchmark"]["total_ms"] == 150.0
    await logger.close()

    logs = list(tmp_path.glob("decisions_*.jsonl"))
    assert len(logs) == 1
    content = json.loads(logs[0].read_text().strip())
    assert content["ticker"] == "AAPL"
    assert content["decision"] == "long_equity"


def test_journal_checkpoints_roundtrip(tmp_path: Path):
    j_path = tmp_path / "test_j.db"
    j = TradeJournal(j_path)
    cp_dir = tmp_path / "checkpoints"

    j.record_cycle_checkpoint(
        ticker="NVDA",
        cycle_ts="2026-09-04T03:00:00Z",
        industry="Technology",
        weights_hash="abc1234",
        market_snapshot={"price": 120.0},
        news_snapshot={"headline": "Tech surge"},
        decision="long_equity",
        trade_placed=True,
        entry_price=120.5,
        stop_loss=115.0,
        take_profit=130.0,
        checkpoint_dir=cp_dir,
    )

    cp_file = cp_dir / "nvda_checkpoints.jsonl"
    assert cp_file.exists()
    row = json.loads(cp_file.read_text().strip())
    assert row["decision"] == "long_equity"
    assert row["outcome"] is None

    # Update outcome when closed
    j.update_checkpoint_outcome(
        ticker="NVDA",
        cycle_ts="2026-09-04T03:00:00Z",
        exit_price=128.0,
        pnl=750.0,
        pnl_pct=0.062,
        exit_reason="close_target",
        hold_days=2,
        checkpoint_dir=cp_dir,
    )
    row_updated = json.loads(cp_file.read_text().strip())
    assert row_updated["outcome"]["exit_reason"] == "close_target"
    assert row_updated["outcome"]["pnl"] == 750.0
    j.close()
