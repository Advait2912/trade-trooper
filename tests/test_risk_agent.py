"""Integration tests for Phase 3 RiskAgent.

All tests are offline — the option-chain endpoint is mocked with respx; no real
credentials, HTTP, or LLM are used.
"""

from __future__ import annotations

import pytest
import respx

from agents.risk_agent import RiskAgent
from schemas.historical import HistoricalAgentResult
from schemas.market import MarketData
from schemas.news import NewsCollectionResult
from schemas.prediction import Phase1Bundle, PredictionResult
from schemas.risk import RiskResult
from tests.conftest import mock_options_chain, mock_options_unavailable

# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------

def _make_historical() -> HistoricalAgentResult:
    return HistoricalAgentResult(
        symbol="NVDA",
        status="ok",
        bars_count=100,
        levels={
            "support_levels": [{"level": 145.0, "touches": 3, "strength": "strong"}],
            "resistance_levels": [{"level": 160.0, "touches": 2, "strength": "moderate"}],
        },
        risk={
            "calculate_drawdown": {"risk_level": "moderate", "max_drawdown": 15.0},
            "analyze_gaps": {"gap_frequency": "occasional", "avg_gap_size": 1.2},
            "calculate_value_at_risk": {"var": 2.5, "cvar": 3.5},
        },
    )


def _make_market(price: float = 150.0, atr: float = 3.2) -> MarketData:
    return MarketData(price=price, atr14=atr)


def _make_prediction() -> PredictionResult:
    return PredictionResult(
        price_forecast=152.0,
        price_forecast_high=155.0,
        price_forecast_low=140.0,
        expected_move_pct=0.03,
        forecast_horizon_days=5,
        iv_forecast=30.0,
        vol_regime="normal",
        vol_percentile=50.0,
        confidence=0.6,
        composite_signal="bullish",
    )


def _bundle(price: float = 150.0, atr: float = 3.2) -> Phase1Bundle:
    return Phase1Bundle(
        news=NewsCollectionResult(ticker="NVDA"),
        market=_make_market(price, atr),
        historical=_make_historical(),
    )


# ===========================================================================
# RiskAgent integration tests
# ===========================================================================

@pytest.mark.asyncio
class TestRiskAgent:

    @respx.mock
    async def test_full_run_returns_typed_result(self, settings):
        mock_options_chain()
        agent = RiskAgent(settings)
        result = await agent.run(_bundle(), _make_prediction())
        assert isinstance(result, RiskResult)

    @respx.mock
    async def test_status_ok_with_chain(self, settings):
        mock_options_chain()
        agent = RiskAgent(settings)
        result = await agent.run(_bundle(), _make_prediction())
        assert result.status == "ok"
        assert result.greeks_source == "alpaca_option_chain"
        assert result.iv_source == "market_implied"

    @respx.mock
    async def test_fallback_when_options_unavailable(self, settings):
        mock_options_unavailable(403)
        agent = RiskAgent(settings)
        result = await agent.run(_bundle(), _make_prediction())
        assert result.greeks_source == "black_scholes_estimated"
        assert result.iv_source == "estimated"
        assert result.status in {"ok", "partial"}

    @respx.mock
    async def test_insufficient_data_zero_price(self, settings):
        mock_options_chain()
        agent = RiskAgent(settings)
        result = await agent.run(_bundle(price=0.0), _make_prediction())
        assert result.status == "insufficient_data"
        assert len(result.errors) > 0

    @respx.mock
    async def test_stop_below_entry(self, settings):
        mock_options_chain()
        agent = RiskAgent(settings)
        result = await agent.run(_bundle(), _make_prediction())
        assert 0.0 < result.stop_loss_level < 150.0

    @respx.mock
    async def test_position_positive(self, settings):
        mock_options_chain()
        agent = RiskAgent(settings)
        result = await agent.run(_bundle(), _make_prediction())
        assert result.position_recommendation.equity.shares > 0
        assert result.capital_at_risk_pct > 0

    @respx.mock
    async def test_risk_level_valid(self, settings):
        mock_options_chain()
        agent = RiskAgent(settings)
        result = await agent.run(_bundle(), _make_prediction())
        assert result.risk_level in {"low", "moderate", "high", "very_high"}
        assert 0.0 <= result.risk_score <= 100.0

    @respx.mock
    async def test_risk_reward_positive(self, settings):
        mock_options_chain()
        agent = RiskAgent(settings)
        result = await agent.run(_bundle(), _make_prediction())
        assert result.risk_reward_ratio > 0.0

    @respx.mock
    async def test_accepts_raw_phase1_dict(self, settings):
        mock_options_chain()
        agent = RiskAgent(settings)
        raw = {
            "news": NewsCollectionResult(ticker="NVDA"),
            "market": _make_market(),
            "historical": _make_historical(),
        }
        result = await agent.run(raw, _make_prediction())
        assert isinstance(result, RiskResult)
        assert result.status == "ok"
