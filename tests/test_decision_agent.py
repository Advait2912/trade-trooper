"""Integration tests for Phase 4 DecisionAgent.

Fully offline: builds typed Phase 1 bundle, PredictionResult and RiskResult
in memory (no HTTP, no LLM).  Reuses the fixture-builders style from
test_risk_agent.py.
"""

from __future__ import annotations

import pytest

from agents.decision_agent import DecisionAgent
from schemas.historical import HistoricalAgentResult
from schemas.market import MarketData
from schemas.news import NewsCollectionResult
from schemas.prediction import Phase1Bundle, PredictionResult
from schemas.risk import (
    EquityPosition,
    OptionPosition,
    PositionRecommendation,
    RiskResult,
)
from utils.config import Settings


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------
def _settings(min_confidence=0.35, min_risk_reward=1.0) -> Settings:
    return Settings(
        alpaca_api_key="k",
        alpaca_api_secret="s",
        account_capital=100_000.0,
        risk_per_trade_pct=0.01,
        max_position_pct=0.05,
        min_risk_reward=min_risk_reward,
        min_confidence=min_confidence,
    )


def _historical(bias: str) -> HistoricalAgentResult:
    summary = {
        "overall_signal": "strong_bullish" if bias == "bullish" else "strong_bearish",
    }
    trend_class = "bullish" if bias == "bullish" else "bearish"
    return HistoricalAgentResult(
        symbol="NVDA",
        status="ok",
        bars_count=100,
        summary=summary,
        historical_trends={"trend_class": trend_class, "trend": "strong_up" if bias == "bullish" else "strong_down"},
        risk={
            "calculate_drawdown": {"risk_level": "low", "max_drawdown": 5.0},
            "analyze_gaps": {"gap_frequency": "rare", "avg_gap_size": 0.4},
            "calculate_value_at_risk": {"var": 1.0, "cvar": 1.5},
        },
    )


def _market(price: float = 150.0, atr: float = 3.0, trend="bullish") -> MarketData:
    return MarketData(price=price, atr14=atr, sma20=140.0, sma50=130.0, rsi14=60.0)


def _news(score: float) -> NewsCollectionResult:
    return NewsCollectionResult(ticker="NVDA", sentiment_score=score)


def _prediction(signal="bullish", confidence=0.7, horizon=5) -> PredictionResult:
    return PredictionResult(
        composite_signal=signal,
        confidence=confidence,
        forecast_horizon_days=horizon,
        price_forecast=152.0,
        price_forecast_high=158.0,
        price_forecast_low=142.0,
        iv_forecast=30.0,
    )


def _bundle(bias="bullish", price=150.0, news_score=0.0) -> Phase1Bundle:
    return Phase1Bundle(
        news=_news(news_score),
        market=_market(price),
        historical=_historical(bias),
    )


def _risk(
    stop=145.0,
    target=160.0,
    r_r=3.0,
    score=30.0,
    level="moderate",
    contracts=2.0,
    equity_shares=100.0,
    call_premium=8.0,
    put_premium=6.0,
    put_delta=-0.4,
    greeks_source="alpaca_option_chain",
) -> RiskResult:
    return RiskResult(
        status="ok",
        risk_score=score,
        risk_level=level,
        stop_loss_level=stop,
        take_profit_level=target,
        risk_reward_ratio=r_r,
        greeks_source=greeks_source,
        iv_source="market_implied" if greeks_source == "alpaca_option_chain" else "estimated",
        position_recommendation=PositionRecommendation(
            equity=EquityPosition(shares=equity_shares, dollar_value=equity_shares * 150.0),
            option=OptionPosition(contracts=contracts, premium_risk=contracts * 100 * call_premium,
                                  delta_exposure=contracts * 100 * 0.5),
        ),
        risk_metrics={
            "calculate_greeks": {
                "greeks_source": greeks_source,
                "greeks": {
                    "call": {"delta": 0.5, "gamma": 0.01, "theta": -0.02, "vega": 0.1, "rho": 0.05},
                    "put": {"delta": put_delta, "gamma": 0.01, "theta": -0.02, "vega": 0.1, "rho": 0.05},
                },
                "call_premium": call_premium,
                "put_premium": put_premium,
                "spread_pct": 0.02,
            },
            "calculate_position_size": {"equity_shares": equity_shares,
                                        "option_contracts": contracts},
        },
    )


# ===========================================================================
# DecisionAgent integration tests
# ===========================================================================
@pytest.mark.asyncio
class TestDecisionAgent:

    async def test_bullish_full_alignment_goes_long_call(self):
        agent = DecisionAgent(_settings())
        result = await agent.run(_bundle("bullish", news_score=0.6),
                                 _prediction("bullish", 0.8), _risk())
        assert result.status == "ok"
        assert result.trade_decision == "long_call"
        assert result.composite_bias == "bullish"
        assert result.confidence_score > 0.5
        assert result.instrument == "option"
        assert result.option_type == "call"
        assert result.option_contracts == 2.0
        assert result.risk_reward_ratio > 0
        assert result.summary

    async def test_bearish_selects_long_put(self):
        agent = DecisionAgent(_settings())
        result = await agent.run(_bundle("bearish", news_score=-0.6),
                                 _prediction("bearish", 0.75), _risk())
        assert result.trade_decision == "long_put"
        assert result.composite_bias == "bearish"
        assert result.option_type == "put"
        assert result.instrument == "option"
        # put stop is above entry, target below entry
        assert result.stop_loss > result.entry_price
        assert result.take_profit < result.entry_price

    async def test_confidence_gate_holds(self):
        agent = DecisionAgent(_settings(min_confidence=0.6))
        result = await agent.run(_bundle("bullish", news_score=0.6),
                                 _prediction("bullish", 0.2), _risk())
        assert result.trade_decision == "hold"
        assert result.confidence_score < 0.6

    async def test_risk_reward_gate_holds(self):
        agent = DecisionAgent(_settings(min_risk_reward=2.0))
        result = await agent.run(_bundle("bullish", news_score=0.6),
                                 _prediction("bullish", 0.8), _risk(r_r=1.0))
        assert result.trade_decision == "hold"

    async def test_very_high_risk_avoids(self):
        agent = DecisionAgent(_settings())
        result = await agent.run(_bundle("bullish", news_score=0.6),
                                 _prediction("bullish", 0.8), _risk(score=90.0, level="very_high"))
        assert result.trade_decision == "avoid"

    async def test_beuristic_no_price_is_hold(self):
        agent = DecisionAgent(_settings())
        bundle = _bundle("bullish")
        bundle.market = MarketData(price=0.0)
        result = await agent.run(bundle, _prediction("bullish", 0.8), _risk())
        assert result.status == "insufficient_data"
        assert result.trade_decision == "hold"

    async def test_neutral_bias_holds(self):
        agent = DecisionAgent(_settings())
        bundle = _bundle("bullish", news_score=0.0)
        bundle.historical = _historical("bullish")
        # Force neutrality via low-confidence + neutral prediction + neutral news.
        result = await agent.run(
            _bundle("bullish", news_score=0.0),
            _prediction("neutral", 0.2), _risk(),
        )
        assert result.trade_decision == "hold"

    async def test_raw_dict_phase1_accepted(self):
        agent = DecisionAgent(_settings())
        bundle = _bundle("bullish", news_score=0.6)
        raw = {
            "news": bundle.news,
            "market": bundle.market,
            "historical": bundle.historical,
        }
        result = await agent.run(raw, _prediction("bullish", 0.8), _risk())
        assert result.trade_decision == "long_call"

    async def test_decision_metrics_auditable(self):
        agent = DecisionAgent(_settings())
        result = await agent.run(_bundle("bullish", news_score=0.6),
                                 _prediction("bullish", 0.8), _risk())
        assert "synthesize_signals" in result.decision_metrics
        assert "rank_opportunities" in result.decision_metrics
        assert result.opportunities, "Expected at least one ranked opportunity"
        assert result.opportunities[0].rank == 1
