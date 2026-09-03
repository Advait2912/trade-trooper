"""Integration tests for Phase 2 PredictionAgent.

All tests are fully offline — no HTTP, no LLM.
"""

from __future__ import annotations

import pytest

from agents.prediction_agent import PredictionAgent
from schemas.historical import HistoricalAgentResult, VolatilityPoint
from schemas.market import MarketData
from schemas.news import InitialAnalysis, NewsArticle, NewsCollectionResult
from schemas.prediction import Phase1Bundle, PredictionResult

# ---------------------------------------------------------------------------
# Helpers: build a realistic Phase1Bundle without any API
# ---------------------------------------------------------------------------

def _make_technical() -> dict:
    return {
        "calculate_rsi": {"rsi": 42.0, "signal": "neutral", "momentum": "down"},
        "calculate_macd": {
            "macd": 0.5, "signal_line": 0.3, "histogram": 0.2,
            "trend": "bullish", "momentum_strength": "moderate",
        },
        "calculate_adx": {
            "adx": 26.0, "plus_di": 22.0, "minus_di": 15.0,
            "trend_strength": "moderate", "trend_direction": "uptrend",
        },
        "calculate_bollinger_bands": {
            "upper_band": 160.0, "middle_band": 150.0, "lower_band": 140.0,
            "current_price": 148.0, "band_position": "between_middle_lower",
            "squeeze": False, "volatility_regime": "normal",
        },
        "calculate_obv": {
            "obv": 2000000.0, "obv_trend": "increasing",
            "volume_confirmation": "strong_confirmation",
        },
    }


def _make_volatility() -> dict:
    return {
        "calculate_historical_volatility": {
            "historical_vol": 28.5,
            "vol_trend": "stable",
            "vol_percentile": 52.0,
        },
        "detect_volatility_regimes": {
            "regime": "expansion",
            "vol_acceleration": 0.05,
            "expected_duration_days": 8,
        },
        "analyze_mean_reversion": {
            "mean_reversion_score": 0.12,
            "mean_price": 149.0,
            "deviation_from_mean": 0.8,
            "reversion_probability": 0.55,
        },
    }


def _make_historical() -> HistoricalAgentResult:
    vp = VolatilityPoint(date="2026-01-01", realized_vol=25.0, rolling_vol_20d=28.5, rolling_vol_60d=27.0)
    return HistoricalAgentResult(
        symbol="NVDA",
        status="ok",
        bars_count=100,
        technical=_make_technical(),
        volatility=_make_volatility(),
        historical_trends={
            "historical_vol": 28.5,
            "vol_percentile": 52.0,
            "mean_reversion_score": 0.12,
        },
        volatility_history=[vp] * 30,
        closes=[100.0 + i * 0.5 for i in range(100)],
        highs=[101.0 + i * 0.5 for i in range(100)],
        lows=[99.0 + i * 0.5 for i in range(100)],
        volumes=[1_000_000] * 100,
    )


def _make_market() -> MarketData:
    return MarketData(
        price=148.50,
        atr14=3.2,
        rsi14=42.0,
        sma20=145.0,
        sma50=140.0,
        return_1d=0.008,
        return_5d=0.02,
        volatility=28.5,
    )


def _make_bullish_news() -> NewsCollectionResult:
    article = NewsArticle(
        id=1, headline="Nvidia signs major GPU supply deal", source="reuters"
    )
    analysis = InitialAnalysis(
        ticker="NVDA",
        event="Major supply agreement",
        relevance=0.85,
        materiality="high",  # type: ignore[arg-type]
        sentiment="bullish",  # type: ignore[arg-type]
        evidence_quality="medium",  # type: ignore[arg-type]
    )
    return NewsCollectionResult(
        ticker="NVDA",
        articles=[article],
        analyses=[analysis],
        primary_article=article,
        primary_analysis=analysis,
        sentiment_score=0.75,
    )


def _make_bearish_news() -> NewsCollectionResult:
    article = NewsArticle(
        id=2, headline="Nvidia loses key customer to competitor", source="bloomberg"
    )
    analysis = InitialAnalysis(
        ticker="NVDA",
        event="Customer loss",
        relevance=0.80,
        materiality="high",  # type: ignore[arg-type]
        sentiment="bearish",  # type: ignore[arg-type]
        evidence_quality="medium",  # type: ignore[arg-type]
    )
    return NewsCollectionResult(
        ticker="NVDA",
        articles=[article],
        analyses=[analysis],
        primary_article=article,
        primary_analysis=analysis,
        sentiment_score=-0.70,
    )


def _make_neutral_news() -> NewsCollectionResult:
    return NewsCollectionResult(ticker="NVDA", sentiment_score=0.0)


def _bundle(
    historical: HistoricalAgentResult | None = None,
    market: MarketData | None = None,
    news: NewsCollectionResult | None = None,
) -> Phase1Bundle:
    return Phase1Bundle(
        news=news or _make_neutral_news(),
        market=market or _make_market(),
        historical=historical or _make_historical(),
    )


# ===========================================================================
# PredictionAgent integration tests
# ===========================================================================

@pytest.mark.asyncio
class TestPredictionAgent:

    async def test_full_run_returns_typed_result(self):
        agent = PredictionAgent()
        result = await agent.run(_bundle())
        assert isinstance(result, PredictionResult)

    async def test_status_ok_with_valid_data(self):
        agent = PredictionAgent()
        result = await agent.run(_bundle())
        assert result.status == "ok"

    async def test_composite_signal_valid(self):
        agent = PredictionAgent()
        result = await agent.run(_bundle())
        assert result.composite_signal in {"bullish", "bearish", "neutral"}

    async def test_confidence_in_range(self):
        agent = PredictionAgent()
        result = await agent.run(_bundle())
        assert 0.0 <= result.confidence <= 1.0

    async def test_price_forecast_positive(self):
        agent = PredictionAgent()
        result = await agent.run(_bundle())
        assert result.price_forecast > 0.0

    async def test_iv_forecast_positive(self):
        agent = PredictionAgent()
        result = await agent.run(_bundle())
        assert result.iv_forecast > 0.0

    async def test_iv_source_is_estimated(self):
        agent = PredictionAgent()
        result = await agent.run(_bundle())
        assert result.iv_source == "estimated"

    async def test_bullish_news_increases_adjusted_momentum(self):
        agent = PredictionAgent()
        neutral = await agent.run(_bundle(news=_make_neutral_news()))
        bullish = await agent.run(_bundle(news=_make_bullish_news()))
        assert bullish.adjusted_momentum >= neutral.adjusted_momentum

    async def test_bearish_news_decreases_adjusted_momentum(self):
        agent = PredictionAgent()
        neutral = await agent.run(_bundle(news=_make_neutral_news()))
        bearish = await agent.run(_bundle(news=_make_bearish_news()))
        assert bearish.adjusted_momentum <= neutral.adjusted_momentum

    async def test_news_adjustment_bounded(self):
        agent = PredictionAgent()
        result = await agent.run(_bundle(news=_make_bullish_news()))
        assert abs(result.news_adjustment) <= 0.20

    async def test_accepts_raw_phase1_dict(self):
        """Pipeline passes a raw dict; agent should accept it."""
        agent = PredictionAgent()
        raw = {
            "news": _make_neutral_news(),
            "market": _make_market(),
            "historical": _make_historical(),
        }
        result = await agent.run(raw)
        assert isinstance(result, PredictionResult)
        assert result.status == "ok"

    async def test_empty_historical_graceful(self):
        """Missing price data should produce an error or insufficient_data, not crash."""
        agent = PredictionAgent()
        empty_hist = HistoricalAgentResult(symbol="NVDA", status="partial")
        result = await agent.run(_bundle(historical=empty_hist))
        # Must not raise; status can be ok (with zero values) or error
        assert result.status in {"ok", "error", "insufficient_data"}

    async def test_zero_price_graceful(self):
        agent = PredictionAgent()
        zero_market = MarketData(price=0.0)
        result = await agent.run(_bundle(market=zero_market))
        assert result.status in {"ok", "error", "insufficient_data"}

    async def test_forecast_low_le_forecast_high(self):
        agent = PredictionAgent()
        result = await agent.run(_bundle())
        assert result.price_forecast_low <= result.price_forecast_high

    async def test_momentum_score_clamped(self):
        agent = PredictionAgent()
        result = await agent.run(_bundle())
        assert -1.0 <= result.momentum_score <= 1.0
        assert -1.0 <= result.adjusted_momentum <= 1.0

    async def test_vol_regime_valid_value(self):
        agent = PredictionAgent()
        result = await agent.run(_bundle())
        assert result.vol_regime in {"low", "normal", "high", "very_high"}
