"""Tests for the FinBERT score mapping and the news-sentiment cache/backtest.

No GPU, torch, or network required — mapping + cache logic are pure/in-memory.
"""

from __future__ import annotations

import pytest

from schemas.historical import HistoricalAgentResult
from schemas.market import MarketData
from schemas.news import NewsArticle, NewsCollectionResult
from schemas.prediction import Phase1Bundle
from tools.finbert_sentiment import map_predictions
from trading.backtest import _predict
from trading.news_cache import NewsCache, aggregate_daily, decision_date

# ===========================================================================
# FinBERT probability -> score/label mapping
# ===========================================================================

_FINBERT_ID2LABEL = {0: "positive", 1: "negative", 2: "neutral"}


class TestMapPredictions:

    def test_positive(self):
        result = map_predictions([[0.8, 0.1, 0.1]], _FINBERT_ID2LABEL)
        assert result[0]["label"] == "positive"
        assert result[0]["score"] == pytest.approx(0.7)

    def test_negative(self):
        result = map_predictions([[0.1, 0.7, 0.2]], _FINBERT_ID2LABEL)
        assert result[0]["label"] == "negative"
        assert result[0]["score"] == pytest.approx(-0.6)

    def test_neutral(self):
        result = map_predictions([[0.1, 0.1, 0.8]], _FINBERT_ID2LABEL)
        assert result[0]["label"] == "neutral"
        assert result[0]["score"] == pytest.approx(0.0)

    def test_handles_different_label_order(self):
        # A hypothetical model with a shuffled class order.
        id2label = {0: "neutral", 1: "positive", 2: "negative"}
        result = map_predictions([[0.1, 0.7, 0.2]], id2label)
        assert result[0]["label"] == "positive"
        assert result[0]["score"] == pytest.approx(0.5)


# ===========================================================================
# decision_date (24h-lookback, no look-ahead)
# ===========================================================================

class TestDecisionDate:

    def test_intraday_maps_to_same_day(self):
        assert decision_date("2026-09-01T12:00:00Z") == "2026-09-01"

    def test_before_close_maps_to_same_day(self):
        assert decision_date("2026-09-01T20:00:00Z") == "2026-09-01"

    def test_after_close_rolls_to_next_day(self):
        # 22:00 UTC is after the 21:00 close -> next trading day.
        assert decision_date("2026-09-01T22:00:00Z") == "2026-09-02"

    def test_invalid_returns_empty(self):
        assert decision_date("") == ""
        assert decision_date("not-a-date") == ""


# ===========================================================================
# Daily aggregation
# ===========================================================================

class TestAggregateDaily:

    def test_mean_and_label(self):
        daily = aggregate_daily([("2026-09-01", 0.5), ("2026-09-01", 0.3)])
        assert daily["2026-09-01"]["score"] == pytest.approx(0.4)
        assert daily["2026-09-01"]["label"] == "bullish"
        assert daily["2026-09-01"]["article_count"] == 2

    def test_empty(self):
        assert aggregate_daily([]) == {}


# ===========================================================================
# NewsCache round-trip
# ===========================================================================

class TestNewsCache:

    def test_round_trip(self, tmp_path):
        cache = NewsCache(tmp_path / "cache.db")
        try:
            cache.add_articles("NVDA", [
                NewsArticle(id=1, headline="Nvidia beats", created_at="2026-09-01T12:00:00Z"),
                NewsArticle(id=2, headline="Nvidia cuts", created_at="2026-09-01T13:00:00Z"),
            ])
            unscored = cache.unscored()
            assert len(unscored) == 2

            cache.set_sentiment(1, 0.8, "positive")
            cache.set_sentiment(2, -0.6, "negative")
            assert cache.unscored() == []  # idempotent

            dated = cache.dated_scores("NVDA")
            assert ("2026-09-01", 0.8) in dated
            assert ("2026-09-01", -0.6) in dated

            daily = aggregate_daily(dated)
            cache.set_daily("NVDA", daily)
            loaded = cache.load_daily_map("NVDA")
            assert loaded["2026-09-01"]["score"] == pytest.approx(0.1)
        finally:
            cache.close()


# ===========================================================================
# Backtest news-aware prediction
# ===========================================================================

def _bullish_technical() -> dict:
    return {
        "calculate_rsi": {"rsi": 28.0, "signal": "oversold"},
        "calculate_macd": {"macd": 1.0, "trend": "bullish", "momentum_strength": "strong"},
        "calculate_adx": {"adx": 30.0, "trend_strength": "strong", "trend_direction": "uptrend"},
        "calculate_bollinger_bands": {"band_position": "below_lower", "volatility_regime": "normal"},
        "calculate_obv": {"obv_trend": "increasing", "volume_confirmation": "strong_confirmation"},
    }


def _bundle() -> Phase1Bundle:
    return Phase1Bundle(
        news=NewsCollectionResult(ticker="NVDA", sentiment_score=0.0),
        market=MarketData(price=150.0, atr14=3.0),
        historical=HistoricalAgentResult(
            symbol="NVDA", technical=_bullish_technical(), volatility={}, historical_trends={}
        ),
    )


class TestBacktestNewsAware:

    def test_bearish_news_reduces_momentum(self):
        base = _predict(_bundle())
        bearish = _predict(_bundle(), day_sentiment={"score": -0.8, "label": "bearish"})
        assert base.news_sentiment == "uncertain"
        assert bearish.news_sentiment == "bearish"
        assert bearish.news_sentiment_score == pytest.approx(-0.8)
        assert bearish.adjusted_momentum < base.adjusted_momentum

    def test_neutral_news_no_change(self):
        base = _predict(_bundle())
        neutral = _predict(_bundle(), day_sentiment={"score": 0.0, "label": "neutral"})
        assert neutral.adjusted_momentum == pytest.approx(base.adjusted_momentum)
