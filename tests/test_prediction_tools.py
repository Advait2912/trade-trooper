"""Unit tests for Phase 2 prediction tools.

All tests are fully deterministic — no HTTP calls, no LLM, no Alpaca API.
Synthetic indicator dicts are constructed directly.
"""

from __future__ import annotations

from schemas.market import MarketData
from tools.prediction_tools.price_move import estimate_price_move
from tools.prediction_tools.technical import (
    apply_news_adjustment,
    calculate_technical_indicators,
)
from tools.prediction_tools.volatility_forecast import forecast_volatility

# ===========================================================================
# Helpers: synthetic Phase 1 technical dicts
# ===========================================================================

def _bullish_technical() -> dict:
    """Indicator bundle that should produce a bullish composite signal."""
    return {
        "calculate_rsi": {"rsi": 28.0, "signal": "oversold", "momentum": "down"},
        "calculate_macd": {
            "macd": 1.2, "signal_line": 0.8, "histogram": 0.4,
            "trend": "bullish", "momentum_strength": "moderate",
        },
        "calculate_adx": {
            "adx": 35.0, "plus_di": 28.0, "minus_di": 12.0,
            "trend_strength": "strong", "trend_direction": "uptrend",
        },
        "calculate_bollinger_bands": {
            "upper_band": 160.0, "middle_band": 150.0, "lower_band": 140.0,
            "current_price": 138.0, "band_position": "below_lower",
            "squeeze": False, "volatility_regime": "normal",
        },
        "calculate_obv": {
            "obv": 5000000.0, "obv_trend": "increasing",
            "volume_confirmation": "strong_confirmation",
        },
    }


def _bearish_technical() -> dict:
    return {
        "calculate_rsi": {"rsi": 76.0, "signal": "overbought", "momentum": "strong_up"},
        "calculate_macd": {
            "macd": -0.8, "signal_line": -0.3, "histogram": -0.5,
            "trend": "bearish", "momentum_strength": "moderate",
        },
        "calculate_adx": {
            "adx": 30.0, "plus_di": 11.0, "minus_di": 27.0,
            "trend_strength": "strong", "trend_direction": "downtrend",
        },
        "calculate_bollinger_bands": {
            "upper_band": 160.0, "middle_band": 150.0, "lower_band": 140.0,
            "current_price": 165.0, "band_position": "above_upper",
            "squeeze": False, "volatility_regime": "normal",
        },
        "calculate_obv": {
            "obv": -4000000.0, "obv_trend": "decreasing",
            "volume_confirmation": "strong_confirmation",
        },
    }


def _neutral_technical() -> dict:
    return {
        "calculate_rsi": {"rsi": 50.0, "signal": "neutral", "momentum": "up"},
        "calculate_macd": {
            "macd": 0.05, "signal_line": 0.04, "histogram": 0.01,
            "trend": "bullish", "momentum_strength": "weak",
        },
        "calculate_adx": {
            "adx": 12.0, "plus_di": 14.0, "minus_di": 13.0,
            "trend_strength": "no_trend", "trend_direction": "ranging",
        },
        "calculate_bollinger_bands": {
            "upper_band": 155.0, "middle_band": 150.0, "lower_band": 145.0,
            # between_middle_lower (-0.3) offsets weak bullish MACD (0.3 scaled)
            "current_price": 147.0, "band_position": "between_middle_lower",
            "squeeze": True, "volatility_regime": "low",
        },
        "calculate_obv": {
            "obv": 100.0, "obv_trend": "flat",
            "volume_confirmation": "moderate_confirmation",
        },
    }


def _minimal_market(price: float = 150.0, atr: float = 3.0) -> MarketData:
    return MarketData(price=price, atr14=atr)


def _vol_bundle(hv: float = 25.0, regime: str = "contraction") -> dict:
    return {
        "calculate_historical_volatility": {
            "historical_vol": hv,
            "vol_trend": "stable",
            "vol_percentile": 45.0,
        },
        "detect_volatility_regimes": {
            "regime": regime,
            "vol_acceleration": 0.02,
            "expected_duration_days": 5,
        },
        "analyze_mean_reversion": {
            "mean_reversion_score": 0.1,
            "mean_price": 150.0,
            "deviation_from_mean": 1.5,
            "reversion_probability": 0.55,
        },
    }


def _hist_trends(hv: float = 25.0) -> dict:
    return {"historical_vol": hv, "vol_percentile": 45.0}


# ===========================================================================
# calculate_technical_indicators tests
# ===========================================================================

class TestCalculateTechnicalIndicators:

    def test_bullish_indicators_produce_bullish_signal(self):
        result = calculate_technical_indicators(_bullish_technical())
        assert result["composite_signal"] == "bullish"
        assert result["momentum_score"] > 0.15

    def test_bearish_indicators_produce_bearish_signal(self):
        result = calculate_technical_indicators(_bearish_technical())
        assert result["composite_signal"] == "bearish"
        assert result["momentum_score"] < -0.15

    def test_neutral_indicators_produce_neutral_signal(self):
        result = calculate_technical_indicators(_neutral_technical())
        # With no_trend ADX (scale=0) and weak signals, expect neutral
        assert result["composite_signal"] == "neutral"

    def test_empty_technical_does_not_crash(self):
        result = calculate_technical_indicators({})
        # All guards return 0.0 for empty dicts → momentum = 0 → neutral
        assert result["composite_signal"] == "neutral"
        assert result["momentum_score"] == 0.0

    def test_partial_technical_does_not_crash(self):
        partial = {"calculate_rsi": {"rsi": 28.0, "signal": "oversold"}}
        result = calculate_technical_indicators(partial)
        assert "composite_signal" in result
        assert -1.0 <= result["momentum_score"] <= 1.0

    def test_momentum_score_always_clamped(self):
        # Force all signals to extreme bullish
        for _ in range(10):
            result = calculate_technical_indicators(_bullish_technical())
            assert -1.0 <= result["momentum_score"] <= 1.0

    def test_returns_all_expected_keys(self):
        result = calculate_technical_indicators(_bullish_technical())
        required = {
            "momentum_score", "composite_signal", "rsi_signal",
            "macd_signal", "adx_trend_strength", "adx_trend_direction",
            "bollinger_regime", "obv_confirmation",
        }
        assert required <= set(result.keys())

    def test_rsi_signal_propagated(self):
        result = calculate_technical_indicators(_bullish_technical())
        assert result["rsi_signal"] == "oversold"

    def test_macd_signal_propagated(self):
        result = calculate_technical_indicators(_bullish_technical())
        assert result["macd_signal"] == "bullish"


# ===========================================================================
# apply_news_adjustment tests
# ===========================================================================

class TestApplyNewsAdjustment:
    """
    Rule: news_adjustment = sign × abs(sentiment_score) × 0.20
    Maximum shift: ±0.20
    """

    def test_bullish_news_increases_momentum(self):
        _, adj = apply_news_adjustment(0.0, "bullish", 0.8)
        assert adj > 0.0

    def test_bearish_news_decreases_momentum(self):
        _, adj = apply_news_adjustment(0.0, "bearish", 0.8)
        assert adj < 0.0

    def test_neutral_news_has_no_effect(self):
        raw_adj, adj = apply_news_adjustment(0.5, "neutral", 0.9)
        assert raw_adj == 0.0
        assert adj == 0.5

    def test_uncertain_news_has_no_effect(self):
        raw_adj, adj = apply_news_adjustment(-0.3, "uncertain", 1.0)
        assert raw_adj == 0.0
        assert adj == -0.3

    def test_zero_sentiment_score_has_no_effect(self):
        raw_adj, _ = apply_news_adjustment(0.4, "bullish", 0.0)
        assert raw_adj == 0.0

    def test_adjustment_is_bounded_by_news_weight(self):
        """Max shift is ±0.20 regardless of inputs."""
        raw_adj, _ = apply_news_adjustment(0.0, "bullish", 999.0)
        assert abs(raw_adj) <= 0.20

    def test_adjusted_momentum_clamped_to_minus_one(self):
        _, adj = apply_news_adjustment(-0.95, "bearish", 1.0)
        assert adj >= -1.0

    def test_adjusted_momentum_clamped_to_plus_one(self):
        _, adj = apply_news_adjustment(0.95, "bullish", 1.0)
        assert adj <= 1.0

    def test_adjustment_magnitude_proportional_to_score(self):
        """Higher sentiment_score → larger absolute adjustment."""
        _, adj_low = apply_news_adjustment(0.0, "bullish", 0.2)
        _, adj_high = apply_news_adjustment(0.0, "bullish", 0.8)
        assert adj_high > adj_low


# ===========================================================================
# forecast_volatility tests
# ===========================================================================

class TestForecastVolatility:

    def test_normal_regime_returns_positive_iv(self):
        result = forecast_volatility(_vol_bundle(25.0, "contraction"), _hist_trends(25.0))
        assert result["iv_forecast"] > 0.0

    def test_expansion_multiplier_raises_iv(self):
        contraction = forecast_volatility(_vol_bundle(25.0, "contraction"), _hist_trends(25.0))
        expansion = forecast_volatility(_vol_bundle(25.0, "expansion"), _hist_trends(25.0))
        assert expansion["iv_forecast"] > contraction["iv_forecast"]

    def test_high_vol_produces_high_regime(self):
        result = forecast_volatility(_vol_bundle(55.0, "high_expansion"), _hist_trends(55.0))
        assert result["vol_regime"] in {"high", "very_high"}

    def test_low_vol_produces_low_regime(self):
        result = forecast_volatility(_vol_bundle(10.0, "low_contraction"), _hist_trends(10.0))
        assert result["vol_regime"] == "low"

    def test_iv_source_always_estimated(self):
        result = forecast_volatility(_vol_bundle(), _hist_trends())
        assert result["iv_source"] == "estimated"

    def test_empty_bundles_does_not_crash(self):
        result = forecast_volatility({}, {})
        assert result["iv_source"] == "estimated"
        assert result["iv_forecast"] > 0.0  # floor applied

    def test_returns_all_expected_keys(self):
        result = forecast_volatility(_vol_bundle(), _hist_trends())
        required = {
            "iv_forecast", "iv_source", "vol_regime", "vol_percentile",
            "hv_20", "hv_60", "mean_reversion_score", "vol_acceleration",
        }
        assert required <= set(result.keys())

    def test_vol_percentile_range(self):
        result = forecast_volatility(_vol_bundle(25.0, "contraction"), _hist_trends(25.0))
        assert 0.0 <= result["vol_percentile"] <= 100.0


# ===========================================================================
# estimate_price_move tests
# ===========================================================================

class TestEstimatePriceMove:

    def test_normal_inputs_produce_valid_forecast(self):
        md = _minimal_market(150.0, 3.0)
        result = estimate_price_move(md, 0.3, "strong", "normal", 0.05)
        assert result["status"] == "ok"
        assert result["price_forecast"] > 0.0

    def test_forecast_low_below_current_price(self):
        md = _minimal_market(150.0, 3.0)
        result = estimate_price_move(md, 0.0, "moderate", "normal", 0.0)
        assert result["price_forecast_low"] < md.price

    def test_forecast_high_above_current_price(self):
        md = _minimal_market(150.0, 3.0)
        result = estimate_price_move(md, 0.0, "moderate", "normal", 0.0)
        assert result["price_forecast_high"] > md.price

    def test_bullish_momentum_skews_forecast_up(self):
        md = _minimal_market(150.0, 3.0)
        result = estimate_price_move(md, 0.8, "strong", "normal", 0.0)
        assert result["price_forecast"] > md.price

    def test_bearish_momentum_skews_forecast_down(self):
        md = _minimal_market(150.0, 3.0)
        result = estimate_price_move(md, -0.8, "strong", "normal", 0.0)
        assert result["price_forecast"] < md.price

    def test_zero_price_returns_insufficient_data(self):
        md = _minimal_market(0.0)
        result = estimate_price_move(md, 0.0, "weak", "normal", 0.0)
        assert result["status"] == "insufficient_data"
        assert result["confidence"] == 0.0
        assert len(result["errors"]) > 0

    def test_negative_price_returns_insufficient_data(self):
        md = _minimal_market(-10.0)
        result = estimate_price_move(md, 0.0, "weak", "normal", 0.0)
        assert result["status"] == "insufficient_data"

    def test_confidence_clamped_to_min(self):
        # High vol + weak trend + no signal → should hit floor
        md = _minimal_market(150.0, 3.0)
        result = estimate_price_move(md, 0.0, "no_trend", "very_high", 0.5)
        assert result["confidence"] >= 0.10

    def test_confidence_clamped_to_max(self):
        md = _minimal_market(150.0, 3.0)
        result = estimate_price_move(md, 0.9, "very_strong", "normal", 0.0)
        assert result["confidence"] <= 0.90

    def test_missing_atr_uses_fallback(self):
        md = _minimal_market(150.0, atr=0.0)
        result = estimate_price_move(md, 0.5, "moderate", "normal", 0.0)
        # Should not crash; uses 2% fallback ATR
        assert result["status"] == "ok"
        assert "ATR unavailable" in " ".join(result["errors"])

    def test_expected_move_pct_positive(self):
        md = _minimal_market(150.0, 3.0)
        result = estimate_price_move(md, 0.0, "moderate", "normal", 0.0)
        assert result["expected_move_pct"] > 0.0

    def test_horizon_days_reflected_in_output(self):
        md = _minimal_market(150.0, 3.0)
        result = estimate_price_move(md, 0.0, "moderate", "normal", 0.0, horizon_days=10)
        assert result["forecast_horizon_days"] == 10
