"""Decision tool unit tests (signal synthesis + opportunity ranking)."""

import pytest

from tools.decision_tools.ranking import (
    calculate_opportunity_score,
    rank_opportunities,
)
from tools.decision_tools.signals import synthesize_signals


# ---------------------------------------------------------------------------
# synthesize_signals
# ---------------------------------------------------------------------------
class _News:
    def __init__(self, sentiment_score=0.0):
        self.sentiment_score = sentiment_score


class _Historical:
    def __init__(self, overall="neutral", trend_class="neutral", trend="ranging"):
        self.summary = {"overall_signal": overall}
        self.historical_trends = {"trend_class": trend_class, "trend": trend}


class _Market:
    def __init__(self, trend="neutral", return_5d=0.0):
        class _Trend:
            value = trend

        self.trend = _Trend()
        self.return_5d = return_5d


class _Prediction:
    def __init__(self, composite_signal="neutral", confidence=0.5):
        self.composite_signal = composite_signal
        self.confidence = confidence


class _Bundle:
    def __init__(self, news, historical, market):
        self.news = news
        self.historical = historical
        self.market = market


def _all_bullish_bundle():
    return _Bundle(
        news=_News(0.8),
        historical=_Historical(overall="strong_bullish", trend_class="bullish", trend="uptrend"),
        market=_Market(trend="bullish", return_5d=0.05),
    )


def test_synthesize_all_bullish():
    sig = synthesize_signals(_all_bullish_bundle(), _Prediction("bullish", 0.8))
    assert sig["composite_bias"] == "bullish"
    assert sig["agreement_score"] > 0.4
    assert "news_sentiment" in sig["signal_votes"]
    assert "prediction_signal" in sig["signal_votes"]


def test_synthesize_mixed_bearish():
    bundle = _Bundle(
        news=_News(-0.9),
        historical=_Historical(overall="strong_bearish", trend_class="bearish", trend="downtrend"),
        market=_Market(trend="bearish", return_5d=-0.04),
    )
    sig = synthesize_signals(bundle, _Prediction("bearish", 0.7))
    assert sig["composite_bias"] == "bearish"
    assert sig["agreement_score"] > 0.3


def test_synthesize_divergence_detected():
    # Technical bullish but news bearish -> divergence recorded.
    bundle = _Bundle(
        news=_News(-0.8),
        historical=_Historical(overall="strong_bullish", trend_class="bullish", trend="uptrend"),
        market=_Market(trend="bullish", return_5d=0.04),
    )
    sig = synthesize_signals(bundle, _Prediction("bullish", 0.7))
    assert sig["composite_bias"] in ("bullish", "bearish", "neutral")
    assert any("news_sentiment" in d for d in sig["divergences"])


def test_synthesize_neutral_when_all_conflicting():
    bundle = _Bundle(
        news=_News(0.0),
        historical=_Historical(overall="neutral", trend_class="neutral", trend="ranging"),
        market=_Market(trend="neutral", return_5d=0.0),
    )
    sig = synthesize_signals(bundle, _Prediction("neutral", 0.5))
    assert sig["composite_bias"] in ("neutral",)
    assert sig["agreement_score"] < 0.35


# ---------------------------------------------------------------------------
# calculate_opportunity_score
# ---------------------------------------------------------------------------
def test_score_components_sum_within_100():
    score = calculate_opportunity_score(
        {"direction_matches": True, "confidence": 0.8, "r_r": 3.0},
        composite_bias="bullish", agreement_score=0.9, risk_score=20.0,
        spread_pct=0.02, greeks_source="alpaca_option_chain",
    )
    assert 0 <= score["score"] <= 100
    total = sum(score["components"].values())
    assert total == pytest.approx(score["score"], abs=0.1)


def test_score_penalises_direction_mismatch():
    matched = calculate_opportunity_score(
        {"direction_matches": True, "confidence": 0.8, "r_r": 3.0},
        composite_bias="bearish", agreement_score=0.9, risk_score=20.0,
        spread_pct=0.02, greeks_source="alpaca_option_chain",
    )
    mismatched = calculate_opportunity_score(
        {"direction_matches": False, "confidence": 0.8, "r_r": 3.0},
        composite_bias="bearish", agreement_score=0.9, risk_score=20.0,
        spread_pct=0.02, greeks_source="alpaca_option_chain",
    )
    assert mismatched["score"] < matched["score"]
    # signal component is the one reduced
    assert mismatched["components"]["signal"] < matched["components"]["signal"]


def test_score_execution_quality_penalty():
    live = calculate_opportunity_score(
        {"direction_matches": True, "confidence": 0.8, "r_r": 3.0},
        agreement_score=0.9, risk_score=20.0, spread_pct=0.01,
        greeks_source="alpaca_option_chain",
    )
    est = calculate_opportunity_score(
        {"direction_matches": True, "confidence": 0.8, "r_r": 3.0},
        agreement_score=0.9, risk_score=20.0, spread_pct=0.01,
        greeks_source="black_scholes_estimated",
    )
    assert est["components"]["execution"] < live["components"]["execution"]


# ---------------------------------------------------------------------------
# rank_opportunities
# ---------------------------------------------------------------------------
def _candidate(opt_type, conf=0.7, r_r=2.5, matches=True, premium=1200.0):
    return {
        "id": f"{opt_type or 'equity'}",
        "symbol": "NVDA",
        "instrument": "option" if opt_type else "equity",
        "option_type": opt_type,
        "direction_matches": matches,
        "entry": 100.0,
        "stop": 95.0 if matches else 105.0,
        "target": 110.0 if matches else 90.0,
        "r_r": r_r,
        "confidence": conf,
        "premium_risk": premium,
        "contracts": 2.0,
        "shares": 100.0,
    }


def test_rank_bullish_prefers_call():
    candidates = [
        _candidate("call", matches=True),
        _candidate("put", matches=False),
        _candidate("", matches=True),
    ]
    ctx = {"composite_bias": "bullish", "agreement_score": 0.8, "risk_score": 30.0,
           "risk_level": "moderate", "spread_pct": 0.02, "greeks_source": "alpaca_option_chain"}
    out = rank_opportunities(candidates, ctx, min_confidence=0.35, min_risk_reward=1.0)
    assert out["trade_decision"] == "long_call"
    assert out["opportunities"][0]["rank"] == 1
    assert out["opportunities"][0]["option_type"] == "call"


def test_rank_bearish_selects_put():
    candidates = [
        _candidate("call", matches=False),
        _candidate("put", matches=True),
    ]
    ctx = {"composite_bias": "bearish", "agreement_score": 0.7, "risk_score": 40.0,
           "risk_level": "moderate", "spread_pct": 0.03, "greeks_source": "alpaca_option_chain"}
    out = rank_opportunities(candidates, ctx, min_confidence=0.35, min_risk_reward=1.0)
    assert out["trade_decision"] == "long_put"


def test_rank_rr_gate_holds():
    candidates = [_candidate("call", matches=True, r_r=0.5)]
    ctx = {"composite_bias": "bullish", "agreement_score": 0.8, "risk_score": 30.0,
           "risk_level": "moderate", "spread_pct": 0.02, "greeks_source": "alpaca_option_chain"}
    out = rank_opportunities(candidates, ctx, min_confidence=0.35, min_risk_reward=1.0)
    assert out["trade_decision"] == "hold"
    assert out["gates"]["call"]["min_risk_reward"]["passed"] is False


def test_rank_confidence_gate_holds():
    candidates = [_candidate("call", matches=True, conf=0.1)]
    ctx = {"composite_bias": "bullish", "agreement_score": 0.8, "risk_score": 30.0,
           "risk_level": "moderate", "spread_pct": 0.02, "greeks_source": "alpaca_option_chain"}
    out = rank_opportunities(candidates, ctx, min_confidence=0.35, min_risk_reward=1.0)
    assert out["trade_decision"] == "hold"
    assert out["gates"]["call"]["min_confidence"]["passed"] is False


def test_rank_very_high_risk_avoids():
    candidates = [_candidate("call", matches=True)]
    ctx = {"composite_bias": "bullish", "agreement_score": 0.8, "risk_score": 90.0,
           "risk_level": "very_high", "spread_pct": 0.02, "greeks_source": "alpaca_option_chain"}
    out = rank_opportunities(candidates, ctx, min_confidence=0.35, min_risk_reward=1.0)
    assert out["trade_decision"] == "avoid"


def test_rank_neutral_holds():
    candidates = [_candidate("call", matches=False), _candidate("put", matches=False)]
    ctx = {"composite_bias": "neutral", "agreement_score": 0.1, "risk_score": 30.0,
           "risk_level": "low", "spread_pct": 0.02, "greeks_source": "alpaca_option_chain"}
    out = rank_opportunities(candidates, ctx, min_confidence=0.35, min_risk_reward=1.0)
    assert out["trade_decision"] == "hold"


def test_rank_confidence_score_bounds():
    candidates = [_candidate("call", matches=True)]
    ctx = {"composite_bias": "bullish", "agreement_score": 0.9, "risk_score": 10.0,
           "risk_level": "low", "spread_pct": 0.01, "greeks_source": "alpaca_option_chain"}
    out = rank_opportunities(candidates, ctx, min_confidence=0.35, min_risk_reward=1.0)
    assert 0.0 <= out["confidence_score"] <= 1.0
    assert out["confidence_score"] >= 0.5
