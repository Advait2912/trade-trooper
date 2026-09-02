"""Phase 2 - Technical signal combiner.

Reuses pre-computed indicator results from ``HistoricalAgentResult.technical``
(which already ran RSI, MACD, ADX, Bollinger, OBV, Stochastic in Phase 1) and
converts them into a single bounded momentum score plus a composite signal.

No indicators are re-computed here.  All math was done in Phase 1 by the
Historical Agent via ``tools/historical/indicators.py``.

Weights (tuned for options directional bias; sum = 1.0):
    MACD trend   : 0.25  — trend-following, reliable on daily bars
    ADX direction: 0.20  — only counts when trend is not "no_trend" / "weak"
    RSI signal   : 0.20  — contrarian when extreme, neutral in midrange
    Bollinger    : 0.15  — mean-reversion / breakout
    OBV          : 0.10  — volume confirmation
    Stochastic   : 0.10  — momentum crossovers

Scores per indicator: +1.0 (bullish), -1.0 (bearish), 0.0 (neutral).
ADX weight is discounted by trend_strength for weak or no-trend conditions.
"""

from __future__ import annotations

import math
from typing import Any


# ---------------------------------------------------------------------------
# Indicator weights (must sum to 1.0)
# ---------------------------------------------------------------------------
_WEIGHTS: dict[str, float] = {
    "macd": 0.25,
    "adx": 0.20,
    "rsi": 0.20,
    "bollinger": 0.15,
    "obv": 0.10,
    "stochastic": 0.10,
}

_ADX_STRENGTH_SCALE: dict[str, float] = {
    "no_trend": 0.0,
    "weak": 0.3,
    "moderate": 0.7,
    "strong": 1.0,
    "very_strong": 1.0,
}


def _score_rsi(rsi_dict: dict[str, Any]) -> float:
    """RSI: contrarian at extremes.

    oversold  → +1.0 (potential bounce up)
    overbought → -1.0 (potential reversal down)
    neutral    →  0.0
    """
    sig = rsi_dict.get("signal", "neutral")
    if sig == "oversold":
        return 1.0
    if sig == "overbought":
        return -1.0
    return 0.0


def _score_macd(macd_dict: dict[str, Any]) -> float:
    """MACD: trend direction, scaled by momentum_strength.

    Scale by strength so a 'weak' crossover doesn't dominate the composite:
        strong   → ±1.00
        moderate → ±0.70
        weak     → ±0.30

    Returns 0.0 when the dict is empty (no data → no opinion).
    """
    if not macd_dict:
        return 0.0
    trend = macd_dict.get("trend", "")
    strength = macd_dict.get("momentum_strength", "weak")
    strength_scale = {"strong": 1.0, "moderate": 0.70, "weak": 0.30}
    scale = strength_scale.get(strength, 0.50)
    if trend == "bullish":
        return 1.0 * scale
    if trend == "bearish":
        return -1.0 * scale
    return 0.0


def _score_adx(adx_dict: dict[str, Any]) -> tuple[float, float]:
    """ADX: direction score scaled by trend strength.

    Returns (raw_direction_score, strength_scale) so the caller can
    discount the weight for weak-trend environments.
    """
    direction = adx_dict.get("trend_direction", "ranging")
    strength = adx_dict.get("trend_strength", "no_trend")
    scale = _ADX_STRENGTH_SCALE.get(strength, 0.0)
    if direction == "uptrend":
        return 1.0, scale
    if direction == "downtrend":
        return -1.0, scale
    return 0.0, scale


def _score_bollinger(bb_dict: dict[str, Any]) -> float:
    """Bollinger Bands: position relative to bands.

    below_lower → +1.0 (oversold / bounce candidate)
    above_upper → -1.0 (overbought / exhaustion candidate)
    between_upper_middle → slight bullish → +0.3
    between_middle_lower → slight bearish → -0.3

    Returns 0.0 when the dict is empty (no data → no opinion).
    """
    if not bb_dict:
        return 0.0
    pos = bb_dict.get("band_position", "")
    mapping = {
        "below_lower": 1.0,
        "between_middle_lower": -0.3,
        "between_upper_middle": 0.3,
        "above_upper": -1.0,
    }
    return mapping.get(pos, 0.0)


def _score_obv(obv_dict: dict[str, Any]) -> float:
    """OBV: volume trend confirmation.

    strong_confirmation + increasing OBV → +0.8
    strong_confirmation + decreasing OBV → -0.8
    moderate_confirmation             → ±0.0
    divergence                        → opposite sign
    """
    trend = obv_dict.get("obv_trend", "flat")
    conf = obv_dict.get("volume_confirmation", "moderate_confirmation")
    if conf == "strong_confirmation":
        if trend == "increasing":
            return 0.8
        if trend == "decreasing":
            return -0.8
    if conf == "divergence":
        # OBV diverges from price — contrarian signal
        if trend == "decreasing":
            return 0.4   # bearish OBV in uptrend → contrarian bearish warning
        if trend == "increasing":
            return -0.4
    return 0.0


def _score_stochastic(stoch_dict: dict[str, Any]) -> float:
    """Stochastic: crossovers at extremes.

    overbought + bearish_cross → -1.0
    oversold   + bullish_cross → +1.0
    overbought only            → -0.5
    oversold only              → +0.5
    """
    sig = stoch_dict.get("signal", "neutral")
    cross = stoch_dict.get("crossover", "none")
    if sig == "oversold":
        return 1.0 if cross == "bullish_cross" else 0.5
    if sig == "overbought":
        return -1.0 if cross == "bearish_cross" else -0.5
    return 0.0


def calculate_technical_indicators(
    technical: dict[str, Any],
) -> dict[str, Any]:
    """Combine pre-computed Phase 1 indicator results into a momentum score.

    Parameters
    ----------
    technical:
        The ``HistoricalAgentResult.technical`` dict already computed in
        Phase 1 — contains nested dicts keyed by tool name.

    Returns
    -------
    dict with:
        momentum_score    : float in [-1.0, +1.0]
        composite_signal  : "bullish" | "bearish" | "neutral"
        rsi_signal        : str
        macd_signal       : str
        adx_trend_strength: str
        adx_trend_direction: str
        bollinger_regime  : str
        obv_confirmation  : str
    """
    rsi_dict = technical.get("calculate_rsi") or {}
    macd_dict = technical.get("calculate_macd") or {}
    adx_dict = technical.get("calculate_adx") or {}
    bb_dict = technical.get("calculate_bollinger_bands") or {}
    obv_dict = technical.get("calculate_obv") or {}
    stoch_dict = technical.get("calculate_stochastic") or {}

    rsi_score = _score_rsi(rsi_dict)
    macd_score = _score_macd(macd_dict)
    adx_score, adx_scale = _score_adx(adx_dict)
    bb_score = _score_bollinger(bb_dict)
    obv_score = _score_obv(obv_dict)
    stoch_score = _score_stochastic(stoch_dict)

    # Effective ADX weight = base_weight × strength_scale
    effective_adx_weight = _WEIGHTS["adx"] * adx_scale
    # Redistribute unused ADX weight proportionally to other indicators
    dropped = _WEIGHTS["adx"] - effective_adx_weight
    other_keys = [k for k in _WEIGHTS if k != "adx"]
    other_total = sum(_WEIGHTS[k] for k in other_keys)
    adjusted_weights: dict[str, float] = {}
    for k in other_keys:
        adjusted_weights[k] = _WEIGHTS[k] + dropped * (_WEIGHTS[k] / other_total)
    adjusted_weights["adx"] = effective_adx_weight

    scores = {
        "rsi": rsi_score,
        "macd": macd_score,
        "adx": adx_score,
        "bollinger": bb_score,
        "obv": obv_score,
        "stochastic": stoch_score,
    }

    raw = sum(adjusted_weights[k] * scores[k] for k in scores)
    momentum_score = max(-1.0, min(1.0, raw))

    if momentum_score > 0.15:
        composite_signal = "bullish"
    elif momentum_score < -0.15:
        composite_signal = "bearish"
    else:
        composite_signal = "neutral"

    return {
        "momentum_score": round(momentum_score, 4),
        "composite_signal": composite_signal,
        "rsi_signal": rsi_dict.get("signal", "neutral"),
        "macd_signal": macd_dict.get("trend", "neutral"),
        "adx_trend_strength": adx_dict.get("trend_strength", "weak"),
        "adx_trend_direction": adx_dict.get("trend_direction", "ranging"),
        "bollinger_regime": bb_dict.get("volatility_regime", "normal"),
        "obv_confirmation": obv_dict.get("volume_confirmation", "moderate_confirmation"),
    }


def apply_news_adjustment(
    momentum_score: float,
    news_sentiment: str,
    news_sentiment_score: float,
) -> tuple[float, float]:
    """Deterministically adjust momentum score for news sentiment.

    Rule (documented and bounded):
        news_sentiment_score is in [-1.0, +1.0] (from NewsCollectionResult).
        The sentiment string maps to a direction sign:
            bullish  → +1
            bearish  → -1
            neutral  → 0
            uncertain → 0

        Raw adjustment = sign × abs(news_sentiment_score) × NEWS_WEIGHT
        NEWS_WEIGHT = 0.20  (news can shift momentum by at most ±0.20)

        The adjustment is then clamped so the final adjusted_momentum
        remains in [-1.0, +1.0].

    This means:
        - News with sentiment_score = 0 has zero effect.
        - News can shift momentum at most 0.20 points in either direction.
        - Technical signals always retain at least 80% of their weight.
        - Neutral/uncertain news never creates a directional bias.

    Returns
    -------
    (news_adjustment, adjusted_momentum)
    """
    NEWS_WEIGHT = 0.20

    sign_map = {"bullish": 1.0, "bearish": -1.0, "neutral": 0.0, "uncertain": 0.0}
    sign = sign_map.get(news_sentiment, 0.0)

    # Use absolute value of sentiment_score; sign comes from the label.
    raw_adj = sign * abs(news_sentiment_score) * NEWS_WEIGHT
    # Clamp the adjustment itself to ±NEWS_WEIGHT
    raw_adj = max(-NEWS_WEIGHT, min(NEWS_WEIGHT, raw_adj))

    adjusted = max(-1.0, min(1.0, momentum_score + raw_adj))
    return round(raw_adj, 4), round(adjusted, 4)
