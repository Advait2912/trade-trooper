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

from typing import Any

from tuning import DEFAULT_TUNING, TuningConfig


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


def _score_adx(
    adx_dict: dict[str, Any],
    adx_scale_map: dict[str, float] | None = None,
) -> tuple[float, float]:
    """ADX: direction score scaled by trend strength.

    Returns (raw_direction_score, strength_scale) so the caller can
    discount the weight for weak-trend environments.
    """
    scale_map = adx_scale_map or DEFAULT_TUNING.adx_strength_scale
    direction = adx_dict.get("trend_direction", "ranging")
    strength = adx_dict.get("trend_strength", "no_trend")
    scale = scale_map.get(strength, 0.0)
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


def _score_squeeze(squeeze_dict: dict[str, Any]) -> float:
    """Score John Carter's TTM Squeeze for directional breakout momentum (-1.0 to +1.0)."""
    sig = squeeze_dict.get("breakout_signal", "neutral")
    slope = squeeze_dict.get("momentum_slope", "flat")
    hist = float(squeeze_dict.get("momentum_hist", 0.0) or 0.0)

    if sig == "bullish_breakout":
        return 1.0
    if sig == "bearish_breakout":
        return -1.0
    if sig == "bullish_expansion":
        return 0.7 if slope == "increasing_bullish" else 0.4
    if sig == "bearish_expansion":
        return -0.7 if slope == "increasing_bearish" else -0.4
    if sig == "consolidating":
        # In a squeeze, early momentum drift can hint at breakout direction
        return 0.2 if hist > 0 else (-0.2 if hist < 0 else 0.0)
    return 0.0


def calculate_technical_indicators(
    technical: dict[str, Any],
    tuning: TuningConfig | None = None,
) -> dict[str, Any]:
    """Combine pre-computed Phase 1 indicator results into a momentum score.

    Parameters
    ----------
    technical:
        The ``HistoricalAgentResult.technical`` dict already computed in
        Phase 1 — contains nested dicts keyed by tool name.
    tuning:
        Optional ``TuningConfig`` overriding the default indicator weights.

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
        squeeze_signal    : str
        iv_regime         : str
    """
    t = tuning or DEFAULT_TUNING
    weights = t.momentum_weights
    rsi_dict = technical.get("calculate_rsi") or {}
    macd_dict = technical.get("calculate_macd") or {}
    adx_dict = technical.get("calculate_adx") or {}
    bb_dict = technical.get("calculate_bollinger_bands") or {}
    obv_dict = technical.get("calculate_obv") or {}
    stoch_dict = technical.get("calculate_stochastic") or {}
    squeeze_dict = technical.get("calculate_ttm_squeeze") or {}
    iv_dict = technical.get("calculate_iv_rank") or {}

    rsi_score = _score_rsi(rsi_dict)
    macd_score = _score_macd(macd_dict)
    adx_score, adx_scale = _score_adx(adx_dict, t.adx_strength_scale)
    bb_score = _score_bollinger(bb_dict)
    obv_score = _score_obv(obv_dict)
    stoch_score = _score_stochastic(stoch_dict)
    squeeze_score = _score_squeeze(squeeze_dict)

    # Effective ADX weight = base_weight × strength_scale
    effective_adx_weight = weights.get("adx", 0.15) * adx_scale
    # Redistribute unused ADX weight proportionally to other indicators
    dropped = weights.get("adx", 0.15) - effective_adx_weight
    other_keys = [k for k in weights if k != "adx"]
    other_total = sum(weights[k] for k in other_keys)
    adjusted_weights: dict[str, float] = {}
    if other_total > 0:
        for k in other_keys:
            adjusted_weights[k] = weights[k] + dropped * (weights[k] / other_total)
    else:
        for k in other_keys:
            adjusted_weights[k] = weights[k]
    adjusted_weights["adx"] = effective_adx_weight

    scores = {
        "rsi": rsi_score,
        "macd": macd_score,
        "adx": adx_score,
        "bollinger": bb_score,
        "obv": obv_score,
        "stochastic": stoch_score,
    }
    if "squeeze" in weights:
        scores["squeeze"] = squeeze_score

    raw = sum(adjusted_weights.get(k, 0.0) * scores[k] for k in scores)
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
        "squeeze_signal": squeeze_dict.get("breakout_signal", "neutral"),
        "iv_regime": iv_dict.get("vol_regime", "fair_value"),
    }


def apply_news_adjustment(
    momentum_score: float,
    news_sentiment: str,
    news_sentiment_score: float,
    news_weight: float | None = None,
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
    nw = news_weight if news_weight is not None else DEFAULT_TUNING.news_weight

    sign_map = {"bullish": 1.0, "bearish": -1.0, "neutral": 0.0, "uncertain": 0.0}
    sign = sign_map.get(news_sentiment, 0.0)

    # Use absolute value of sentiment_score; sign comes from the label.
    raw_adj = sign * abs(news_sentiment_score) * nw
    # Clamp the adjustment itself to ±NEWS_WEIGHT
    raw_adj = max(-nw, min(nw, raw_adj))

    adjusted = max(-1.0, min(1.0, momentum_score + raw_adj))
    return round(raw_adj, 4), round(adjusted, 4)
