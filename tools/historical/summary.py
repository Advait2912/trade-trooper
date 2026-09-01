"""Technical summary tool (Phase 1 - Historical Agent).

Votes across all computed indicators into one overall signal, plus a short
human-readable summary.
"""

from __future__ import annotations

from typing import Any


def generate_technical_summary(
    symbol: str,
    all_indicators: dict[str, Any],
) -> dict[str, Any]:
    """Generate a comprehensive technical analysis summary.

    Votes deterministic interpretations from the pre-computed indicator
    bundle (see ``tools.historical.indicators`` + friends) into a single
    ``overall_signal``.
    """
    votes: dict[str, int] = {"bullish": 0, "bearish": 0, "neutral": 0}
    notes: list[str] = []

    _vote(votes, notes, all_indicators, "calculate_moving_averages",
          bullish_val="bullish", bearish_val="bearish",
          reason_kind="signal", heading="Moving averages")
    _vote(votes, notes, all_indicators, "calculate_rsi",
          bullish_val="oversold", bearish_val="overbought",
          reason_kind="signal", heading="RSI")
    _vote(votes, notes, all_indicators, "calculate_macd",
          bullish_val="bullish", bearish_val="bearish",
          reason_kind="trend", heading="MACD")
    _vote(votes, notes, all_indicators, "calculate_bollinger_bands",
          bullish_val="below_lower", bearish_val="above_upper",
          reason_kind="band_position", heading="Bollinger Bands")
    _vote(votes, notes, all_indicators, "calculate_adx",
          bullish_val="uptrend", bearish_val="downtrend",
          reason_kind="trend_direction", heading="ADX")
    _vote(votes, notes, all_indicators, "calculate_obv",
          bullish_val="increasing", bearish_val="decreasing",
          reason_kind="obv_trend", heading="On-Balance Volume")
    _vote(votes, notes, all_indicators, "identify_trend",
          bullish_val="bullish", bearish_val="bearish",
          reason_kind="trend_class", heading="Trend")

    bull, bear, neut = votes["bullish"], votes["bearish"], votes["neutral"]
    if bull >= 5 and bull - bear >= 3:
        overall = "strong_bullish"
    elif bear >= 5 and bear - bull >= 3:
        overall = "strong_bearish"
    elif bull >= 3 and bull - bear >= 1:
        overall = "bullish"
    elif bear >= 3 and bear - bull >= 1:
        overall = "bearish"
    else:
        overall = "neutral"

    key_levels: dict[str, float] = {}
    levels = all_indicators.get("identify_support_resistance", {})
    resistances = levels.get("resistance_levels") or []
    supports = levels.get("support_levels") or []
    if resistances:
        key_levels["resistance"] = float(resistances[0]["level"])
    if supports:
        key_levels["support"] = float(supports[0]["level"])

    summary_text = (
        f"{symbol}: technical indicators vote {bull} bullish / {bear} bearish "
        f"/ {neut} neutral; overall signal: {overall}."
    )
    if notes:
        summary_text += " " + " ".join(notes[:4])

    return {
        "overall_signal": overall,
        "bullish_signals": bull,
        "bearish_signals": bear,
        "neutral_signals": neut,
        "key_levels": key_levels,
        "summary_text": summary_text,
    }


def _vote(
    votes: dict[str, int],
    notes: list[str],
    all_indicators: dict[str, Any],
    tool_key: str,
    bullish_val: str,
    bearish_val: str,
    reason_kind: str,
    heading: str,
    magnitude_key: str | None = None,
    magnitude_min: str | None = None,
) -> None:
    """Cast one vote from a nested indicator dict."""
    bundle = all_indicators.get(tool_key)
    if not isinstance(bundle, dict):
        votes["neutral"] += 1
        return

    if magnitude_key and magnitude_min:
        mag = bundle.get(magnitude_key)
        if isinstance(mag, str):
            order = ["no_trend", "weak", "moderate", "strong", "very_strong"]
            if mag in order and order.index(mag) < order.index(magnitude_min):
                votes["neutral"] += 1
                return

    value = bundle.get(reason_kind)
    if value == bullish_val:
        vote = "bullish"
    elif value == bearish_val:
        vote = "bearish"
    else:
        vote = "neutral"

    votes[vote] += 1
    notes.append(f"{heading}: {value}")
