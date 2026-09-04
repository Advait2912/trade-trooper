"""Phase 4 - Signal synthesis tool.

``synthesize_signals`` aggregates every deterministic directional signal
produced by Phases 1-2 into a single weighted view.  It votes per source,
weights by each source's reliability, and returns a ``composite_bias``, an
``agreement_score``, an auditable per-source vote map, and a list of
divergences (strong sources pointing in opposite directions).

Direction only — risk/instrument selection is handled separately by
``rank_opportunities``.
"""

from __future__ import annotations

from typing import Any

from tuning import DEFAULT_TUNING, TuningConfig

_BIAS = {"bullish": 1.0, "bearish": -1.0, "neutral": 0.0}

# strength tiers for the technical summary overall_signal
_SUMMARY_STRENGTH: dict[str, float] = {
    "strong_bullish": 1.0,
    "bullish": 0.75,
    "neutral": 0.0,
    "bearish": -0.75,
    "strong_bearish": -1.0,
}

_NEWS_SENTIMENT: dict[str, float] = {
    "bullish": 1.0,
    "bearish": -1.0,
    "neutral": 0.0,
    "uncertain": 0.0,
}


class Signal:
    """A single weighted directional vote."""

    __slots__ = ("source", "bias", "weight", "detail")

    def __init__(self, source: str, bias: str, weight: float, detail: str = "") -> None:
        self.source = source
        self.bias = bias
        self.weight = weight
        self.detail = detail


def synthesize_signals(
    bundle: Any,
    prediction: Any,
    risk: Any | None = None,
    tuning: TuningConfig | None = None,
) -> dict[str, Any]:
    """Synthesize a directional composite signal from all Phase 1-2 signals.

    Parameters
    ----------
    bundle:
        A ``Phase1Bundle`` (or any object exposing ``news``, ``market``,
        ``historical``).
    prediction:
        A ``PredictionResult``.
    risk:
        Optional ``RiskResult`` (used only to reflect risk posture in the
        notes; risk does not change direction).

    Returns
    -------
    dict with ``composite_bias``, ``agreement_score``, ``signal_votes``,
    ``divergences`` and ``notes``.
    """
    votes: list[Signal] = []
    sw = (tuning or DEFAULT_TUNING).signal_weights

    news = getattr(bundle, "news", None)
    if news is not None:
        sentiment_score = float(getattr(news, "sentiment_score", 0.0) or 0.0)
        bias = _bias_from_score(sentiment_score, threshold=0.15)
        votes.append(Signal("news_sentiment", bias, sw["news_sentiment"],
                            f"sentiment_score={sentiment_score:.3f}"))

    historical = getattr(bundle, "historical", None)
    if historical is not None:
        summary = getattr(historical, "summary", None) or {}
        overall = str(summary.get("overall_signal", "neutral"))
        strength = _SUMMARY_STRENGTH.get(overall, 0.0)
        bias = "bullish" if strength > 0 else ("bearish" if strength < 0 else "neutral")
        votes.append(Signal("technical_summary", bias, sw["technical_summary"],
                            f"overall_signal={overall}"))

        trends = getattr(historical, "historical_trends", None) or {}
        trend_class = str(trends.get("trend_class", "neutral"))
        bias = "bullish" if trend_class == "bullish" else ("bearish" if trend_class == "bearish" else "neutral")
        votes.append(Signal("historical_trend", bias, sw["historical_trend"],
                            f"trend={trends.get('trend', 'n/a')}"))

    if prediction is not None:
        composite = str(getattr(prediction, "composite_signal", "neutral"))
        bias = "bullish" if composite == "bullish" else ("bearish" if composite == "bearish" else "neutral")
        # Direction-only vote.  Prediction confidence is deliberately NOT used
        # here: it is reserved for the Phase 4 confidence gate and Phase 3
        # position sizing, so the tuning harness can adjust voting weights and
        # confidence knobs independently (no double-counting).
        votes.append(Signal("prediction_signal", bias, sw["prediction_signal"],
                            f"composite_signal={composite}"))

    market = getattr(bundle, "market", None)
    if market is not None:
        trend = getattr(market, "trend", None)
        trend_val = getattr(trend, "value", None) if trend is not None else None
        bias = _bias_from_score(_BIAS.get(str(trend_val), 0.0) if str(trend_val) in _BIAS else 0.0, threshold=0.0)
        if bias == "neutral":
            trend_score = float(getattr(market, "return_5d", 0.0) or 0.0)
            bias = _bias_from_score(trend_score, threshold=0.0)
        votes.append(Signal("market_trend", bias, sw["market_trend"],
                            f"trend={trend_val}"))

    # ---- aggregate ----
    total_weight = sum(v.weight for v in votes) or 1.0
    bull_weight = sum(v.weight for v in votes if v.bias == "bullish")
    bear_weight = sum(v.weight for v in votes if v.bias == "bearish")
    weighted = (bull_weight - bear_weight) / total_weight
    composite_bias = _bias_from_score(weighted, threshold=0.10)
    agreement_score = abs(weighted)

    # divergences: any bullish vote opposed by a bearish vote among strong sources
    divergences: list[str] = []
    directional = [v for v in votes if v.bias != "neutral"]
    for v in directional:
        opposed = [o for o in directional if o.source != v.source and o.bias != v.bias and o.bias != "neutral"]
        if opposed and v.weight >= 0.20:
            divergences.append(
                f"{v.source} ({v.bias}) conflicts with {opposed[0].source} ({opposed[0].bias})"
            )
    divergences = list(dict.fromkeys(divergences))[:5]

    notes = [
        f"votes={len(votes)}, bull={bull_weight:.2f}, bear={bear_weight:.2f}, "
        f"agreement={agreement_score:.2f}"
    ]

    return {
        "composite_bias": composite_bias,
        "agreement_score": round(agreement_score, 4),
        "signal_votes": {
            v.source: {"bias": v.bias, "weight": v.weight, "detail": v.detail}
            for v in votes
        },
        "divergences": divergences,
        "notes": notes,
    }


def _bias_from_score(score: float, threshold: float) -> str:
    if score > threshold:
        return "bullish"
    if score < -threshold:
        return "bearish"
    return "neutral"
