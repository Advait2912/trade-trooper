"""Deterministic news tools (Phase 1 - News Collection Agent)."""

from __future__ import annotations

from typing import Any

from schemas.news import InitialAnalysis

_SENTIMENT_CONTRIBUTION = {
    "bullish": 1.0,
    "bearish": -1.0,
    "neutral": 0.0,
    "uncertain": 0.0,
}


def sentiment_analysis(analyses: list[InitialAnalysis]) -> dict[str, Any]:
    """Score the sentiment of collected news signals.

    Returns a deterministic aggregate in [-1, +1] weighted by each article's
    relevance, plus per-sentiment counts.
    """
    counts = {"bullish": 0, "bearish": 0, "neutral": 0, "uncertain": 0}
    for a in analyses:
        counts[a.sentiment.value] = counts.get(a.sentiment.value, 0) + 1

    total_weight = sum(a.relevance for a in analyses)
    if not analyses or total_weight <= 0:
        score = 0.0
    else:
        score = round(
            sum(
                _SENTIMENT_CONTRIBUTION[a.sentiment.value] * a.relevance
                for a in analyses
            )
            / total_weight,
            4,
        )

    return {
        "sentiment_score": score,
        "counts": counts,
        "articles_analyzed": len(analyses),
    }
