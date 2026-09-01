"""Phase 2 – Prediction Agent schemas.

``PredictionResult`` is the typed output produced by the Prediction Agent and
consumed by the Phase 3 Risk Agent.

``Phase1Bundle`` is an **extraction view** over the raw Phase 1 dict returned
by ``Pipeline._phase1()``.  It avoids passing raw dicts around between phases.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from schemas.historical import HistoricalAgentResult
from schemas.market import MarketData
from schemas.news import NewsCollectionResult


# ---------------------------------------------------------------------------
# Phase 1 → Phase 2 data contract
# ---------------------------------------------------------------------------
class Phase1Bundle(BaseModel):
    """Typed view of Phase 1 outputs passed to PredictionAgent.

    The historical bars (closes/highs/lows/volumes) are stored on
    ``HistoricalAgentResult`` after the Phase 1 schema extension.
    """

    news: NewsCollectionResult = Field(default_factory=lambda: NewsCollectionResult())
    market: MarketData = Field(default_factory=MarketData)
    historical: HistoricalAgentResult = Field(
        default_factory=lambda: HistoricalAgentResult()
    )

    @classmethod
    def from_phase1_dict(cls, phase1: dict) -> "Phase1Bundle":
        """Extract from the raw dict returned by ``Pipeline._phase1()``."""
        return cls(
            news=phase1.get("news") or NewsCollectionResult(),
            market=phase1.get("market") or MarketData(),
            historical=phase1.get("historical") or HistoricalAgentResult(),
        )


# ---------------------------------------------------------------------------
# Phase 2 output
# ---------------------------------------------------------------------------
class PredictionResult(BaseModel):
    """Deterministic prediction produced by Phase 2.

    All numbers are computed by deterministic tools; no LLM is involved.

    ``iv_forecast`` is an *estimated forward volatility / IV proxy* derived
    from realized historical volatility scaled by a regime multiplier.
    It is **not** market-observed implied volatility from an options chain.
    ``iv_source`` is always ``"estimated"`` until Phase 3 fetches a real
    options chain.
    """

    # ---- price forecast ----
    price_forecast: float = 0.0
    price_forecast_low: float = 0.0
    price_forecast_high: float = 0.0
    expected_move_pct: float = 0.0
    forecast_horizon_days: int = 5

    # ---- volatility forecast ----
    iv_forecast: float = 0.0
    iv_source: str = "estimated"   # always "estimated" in Phase 2
    vol_regime: str = "normal"     # low | normal | high | very_high
    vol_percentile: float = 0.0
    vol_acceleration: float = 0.0
    hv_20: float = 0.0
    hv_60: float = 0.0

    # ---- composite signal ----
    composite_signal: str = "neutral"   # bullish | bearish | neutral
    momentum_score: float = 0.0        # [-1.0, +1.0] — technical only
    adjusted_momentum: float = 0.0     # after deterministic news adjustment

    # ---- sub-signals (for auditability) ----
    rsi_signal: str = "neutral"
    macd_signal: str = "neutral"
    adx_trend_strength: str = "weak"
    adx_trend_direction: str = "ranging"
    bollinger_regime: str = "normal"
    obv_confirmation: str = "moderate_confirmation"
    mean_reversion_score: float = 0.0

    # ---- news adjustment (deterministic) ----
    news_sentiment: str = "uncertain"  # bullish | bearish | neutral | uncertain
    news_sentiment_score: float = 0.0  # raw NewsCollectionResult.sentiment_score
    news_adjustment: float = 0.0      # bounded delta applied to momentum

    # ---- meta ----
    confidence: float = 0.0   # [0.0, 1.0]
    status: str = "ok"        # ok | error | insufficient_data
    summary: str = ""
    errors: list[str] = Field(default_factory=list)

