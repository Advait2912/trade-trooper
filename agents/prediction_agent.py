"""Phase 2 - Prediction Agent.

Receives the complete Phase 1 bundle and runs three deterministic prediction
tools in sequence.  No LLM call is made in this phase; all numbers are
computed by pure functions.

Flow:
    Phase1Bundle
        ↓
    1. calculate_technical_indicators(historical.technical)
           → momentum_score, composite_signal, sub-signals
        ↓
    2. apply_news_adjustment(momentum_score, news_sentiment, news_sentiment_score)
           → adjusted_momentum, news_adjustment
        ↓
    3. forecast_volatility(historical.volatility, historical.historical_trends)
           → iv_forecast (estimated), vol_regime, vol_percentile, hv_20, hv_60,
             mean_reversion_score, vol_acceleration
        ↓
    4. estimate_price_move(market, adjusted_momentum, adx_strength, vol_regime,
                           mean_reversion_score)
           → price_forecast, price_forecast_low/high, expected_move_pct, confidence
        ↓
    PredictionResult
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tuning import TuningConfig

from schemas.prediction import Phase1Bundle, PredictionResult
from tools.prediction_tools.price_move import estimate_price_move
from tools.prediction_tools.technical import (
    apply_news_adjustment,
    calculate_technical_indicators,
)
from tools.prediction_tools.volatility_forecast import forecast_volatility
from tuning import TuningConfig

log = logging.getLogger("market_intel_agent.prediction_agent")


class PredictionAgent:
    """Phase 2: deterministic prediction from Phase 1 data bundle."""

    name = "prediction"
    description = (
        "Combines Phase 1 indicator data into a directional price forecast "
        "and estimated forward volatility. No LLM is used."
    )
    phase = 2
    tools = [
        "calculate_technical_indicators",
        "apply_news_adjustment",
        "forecast_volatility",
        "estimate_price_move",
    ]

    async def run(
        self,
        phase1: dict | Phase1Bundle,
        tuning: TuningConfig | None = None,
    ) -> PredictionResult:
        """Run Phase 2 prediction.

        Parameters
        ----------
        phase1:
            Either the raw dict from ``Pipeline._phase1()`` or a typed
            ``Phase1Bundle``.
        tuning:
            Optional ``TuningConfig`` overriding indicator weights.
        """
        # Accept both raw dict and typed bundle
        if isinstance(phase1, dict):
            bundle = Phase1Bundle.from_phase1_dict(phase1)
        else:
            bundle = phase1

        errors: list[str] = []

        try:
            # ------------------------------------------------------------------
            # Step 1: Technical signals (reuse Phase 1 pre-computed indicators)
            # ------------------------------------------------------------------
            tech = calculate_technical_indicators(bundle.historical.technical, tuning=tuning)
            momentum_score: float = tech["momentum_score"]

            # ------------------------------------------------------------------
            # Step 2: Deterministic news adjustment
            # ------------------------------------------------------------------
            news = bundle.news
            news_sentiment_str: str = "uncertain"
            news_raw_score: float = float(news.sentiment_score)

            if news.primary_analysis is not None:
                news_sentiment_str = news.primary_analysis.sentiment.value
            elif abs(news_raw_score) > 0.05:
                # Infer direction from raw score if no primary analysis
                news_sentiment_str = "bullish" if news_raw_score > 0 else "bearish"

            news_adjustment, adjusted_momentum = apply_news_adjustment(
                momentum_score,
                news_sentiment_str,
                news_raw_score,
                news_weight=(tuning.news_weight if tuning else None),
            )

            # ------------------------------------------------------------------
            # Step 3: Volatility forecast
            # ------------------------------------------------------------------
            vol = forecast_volatility(
                bundle.historical.volatility,
                bundle.historical.historical_trends,
                tuning=tuning,
            )

            # ------------------------------------------------------------------
            # Step 4: Price move estimate
            # ------------------------------------------------------------------
            move = estimate_price_move(
                market=bundle.market,
                adjusted_momentum=adjusted_momentum,
                adx_trend_strength=tech["adx_trend_strength"],
                vol_regime=vol["vol_regime"],
                mean_reversion_score=vol["mean_reversion_score"],
                tuning=tuning,
            )

            if move.get("errors"):
                errors.extend(move["errors"])

            return PredictionResult(
                # Price forecast
                price_forecast=move["price_forecast"],
                price_forecast_low=move["price_forecast_low"],
                price_forecast_high=move["price_forecast_high"],
                expected_move_pct=move["expected_move_pct"],
                forecast_horizon_days=move["forecast_horizon_days"],
                # Volatility
                iv_forecast=vol["iv_forecast"],
                iv_source=vol["iv_source"],
                vol_regime=vol["vol_regime"],
                vol_percentile=vol["vol_percentile"],
                vol_acceleration=vol["vol_acceleration"],
                hv_20=vol["hv_20"],
                hv_60=vol["hv_60"],
                # Composite signal
                composite_signal=tech["composite_signal"],
                momentum_score=momentum_score,
                adjusted_momentum=adjusted_momentum,
                # Sub-signals
                rsi_signal=tech["rsi_signal"],
                macd_signal=tech["macd_signal"],
                adx_trend_strength=tech["adx_trend_strength"],
                adx_trend_direction=tech["adx_trend_direction"],
                bollinger_regime=tech["bollinger_regime"],
                obv_confirmation=tech["obv_confirmation"],
                mean_reversion_score=vol["mean_reversion_score"],
                # News
                news_sentiment=news_sentiment_str,
                news_sentiment_score=news_raw_score,
                news_adjustment=news_adjustment,
                # Meta
                confidence=move["confidence"],
                status=move.get("status", "ok"),
                summary=f"Forecast: ${move['price_forecast']:.2f} ({tech['composite_signal']} bias, conf: {move['confidence']:.2f})",
                errors=errors,
            )

        except Exception as exc:
            log.exception("PredictionAgent failed: %s", exc)
            return PredictionResult(
                status="error",
                summary=f"Error: {exc}",
                errors=[f"PredictionAgent error: {exc}"],
            )

