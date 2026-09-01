"""Phase 2 - Price move estimator.

Produces a directional price forecast over a fixed horizon using:
    1. ATR-based 1-sigma expected move (√T scaling).
    2. Adjusted momentum score (technical + news) to skew the direction.
    3. A rule-based confidence score.

Formula:
    atr_pct       = MarketData.atr14 / price
    expected_move = atr_pct × √horizon_days        # 1-sigma for horizon
    direction_bias = adjusted_momentum × expected_move × 0.50
    price_forecast = price × (1 + direction_bias)
    forecast_low   = price × (1 - expected_move)
    forecast_high  = price × (1 + expected_move)

The ×0.50 dampening means that even at full momentum (±1.0), the center
of the forecast is shifted by at most half the 1-sigma move — keeping the
forecast range realistic and auditable.

Confidence scoring (base 0.50, additive/subtractive adjustments):
    +0.15  ADX trend_strength in {strong, very_strong}
    +0.10  vol_regime == "normal"  (predictable environment)
    +0.10  |adjusted_momentum| > 0.40  (clear signal)
    -0.10  vol_regime in {high, very_high}  (chaotic / gap risk)
    -0.15  mean_reversion_score contradicts momentum direction
              (score > 0 AND momentum < 0, or vice versa)
    Clamped to [0.10, 0.90].
"""

from __future__ import annotations

import math
from typing import Any

from schemas.market import MarketData

_HORIZON_DEFAULT = 5   # trading days (one calendar week)
_DIRECTION_DAMPEN = 0.50
_CONF_BASE = 0.50
_CONF_MIN = 0.10
_CONF_MAX = 0.90


def estimate_price_move(
    market: MarketData,
    adjusted_momentum: float,
    adx_trend_strength: str,
    vol_regime: str,
    mean_reversion_score: float,
    horizon_days: int = _HORIZON_DEFAULT,
) -> dict[str, Any]:
    """Estimate the expected price move over ``horizon_days`` trading days.

    Parameters
    ----------
    market:
        Phase 1 ``MarketData`` — supplies ``price`` and ``atr14``.
    adjusted_momentum:
        Momentum score after news adjustment, in [-1.0, +1.0].
    adx_trend_strength:
        From the technical bundle: no_trend | weak | moderate | strong | very_strong.
    vol_regime:
        From the volatility bundle: low | normal | high | very_high.
    mean_reversion_score:
        From volatility bundle: negative = trending, positive = mean-reverting.
    horizon_days:
        Number of trading days ahead for the forecast.

    Returns
    -------
    dict with:
        price_forecast      : float
        price_forecast_low  : float
        price_forecast_high : float
        expected_move_pct   : float  — 1-sigma move as a fraction (e.g. 0.035)
        forecast_horizon_days: int
        confidence          : float  — [0.10, 0.90]
        status              : "ok" | "insufficient_data"
    """
    errors: list[str] = []

    price = market.price
    if price <= 0.0 or not math.isfinite(price):
        return {
            "price_forecast": 0.0,
            "price_forecast_low": 0.0,
            "price_forecast_high": 0.0,
            "expected_move_pct": 0.0,
            "forecast_horizon_days": horizon_days,
            "confidence": 0.0,
            "status": "insufficient_data",
            "errors": ["Invalid or zero price — cannot forecast."],
        }

    atr = market.atr14
    if atr <= 0.0 or not math.isfinite(atr):
        # Fall back to 2 % daily ATR if ATR is unavailable
        atr = price * 0.02
        errors.append("ATR unavailable; used 2 % of price as fallback.")

    atr_pct = atr / price
    horizon_days = max(1, horizon_days)
    expected_move = atr_pct * math.sqrt(horizon_days)

    direction_bias = adjusted_momentum * expected_move * _DIRECTION_DAMPEN
    price_forecast = price * (1.0 + direction_bias)
    forecast_low = price * (1.0 - expected_move)
    forecast_high = price * (1.0 + expected_move)

    # ---- Confidence ----
    conf = _CONF_BASE

    if adx_trend_strength in {"strong", "very_strong"}:
        conf += 0.15
    if vol_regime == "normal":
        conf += 0.10
    if abs(adjusted_momentum) > 0.40:
        conf += 0.10
    if vol_regime in {"high", "very_high"}:
        conf -= 0.10
    # Mean-reversion contradicts momentum direction
    if (mean_reversion_score > 0.05 and adjusted_momentum > 0.10) or \
       (mean_reversion_score > 0.05 and adjusted_momentum < -0.10):
        # High reversion score means price is trending toward the mean —
        # if momentum says up but reversion is high, confidence drops.
        conf -= 0.15

    confidence = max(_CONF_MIN, min(_CONF_MAX, conf))

    return {
        "price_forecast": round(price_forecast, 4),
        "price_forecast_low": round(forecast_low, 4),
        "price_forecast_high": round(forecast_high, 4),
        "expected_move_pct": round(expected_move, 6),
        "forecast_horizon_days": horizon_days,
        "confidence": round(confidence, 4),
        "status": "ok",
        "errors": errors,
    }
