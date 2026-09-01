"""Volatility analysis tools (Phase 1 - Historical Agent).

Realized volatility stats, regime detection, mean-reversion (Hurst) analysis
and cross-asset correlation.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from hurst import compute_Hc

_SQRT_252 = math.sqrt(252.0)


def _log_returns(prices: Sequence[float]) -> pd.Series:
    series = pd.Series([float(p) for p in prices], dtype="float64")
    shifted = series.shift(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        logr = np.log(series / shifted)
    return logr.where((series > 0) & (shifted > 0))


def _annualized_vol(returns: pd.Series, period: int) -> float | None:
    if len(returns.dropna()) < max(2, period):
        return None
    window = returns.tail(period)
    std = float(np.asarray(window.std()))
    if math.isnan(std):
        return None
    return float(std * _SQRT_252 * 100.0)


def calculate_historical_volatility(
    price_history: Sequence[float],
    period: int = 20,
) -> dict[str, Any]:
    """Calculate realized volatility from historical prices (annualized %)."""
    closes = [float(p) for p in price_history]
    if len(closes) < period + 2:
        return {"historical_vol": 0.0, "vol_trend": "stable", "vol_percentile": 0.0}

    returns = _log_returns(closes)
    recent = _annualized_vol(returns, period) or 0.0
    prev = _annualized_vol(returns, min(period * 3, len(returns) - 1)) or 0.0

    if prev > 0:
        ratio = recent / prev
        vol_trend = "increasing" if ratio >= 1.1 else ("decreasing" if ratio <= 0.9 else "stable")
    else:
        vol_trend = "stable"

    # Percentile of current period-vol vs the trailing 252 daily vols.
    vols = (returns.rolling(window=period).std() * _SQRT_252 * 100.0).dropna()
    if len(vols) >= 10 and not math.isinf(recent) and recent > 0:
        percentile = float((vols <= recent).mean() * 100.0)
    else:
        percentile = 0.0

    return {
        "historical_vol": round(recent, 4),
        "vol_trend": vol_trend,
        "vol_percentile": round(percentile, 1),
    }


def detect_volatility_regimes(
    volatility_history: Sequence[float],
    sensitivity: int = 3,
) -> dict[str, Any]:
    """Detect if volatility is in expansion or contraction phase."""
    vols = [float(v) for v in volatility_history]
    vols = [v for v in vols if not math.isnan(v)]
    if len(vols) < 15:
        return {
            "regime": "contraction",
            "vol_acceleration": 0.0,
            "expected_duration_days": 1,
        }

    current = float(vols[-1])
    hist = pd.Series(vols)
    p90, median = float(hist.quantile(0.90)), float(hist.median())
    recent = hist.tail(5).mean()
    prev = hist.iloc[-10:-5].mean() if len(hist) >= 10 else median
    ref = prev if prev > 0 else median
    accel = (recent - ref) / ref if ref > 0 else 0.0

    hi, lo = 0.05, -0.05
    if sensitivity >= 4:
        hi, lo = 0.03, -0.03
    elif sensitivity <= 2:
        hi, lo = 0.08, -0.08

    ratio = current / median if median > 0 else 1.0
    if accel > hi and ratio >= 1.0:
        regime = "high_expansion"
    elif accel > hi:
        regime = "expansion"
    elif accel < lo and ratio <= 0.7:
        regime = "low_contraction"
    elif accel < lo:
        regime = "contraction"
    else:
        regime = "expansion" if accel > 0 else "contraction"

    strength = min(1.0, abs(accel) * 20)
    duration = max(1, int(round(strength * 30 * len(vols) / 252)))
    if p90 and current >= p90 and regime.startswith("high"):
        duration = min(duration, 10)

    return {
        "regime": regime,
        "vol_acceleration": round(float(accel), 4),
        "expected_duration_days": duration,
    }


def analyze_mean_reversion(
    price_history: Sequence[float],
    lookback_days: int = 60,
) -> dict[str, Any]:
    """Analyze if price is mean-reverting or trending (Hurst exponent)."""
    closes = [float(p) for p in price_history]
    window = closes[-lookback_days:]
    if len(window) < 50 or any(c <= 0 for c in window):
        return {
            "mean_reversion_score": 0.0,
            "mean_price": round(sum(window) / len(window), 4) if window else 0.0,
            "deviation_from_mean": 0.0,
            "reversion_probability": 0.5,
        }

    returns = np.diff(np.log(np.array(window, dtype=float)))
    try:
        h = compute_Hc(returns)[0]
    except Exception:
        h = 0.5
    h = max(0.0, min(1.0, float(h)))

    score = 0.5 - h  # H=0.5 random walk -> 0.0
    score = max(-1.0, min(1.0, score * 2.0))

    mean_price = float(np.mean(window))
    current = float(window[-1])
    deviation = ((current - mean_price) / mean_price * 100.0) if mean_price else 0.0

    reversion_probability = 0.5 + score * 0.4
    reversion_probability = max(0.05, min(0.95, reversion_probability))

    return {
        "mean_reversion_score": round(score, 3),
        "mean_price": round(mean_price, 4),
        "deviation_from_mean": round(deviation, 4),
        "reversion_probability": round(reversion_probability, 3),
    }


def calculate_correlation(
    price_history_1: Sequence[float],
    price_history_2: Sequence[float],
    period: int = 60,
) -> dict[str, Any]:
    """Calculate correlation between two assets (log returns)."""
    n = min(len(price_history_1), len(price_history_2), period + 1)
    if n < 10:
        return {"correlation": 0.0, "interpretation": "neutral"}

    r1 = _log_returns(price_history_1[-n:])
    r2 = _log_returns(price_history_2[-n:])
    corr = float(r1.corr(r2))
    if math.isnan(corr):
        corr = 0.0

    if corr >= 0.8:
        interp = "highly_correlated"
    elif corr >= 0.5:
        interp = "correlated"
    elif corr > -0.5:
        interp = "neutral"
    elif corr > -0.8:
        interp = "inversely_correlated"
    else:
        interp = "highly_inversely_correlated"

    return {"correlation": round(corr, 4), "interpretation": interp}
