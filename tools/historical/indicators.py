"""Deterministic technical indicators (Phase 1 - Historical Data Agent).

Two layers:
1. Pure-Python core (``sma``, ``rsi``, ``atr``, ``return_over``, ...) — kept
   from the original ``analysis/indicators.py``, no external dependencies.
2. Tool-level functions backed by the ``ta`` library (``calculate_macd``,
   ``calculate_adx``, ...) matching the declared tool contracts.

All functions accept plain lists and return JSON-ready dicts; none raise on
insufficient history — they return neutral/None values instead.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from ta.momentum import StochasticOscillator
from ta.trend import MACD, ADXIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import OnBalanceVolumeIndicator


# ===========================================================================
# Pure-Python core
# ===========================================================================
def sma(values: Sequence[float], period: int) -> float | None:
    """Simple moving average over the most recent `period` values."""
    if period <= 0 or len(values) < period:
        return None
    window = values[-period:]
    return sum(window) / period


def ema(values: Sequence[float], period: int) -> float | None:
    """Exponential moving average of the current value."""
    if period <= 0 or len(values) < period:
        return None
    series = pd.Series(values, dtype="float64")
    return float(series.ewm(span=period, adjust=False).mean().iloc[-1])


def rsi(closes: Sequence[float], period: int = 14) -> float | None:
    """Wilder's Relative Strength Index (pure Python).

    A flat series yields 50.0 (neutral); all-gains -> 100; all-losses -> 0.
    """
    if period <= 0 or len(closes) < period + 1:
        return None

    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [d if d > 0 else 0.0 for d in deltas]
    losses = [-d if d < 0 else 0.0 for d in deltas]

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0.0 and avg_gain == 0.0:
        return 50.0
    if avg_loss == 0.0:
        return 100.0
    if avg_gain == 0.0:
        return 0.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def true_ranges(
    highs: Sequence[float], lows: Sequence[float], closes: Sequence[float]
) -> list[float]:
    """True range for each bar, using the prior close as reference."""
    n = len(highs)
    trs: list[float] = []
    for i in range(n):
        h = highs[i]
        l = lows[i]
        prev_close = closes[i - 1] if i > 0 else closes[i]
        tr = max(h - l, abs(h - prev_close), abs(l - prev_close))
        trs.append(tr)
    return trs


def atr(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    period: int = 14,
) -> float | None:
    """Wilder's Average True Range. Requires at least `period + 1` bars."""
    if period <= 0 or len(highs) < period + 1:
        return None
    effective = min(period, len(highs) - 1)
    return float(
        AverageTrueRange(
            pd.Series(highs, dtype="float64"),
            pd.Series(lows, dtype="float64"),
            pd.Series(closes, dtype="float64"),
            window=effective,
        ).average_true_range().iloc[-1]
    )


def return_over(closes: Sequence[float], days: int) -> float | None:
    """Percentage return over the trailing `days` bars.

    e.g. days=1 -> (close[-1] / close[-2]) - 1.
    """
    if days <= 0 or len(closes) < days + 1:
        return None
    start = closes[-(days + 1)]
    end = closes[-1]
    if start == 0:
        return None
    return (end / start) - 1.0


def volume_vs_average(volumes: Sequence[float], period: int = 20) -> float | None:
    """Latest volume divided by the trailing average volume."""
    if period <= 0 or len(volumes) < period or not volumes[-1]:
        return None
    avg = sum(volumes[-period:]) / period
    if avg == 0:
        return None
    return volumes[-1] / avg


def volatility(closes: Sequence[float], period: int = 20) -> float | None:
    """Short-term realized volatility: annualized std of daily log returns."""
    if period <= 0 or len(closes) < period + 1:
        return None
    logs = [
        math.log(closes[i] / closes[i - 1])
        for i in range(len(closes) - period, len(closes))
        if closes[i - 1] > 0 and closes[i] > 0
    ]
    if len(logs) < 2:
        return None
    mean = sum(logs) / len(logs)
    variance = sum((x - mean) ** 2 for x in logs) / (len(logs) - 1)
    return math.sqrt(variance) * math.sqrt(252)


# ===========================================================================
# Tool-level indicator functions (ta-backed)
# ===========================================================================
def _series(values: Sequence[float]) -> pd.Series:
    return pd.Series(values, dtype="float64")


def _clean(value: Any, default: float = 0.0) -> float:
    try:
        f = float(value)
        return default if np.isnan(f) else f
    except (TypeError, ValueError):
        return default


def _round(value: float, digits: int = 4) -> float:
    return round(value, digits)


def calculate_moving_averages(
    price_history: Sequence[float],
    periods: Sequence[int] = (20, 50, 200),
    type: str = "both",
) -> dict[str, Any]:
    """Calculate simple and exponential moving averages.

    Returns MA values for the requested periods, the current price, and a
    signal derived from price relative to the short/long MAs.
    """
    closes = [float(p) for p in price_history]
    current = closes[-1] if closes else 0.0
    out: dict[str, Any] = {}
    if type in ("SMA", "both"):
        for p in periods:
            value = sma(closes, p)
            out[f"sma_{p}"] = _clean(value)
    if type in ("EMA", "both"):
        for p in periods:
            value = ema(closes, p)
            out[f"ema_{p}"] = _clean(value)

    ma_20 = sma(closes, 20) or 0.0
    ma_50 = sma(closes, 50) or 0.0
    ma_200 = sma(closes, 200) or 0.0
    out["ma_20"], out["ma_50"], out["ma_200"] = (
        _clean(ma_20),
        _clean(ma_50),
        _clean(ma_200),
    )
    out["current_price"] = _round(current, 4)
    out["signal"] = _ma_signal(current, ma_20, ma_50, ma_200)
    return out


def _ma_signal(price: float, *mas: float) -> str:
    above = [m for m in mas if m > 0 and price > m]
    below = [m for m in mas if m > 0 and price < m]
    if price <= 0 or len(above) == len(below) == 0:
        return "neutral"
    if len(above) > len(below):
        return "bullish"
    if len(below) > len(above):
        return "bearish"
    short_ma = mas[0] if mas else 0.0
    if short_ma > 0:
        return "bullish" if price >= short_ma else "bearish"
    return "neutral"


def calculate_rsi(price_history: Sequence[float], period: int = 14) -> dict[str, Any]:
    """Calculate Relative Strength Index (momentum indicator)."""
    closes = [float(p) for p in price_history]
    value = rsi(closes, period)
    if value is None:
        return {"rsi": 0.0, "signal": "neutral", "momentum": "down"}
    value = _clean(value, 50.0)
    if value > 70:
        signal = "overbought"
    elif value < 30:
        signal = "oversold"
    else:
        signal = "neutral"
    if value >= 80:
        momentum = "strong_up"
    elif value >= 60:
        momentum = "up"
    elif value <= 20:
        momentum = "strong_down"
    elif value <= 40:
        momentum = "down"
    else:
        momentum = "up" if value >= 50 else "down"
    return {"rsi": _round(value), "signal": signal, "momentum": momentum}


def calculate_macd(
    price_history: Sequence[float],
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> dict[str, Any]:
    """Calculate MACD (trend-following momentum indicator)."""
    closes = [float(p) for p in price_history]
    if len(closes) < slow_period + signal_period:
        return {
            "macd": 0.0,
            "signal_line": 0.0,
            "histogram": 0.0,
            "trend": "bearish",
            "momentum_strength": "weak",
        }
    indicator = MACD(
        _series(closes),
        window_fast=fast_period,
        window_slow=slow_period,
        window_sign=signal_period,
    )
    macd = _clean(indicator.macd().iloc[-1])
    signal_line = _clean(indicator.macd_signal().iloc[-1])
    histogram = _clean(indicator.macd_diff().iloc[-1])
    trend = "bullish" if macd > signal_line else "bearish"
    price = closes[-1] or 1.0
    act = abs(histogram) / price
    if act > 0.02:
        momentum = "strong"
    elif act > 0.01:
        momentum = "moderate"
    else:
        momentum = "weak"
    return {
        "macd": _round(macd),
        "signal_line": _round(signal_line),
        "histogram": _round(histogram),
        "trend": trend,
        "momentum_strength": momentum,
    }


def calculate_bollinger_bands(
    price_history: Sequence[float],
    period: int = 20,
    std_dev: float = 2.0,
) -> dict[str, Any]:
    """Calculate Bollinger Bands (volatility indicator)."""
    closes = [float(p) for p in price_history]
    if len(closes) < period:
        return {
            "upper_band": 0.0,
            "middle_band": 0.0,
            "lower_band": 0.0,
            "current_price": 0.0,
            "band_position": "between_middle_lower",
            "squeeze": False,
            "volatility_regime": "normal",
        }
    all_bbands = BollingerBands(
        _series(closes), window=period, window_dev=std_dev  # type: ignore[arg-type]
    )
    upper = _clean(all_bbands.bollinger_hband().iloc[-1])
    middle = _clean(all_bbands.bollinger_mavg().iloc[-1])
    lower = _clean(all_bbands.bollinger_lband().iloc[-1])
    current = closes[-1]

    if current > upper:
        position = "above_upper"
    elif current > middle:
        position = "between_upper_middle"
    elif current > lower:
        position = "between_middle_lower"
    else:
        position = "below_lower"

    # Squeeze/regime: compare current bandwidth to its recent history.
    upper_s = all_bbands.bollinger_hband()
    lower_s = all_bbands.bollinger_lband()
    middle_s = all_bbands.bollinger_mavg()
    band_width = (upper_s - lower_s) / middle_s.replace(0.0, np.nan)
    current_bw = float(band_width.iloc[-1]) if not np.isnan(band_width.iloc[-1]) else 0.0
    hist = band_width.dropna().tail(100)
    if len(hist) >= period:
        q83, q17 = float(hist.quantile(0.83)), float(hist.quantile(0.17))
        squeeze = current_bw <= q17
        regime = "high" if current_bw >= q83 else ("low" if squeeze else "normal")
    else:
        squeeze, regime = False, "normal"

    return {
        "upper_band": _round(upper),
        "middle_band": _round(middle),
        "lower_band": _round(lower),
        "current_price": _round(current),
        "band_position": position,
        "squeeze": squeeze,
        "volatility_regime": regime,
    }


def calculate_stochastic(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    period: int = 14,
) -> dict[str, Any]:
    """Calculate Stochastic Oscillator (%K, %D)."""
    if len(closes) < period + 2:
        return {
            "k_percent": 0.0,
            "d_percent": 0.0,
            "signal": "neutral",
            "crossover": "none",
        }
    stoch = StochasticOscillator(
        _series(highs), _series(lows), _series(closes), window=period
    )
    k_vals = np.asarray(stoch.stoch(), dtype=float)
    k_vals = k_vals[~np.isnan(k_vals)]
    if len(k_vals) < 3:
        return {
            "k_percent": 0.0,
            "d_percent": 0.0,
            "signal": "neutral",
            "crossover": "none",
        }
    k = float(k_vals[-1])
    d = float(k_vals[-3:].mean())

    if k > 80:
        signal = "overbought"
    elif k < 20:
        signal = "oversold"
    else:
        signal = "neutral"

    if len(k_vals) >= 4:
        prev_k = float(k_vals[-2])
        prev_d = float(k_vals[-4:-1].mean())
        if prev_k <= prev_d and k > d:
            crossover = "bullish_cross"
        elif prev_k >= prev_d and k < d:
            crossover = "bearish_cross"
        else:
            crossover = "none"
    else:
        crossover = "none"

    return {
        "k_percent": _round(k),
        "d_percent": _round(d),
        "signal": signal,
        "crossover": crossover,
    }


def calculate_atr(
    high_prices: Sequence[float],
    low_prices: Sequence[float],
    close_prices: Sequence[float],
    period: int = 14,
) -> dict[str, Any]:
    """Calculate Average True Range (volatility measure)."""
    value = atr(high_prices, low_prices, close_prices, period)
    if value is None:
        return {"atr": 0.0, "atr_percent": 0.0, "volatility_level": "normal"}
    current = close_prices[-1] or 1.0
    atr_percent = value / current * 100.0
    if atr_percent > 5.0:
        level = "very_high"
    elif atr_percent > 3.5:
        level = "high"
    elif atr_percent > 2.0:
        level = "normal"
    elif atr_percent > 1.0:
        level = "low"
    else:
        level = "very_low"
    return {
        "atr": _round(value),
        "atr_percent": _round(atr_percent),
        "volatility_level": level,
    }


def calculate_adx(
    high_prices: Sequence[float],
    low_prices: Sequence[float],
    close_prices: Sequence[float],
    period: int = 14,
) -> dict[str, Any]:
    """Calculate ADX (trend strength indicator)."""
    if len(close_prices) < period * 2:
        return {
            "adx": 0.0,
            "plus_di": 0.0,
            "minus_di": 0.0,
            "trend_strength": "no_trend",
            "trend_direction": "ranging",
        }
    adx_ind = ADXIndicator(
        _series(high_prices), _series(low_prices), _series(close_prices), window=period
    )
    adx_val = _clean(adx_ind.adx().iloc[-1])
    plus_di = _clean(adx_ind.adx_pos().iloc[-1])
    minus_di = _clean(adx_ind.adx_neg().iloc[-1])

    if adx_val >= 50:
        strength = "very_strong"
    elif adx_val >= 35:
        strength = "strong"
    elif adx_val >= 25:
        strength = "moderate"
    elif adx_val >= 15:
        strength = "weak"
    else:
        strength = "no_trend"

    if plus_di > minus_di * 1.05:
        direction = "uptrend"
    elif minus_di > plus_di * 1.05:
        direction = "downtrend"
    else:
        direction = "ranging"

    return {
        "adx": _round(adx_val),
        "plus_di": _round(plus_di),
        "minus_di": _round(minus_di),
        "trend_strength": strength,
        "trend_direction": direction,
    }


def calculate_obv(
    price_history: Sequence[float],
    volume_history: Sequence[int],
) -> dict[str, Any]:
    """Calculate On-Balance Volume (volume indicator)."""
    closes = [float(p) for p in price_history]
    volumes = [float(v) for v in volume_history]
    if not closes or not volumes or len(closes) != len(volumes):
        return {
            "obv": 0.0,
            "obv_trend": "flat",
            "volume_confirmation": "divergence",
        }
    obv_series = OnBalanceVolumeIndicator(
        _series(closes), _series(volumes)
    ).on_balance_volume()
    obv_series = obv_series.astype(float).dropna()
    if len(obv_series) < 3:
        return {
            "obv": 0.0,
            "obv_trend": "flat",
            "volume_confirmation": "divergence",
        }

    last = float(obv_series.iloc[-1])
    window = obv_series.tail(min(20, len(obv_series)))
    slope = float(window.diff().mean())
    lookback_price = (
        (closes[-1] - closes[-min(20, len(closes))])
        / (closes[-min(20, len(closes))] or 1.0)
    )

    if slope > 0.005 * (abs(last) + 1):
        obv_trend = "increasing"
    elif slope < -0.005 * (abs(last) + 1):
        obv_trend = "decreasing"
    else:
        obv_trend = "flat"

    if (obv_trend == "increasing" and lookback_price > 0) or (
        obv_trend == "decreasing" and lookback_price < 0
    ):
        confirmation = "strong_confirmation"
    elif obv_trend == "flat":
        confirmation = "moderate_confirmation"
    else:
        confirmation = "divergence"

    return {
        "obv": _round(last),
        "obv_trend": obv_trend,
        "volume_confirmation": confirmation,
    }
