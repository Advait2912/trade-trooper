"""Deterministic technical indicators.

All calculations are done locally in pure Python. The LLM is never asked to
compute numbers; it only receives the results.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence


def sma(values: Sequence[float], period: int) -> Optional[float]:
    """Simple moving average over the most recent `period` values."""
    if period <= 0 or len(values) < period:
        return None
    window = values[-period:]
    return sum(window) / period


def rsi(closes: Sequence[float], period: int = 14) -> Optional[float]:
    """Wilder's Relative Strength Index.

    Returns None when there is not enough data (need at least `period + 1`
    closes to compute a single average gain/loss).
    """
    if period <= 0 or len(closes) < period + 1:
        return None

    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [d if d > 0 else 0.0 for d in deltas]
    losses = [-d if d < 0 else 0.0 for d in deltas]

    # Seed with a simple average over the first `period` changes.
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    # Wilder smoothing for the remainder.
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0.0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def true_ranges(
    highs: Sequence[float], lows: Sequence[float], closes: Sequence[float]
) -> List[float]:
    """True range for each bar, using the prior close as reference."""
    n = len(highs)
    trs: List[float] = []
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
) -> Optional[float]:
    """Wilder's Average True Range. Requires at least `period + 1` bars."""
    if period <= 0 or len(highs) < period + 1:
        return None
    trs = true_ranges(highs, lows, closes)
    avg = sum(trs[:period]) / period
    for i in range(period, len(trs)):
        avg = (avg * (period - 1) + trs[i]) / period
    return avg


def return_over(closes: Sequence[float], days: int) -> Optional[float]:
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


def volume_vs_average(
    volumes: Sequence[float], period: int = 20
) -> Optional[float]:
    """Latest volume divided by the trailing average volume."""
    if period <= 0 or len(volumes) < period or not volumes[-1]:
        return None
    avg = sum(volumes[-period:]) / period
    if avg == 0:
        return None
    return volumes[-1] / avg


def volatility(closes: Sequence[float], period: int = 20) -> Optional[float]:
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
