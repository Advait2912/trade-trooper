"""Price levels, trend and chart-pattern tools (Phase 1 - Historical Agent).

Heuristic, deterministic implementations: swing-point detection with local
extrema, level clustering by touch count, linear-regression trend analysis,
and the "core 4" chart patterns (double top/bottom, head & shoulders + its
inverse).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

_SWING_WINDOW = {"high": 3, "medium": 5, "low": 7}


# ---------------------------------------------------------------------------
# Swing points
# ---------------------------------------------------------------------------
def _swing_indices(
    highs: Sequence[float], lows: Sequence[float], window: int
) -> tuple[list[int], list[int]]:
    """Return indices of confirmed swing highs and swing lows."""
    n = len(highs)
    swing_highs: list[int] = []
    swing_lows: list[int] = []
    for i in range(window, n - window):
        hi = highs[i]
        lo = lows[i]
        if all(hi >= highs[i - j] for j in range(1, window + 1)) and all(
            hi >= highs[i + j] for j in range(1, window + 1)
        ):
            swing_highs.append(i)
        if all(lo <= lows[i - j] for j in range(1, window + 1)) and all(
            lo <= lows[i + j] for j in range(1, window + 1)
        ):
            swing_lows.append(i)
    return swing_highs, swing_lows


def _cluster_levels(levels: list[float], tolerance: float) -> list[dict[str, Any]]:
    """Group nearby levels; return {level, touches, strength} top-heavy."""
    if not levels:
        return []
    groups: list[list[float]] = []
    for value in sorted(levels):
        placed = False
        for g in groups:
            if abs((sum(g) / len(g)) - value) / (sum(g) / len(g) or 1.0) <= tolerance:
                g.append(value)
                placed = True
                break
        if not placed:
            groups.append([value])

    out: list[dict[str, Any]] = []
    for g in groups:
        level = sum(g) / len(g)
        touches = len(g)
        strength = "strong" if touches >= 3 else ("moderate" if touches >= 2 else "weak")
        out.append(
            {"level": round(level, 4), "touches": touches, "strength": strength}
        )
    return sorted(out, key=lambda r: (-r["touches"], -r["level"]))[:6]


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
def identify_support_resistance(
    price_history: Sequence[dict[str, Any]],
    sensitivity: str = "medium",
) -> dict[str, Any]:
    """Identify key support and resistance levels from swing points."""
    window = _SWING_WINDOW.get(sensitivity, 5)
    highs = [float(b["high"]) for b in price_history]
    lows = [float(b["low"]) for b in price_history]
    if len(highs) < 2 * window + 1 or not highs:
        return {"resistance_levels": [], "support_levels": []}

    swing_highs, swing_lows = _swing_indices(highs, lows, window)
    # Recent levels get a modest bump so ties prefer newer prices.
    tolerance = 0.01  # 1% clustering tolerance

    resistance = _cluster_levels([highs[i] for i in swing_highs], tolerance)
    support = _cluster_levels([lows[i] for i in swing_lows], tolerance)

    return {"resistance_levels": resistance, "support_levels": support}


def identify_trend(
    price_history: Sequence[float],
    lookback_period: int = 60,
) -> dict[str, Any]:
    """Identify current trend direction and strength via linear regression."""
    closes = [float(p) for p in price_history]
    lookback = min(lookback_period, len(closes))
    if lookback < 5 or any(c <= 0 for c in closes):
        return {
            "trend": "ranging",
            "trend_class": "neutral",  # Fix: added to prevent KeyError in HistoricalAgent
            "trend_strength": 0.0,
            "angle": 0.0,
            "days_in_trend": 0,
        }

    window = closes[-lookback:]
    y = [math.log(c) for c in window]
    n = len(y)
    x_mean = (n - 1) / 2.0
    y_mean = sum(y) / n
    cov = sum((i - x_mean) * (yv - y_mean) for i, yv in enumerate(y))
    var = sum((i - x_mean) ** 2 for i in range(n))
    slope = cov / var if var else 0.0
    r2 = (cov * cov / (var * sum((yv - y_mean) ** 2 for yv in y)) if var else 0.0)

    angle = math.degrees(math.atan(slope))
    strength = max(0.0, min(1.0, r2))

    daily = slope  # log-return fraction per bar
    if daily > 0.004:
        trend = "strong_uptrend"
    elif daily > 0.001:
        trend = "uptrend"
    elif daily > 0.0003:
        trend = "weak_uptrend"
    elif daily < -0.004:
        trend = "strong_downtrend"
    elif daily < -0.001:
        trend = "downtrend"
    elif daily < -0.0003:
        trend = "weak_downtrend"
    else:
        trend = "ranging"

    up = daily >= 0
    days = 0
    for i in range(n - 1, 0, -1):
        move = y[i] - y[i - 1]
        if (up and move >= 0) or (not up and move <= 0):
            days += 1
        elif abs(move) < 1e-6:
            days += 1
        else:
            break

    return {
        "trend": trend,
        "trend_class": _trend_class(trend),
        "trend_strength": round(strength, 4),
        "angle": round(angle, 2),
        "days_in_trend": days,
    }


def _trend_class(trend: str) -> str:
    if "up" in trend:
        return "bullish"
    if "down" in trend:
        return "bearish"
    return "neutral"


def detect_chart_patterns(
    price_history: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Detect core chart patterns: double top/bottom, H&S (+ inverse).

    Returns ``patterns_found`` (possibly empty); unknown patterns simply do
    not appear in the list (matching the "none" semantics of the contract).
    """
    highs = [float(b["high"]) for b in price_history]
    lows = [float(b["low"]) for b in price_history]
    if len(highs) < 15:
        return {"patterns_found": []}

    swing_highs, swing_lows = _swing_indices(highs, lows, 3)
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return {"patterns_found": []}

    found: list[dict[str, Any]] = []
    n = len(highs)

    def append(pattern: str, conf: float, target: float, end_idx: int) -> None:
        days_left = max(0, n - 1 - end_idx)
        if conf < 0.45:
            return
        found.append(
            {
                "pattern": pattern,
                "confidence": round(conf, 2),
                "breakout_target": round(target, 4),
                "time_to_completion_days": days_left,
            }
        )

    for i in range(len(swing_highs) - 1):
        a, c = swing_highs[i], swing_highs[i + 1]
        ha, hc = highs[a], highs[c]
        if ha == 0 or hc == 0:
            continue
        same_ac = abs(ha - hc) / ha
        # double top: two similar swing highs with a trough between them.
        if same_ac <= 0.015 and c - a >= 5:
            trough = min(lows[a:c])
            peak = max(ha, hc)
            target = peak - (peak - trough)
            conf = 0.9 - same_ac * 30 + min(0.08, (c - a) / 100)
            append("double_top", conf, target, c)

    for i in range(len(swing_highs) - 2):
        a, b, c = swing_highs[i : i + 3]
        ha, hb, hc = highs[a], highs[b], highs[c]
        if ha == 0 or hb == 0 or hc == 0:
            continue
        same_ac = abs(ha - hc) / ha
        # head & shoulders: b (head) clearly above both shoulders.
        if hb >= ha * 1.015 and hb >= hc * 1.015:
            trough1 = min(lows[a:b])
            trough2 = min(lows[b:c])
            neckline = (trough1 + trough2) / 2
            target = neckline - (hb - neckline)
            conf = 0.9 - same_ac * 10
            append("head_shoulders", conf, target, c)

    for i in range(len(swing_lows) - 1):
        a, c = swing_lows[i], swing_lows[i + 1]
        la, lc = lows[a], lows[c]
        if la == 0 or lc == 0:
            continue
        same_ac = abs(la - lc) / la
        # double bottom: two similar swing lows with a peak between them.
        if same_ac <= 0.015 and c - a >= 5:
            peak = max(highs[a:c])
            trough = min(la, lc)
            target = trough + (peak - trough)
            conf = 0.9 - same_ac * 30 + min(0.08, (c - a) / 100)
            append("double_bottom", conf, target, c)

    for i in range(len(swing_lows) - 2):
        a, b, c = swing_lows[i : i + 3]
        la, lb, lc = lows[a], lows[b], lows[c]
        if la == 0 or lb == 0 or lc == 0:
            continue
        same_ac = abs(la - lc) / la
        # inverse head & shoulders: center trough clearly below both shoulders.
        if lb <= la * 0.985 and lb <= lc * 0.985:
            peak1 = max(highs[a:b])
            peak2 = max(highs[b:c])
            neckline = (peak1 + peak2) / 2
            target = neckline + (neckline - lb)
            conf = 0.9 - same_ac * 10
            append("inverse_head_shoulders", conf, target, c)

    found.sort(key=lambda p: -p["confidence"])
    return {"patterns_found": found}
