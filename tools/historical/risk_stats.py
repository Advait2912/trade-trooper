"""Risk statistics tools (Phase 1 - Historical Agent).

Drawdown + recovery, gap analysis, Value at Risk / CVaR and multi-period
returns.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import date
from typing import Any

import numpy as np


def calculate_drawdown(price_history: Sequence[float]) -> dict[str, Any]:
    """Calculate maximum drawdown and recovery time (bar counts)."""
    prices = [float(p) for p in price_history]
    if not prices:
        return {
            "max_drawdown": 0.0,
            "current_drawdown": 0.0,
            "recovery_time_days": 0,
            "risk_level": "low",
        }

    peak = prices[0]
    peak_idx = 0
    max_dd = 0.0
    max_dd_end = -1
    running_peak = prices[0]
    for i, p in enumerate(prices):
        if p >= running_peak:
            running_peak = p
            peak_idx = i
        dd = (p - running_peak) / running_peak
        if dd < max_dd:
            max_dd = dd
            max_dd_end = i
            peak = running_peak

    current_dd = (prices[-1] - running_peak) / running_peak
    if max_dd_end >= 0:
        recovery = peak_idx  # notional; recomputed below properly
        recovered = False
        for j in range(max_dd_end + 1, len(prices)):
            if prices[j] >= peak * 0.999:
                recovery = j
                recovered = True
                break
        if not recovered:
            recovery = len(prices) - 1 - max_dd_end
    else:
        recovery = 0

    max_pct = abs(max_dd) * 100.0
    if max_pct >= 40:
        risk = "extreme"
    elif max_pct >= 25:
        risk = "high"
    elif max_pct >= 10:
        risk = "moderate"
    else:
        risk = "low"

    return {
        "max_drawdown": round(max_pct, 4),
        "current_drawdown": round(current_dd * 100.0, 4),
        "recovery_time_days": max(0, int(recovery)),
        "risk_level": risk,
    }


def analyze_gaps(
    price_history: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Analyze price gaps in history (close -> next open)."""
    if len(price_history) < 3:
        return {"gaps": [], "gap_frequency": "rare", "avg_gap_size": 0.0}

    closes = [float(b["close"]) for b in price_history]
    opens: list[float] = []
    for b in price_history[1:]:
        next_open = b.get("next_open", b.get("open", 0.0)) or 0.0
        opens.append(float(next_open))
    closes_shifted = closes[:-1]

    gaps = []
    for i, (prev_close, next_open) in enumerate(zip(closes_shifted, opens)):
        if prev_close <= 0:
            continue
        gap_size = next_open - prev_close
        gap_pct = gap_size / prev_close * 100.0
        if abs(gap_pct) < 0.001:
            continue
        filled = _gap_filled(closes[i], prev_close, gap_pct, price_history[i + 2 :])
        gaps.append(
            {
                "date": price_history[i + 1].get("date", ""),
                "gap_size": round(gap_size, 4),
                "gap_size_percent": round(gap_pct, 4),
                "filled": filled,
            }
        )

    relevant = [g for g in gaps if abs(g["gap_size_percent"]) >= 0.5]
    ratio = len(relevant) / len(gaps) if gaps else 0.0
    if ratio >= 0.25:
        freq = "very_frequent"
    elif ratio >= 0.12:
        freq = "frequent"
    elif ratio >= 0.05:
        freq = "occasional"
    else:
        freq = "rare"
    avg = (
        sum(abs(g["gap_size_percent"]) for g in relevant) / len(relevant)
        if relevant
        else 0.0
    )
    return {"gaps": gaps, "gap_frequency": freq, "avg_gap_size": round(avg, 4)}


def _gap_filled(
    first_close_after: float,
    prev_close: float,
    gap_pct: float,
    later_bars: Sequence[dict[str, Any]],
) -> bool:
    """A gap is 'filled' if price retraces through the pre-gap close."""
    if not later_bars:
        return False
    if gap_pct > 0:
        return any(float(b["low"]) <= prev_close for b in later_bars if "low" in b) or any(
            float(b["close"]) <= prev_close for b in later_bars
        )
    return any(float(b["high"]) >= prev_close for b in later_bars if "high" in b) or any(
        float(b["close"]) >= prev_close for b in later_bars
    )


def calculate_value_at_risk(
    returns_history: Sequence[float],
    confidence_level: float = 95,
) -> dict[str, Any]:
    """Calculate VaR % and CVaR % (expected loss beyond VaR)."""
    returns = np.array([float(r) for r in returns_history], dtype=float)
    returns = returns[~np.isnan(returns)]
    if len(returns) < 10:
        return {"var": 0.0, "cvar": 0.0}

    alpha = 1.0 - confidence_level / 100.0
    var_pos = max(1, int(math.ceil(alpha * len(returns))))
    var = float(np.sort(returns)[var_pos - 1]) * 100.0 if math.isclose(alpha, 0) or var_pos <= len(returns) else 0.0
    tail = returns[returns <= var / 100.0]
    cvar = float(np.mean(tail)) * 100.0 if len(tail) else var

    return {"var": round(var, 4), "cvar": round(cvar, 4)}


def calculate_returns(
    price_history: Sequence[float],
    periods: Sequence[int] = (5, 20, 60, 252),
) -> dict[str, Any]:
    """Calculate returns over various periods (as fractions of price)."""
    closes = [float(p) for p in price_history]
    labels = ["return_1w", "return_1m", "return_3m", "return_1y"]
    out: dict[str, Any] = {}
    for label, days in zip(labels, periods[: len(labels)]):
        value = _return_over(closes, int(days))
        out[label] = round(value, 4) if value is not None else 0.0

    # Approximate YTD: estimate trading days elapsed since Jan 1.
    day_of_year = date.today().timetuple().tm_yday
    ytd_bars = max(21, min(252, int(252 * day_of_year / 365)))
    ytd = _return_over(closes, ytd_bars)
    out["ytd_return"] = round(ytd, 4) if ytd is not None else 0.0
    return out


def _return_over(closes: list[float], days: int) -> float | None:
    if days <= 0 or len(closes) < days + 1:
        return None
    start = closes[-(days + 1)]
    if start == 0:
        return None
    return (closes[-1] / start) - 1.0
