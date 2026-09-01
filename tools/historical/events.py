"""Trading event detection (Phase 1 - Historical Agent).

Infers significant events (gaps, vol spikes, volume surges, reversals) purely
from Alpaca bars. Alpaca does not expose earnings dates, so earnings-*like*
events are inferred from the price/volume signature they typically leave.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from tools.historical.risk_stats import analyze_gaps


def identify_trading_events(
    price_history: Sequence[dict[str, Any]],
    volume_history: Sequence[int],
    earnings_dates: Sequence[str] = (),
) -> dict[str, Any]:
    """Identify significant trading events from bars (gap/vol/volume signatures)."""
    if len(price_history) < 25:
        return {"events": []}

    closes = [float(b["close"]) for b in price_history]
    highs = [float(b["high"]) for b in price_history]
    lows = [float(b["low"]) for b in price_history]
    volumes = [float(v) for v in volume_history]

    events: list[dict[str, Any]] = []

    # --- gaps -----------------------------------------------------------------
    gap_info = analyze_gaps(
        [{"date": b.get("date", ""), **b} for b in price_history]
    )
    for g in gap_info["gaps"]:
        if abs(g["gap_size_percent"]) >= 1.0:
            events.append(
                {
                    "date": g["date"],
                    "event_type": "gap",
                    "magnitude": abs(g["gap_size_percent"]),
                    "impact_on_options": (
                        "elevated" if abs(g["gap_size_percent"]) >= 3.0 else "moderate"
                    ),
                }
            )

    # --- volume surges + vol spikes ------------------------------------------
    vol_ratio = _volume_ratio(volumes)
    if volumes and len(volumes) >= 20 and vol_ratio >= 2.0:
        events.append(
            {
                "date": str(price_history[-1].get("date", "")),
                "event_type": "volume_surge",
                "magnitude": round(vol_ratio, 2),
                "impact_on_options": "moderate" if vol_ratio < 3.0 else "elevated",
            }
        )

    short_vol = _annualized_percent(closes[-20:])
    long_vol = _annualized_percent(closes[-60:])
    if short_vol and long_vol and long_vol > 0 and short_vol / long_vol >= 1.75:
        events.append(
            {
                "date": str(price_history[-1].get("date", "")),
                "event_type": "vol_spike",
                "magnitude": round(short_vol / long_vol, 2),
                "impact_on_options": "elevated",
            }
        )

    # --- reversals (large move day followed by opposite move) ----------------
    if len(closes) >= 5:
        moves = [
            (closes[i] / closes[i - 1] - 1.0) for i in range(1, len(closes))
        ]
        for i in range(1, len(moves)):
            if abs(moves[i - 1]) >= 0.03 and moves[i] * moves[i - 1] < 0:
                if abs(moves[i]) >= 0.015:
                    events.append(
                        {
                            "date": str(price_history[i + 1].get("date", "")),
                            "event_type": "reversal",
                            "magnitude": round(abs(moves[i]) * 100, 2),
                            "impact_on_options": "moderate",
                        }
                    )
                if len(events) > 25:
                    break

    # --- earnings signatures ---------------------------------------------------
    for d in earnings_dates:
        events.append(
            {
                "date": d,
                "event_type": "earnings",
                "magnitude": 0.0,
                "impact_on_options": "unknown (no Alpaca earnings data)",
            }
        )

    return {"events": events[:30]}


def _volume_ratio(volumes: list[float]) -> float:
    if len(volumes) < 21:
        return 0.0
    recent = volumes[-1]
    avg = sum(volumes[-21:-1]) / 20
    return recent / avg if avg > 0 else 0.0


def _annualized_percent(closes: list[float]) -> float | None:
    if len(closes) < 3:
        return None
    logr = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes)) if closes[i] > 0 and closes[i - 1] > 0]
    if len(logr) < 2:
        return None
    mean = sum(logr) / len(logr)
    var = sum((x - mean) ** 2 for x in logr) / (len(logr) - 1)
    return math.sqrt(var) * math.sqrt(252) * 100.0
