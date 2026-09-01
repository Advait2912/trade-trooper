"""Market data schemas (Phase 1 - Market Data Agent)."""

from __future__ import annotations

from pydantic import BaseModel


class MarketData(BaseModel):
    """Deterministically computed market snapshot + indicators."""

    price: float = 0.0
    previous_close: float = 0.0
    volume: float = 0.0
    return_1d: float = 0.0
    return_5d: float = 0.0
    return_20d: float = 0.0
    sma20: float = 0.0
    sma50: float = 0.0
    rsi14: float = 0.0
    atr14: float = 0.0
    volume_vs_average: float = 0.0
    volatility: float = 0.0
    as_of: str = ""
    stale: bool = False
    market_closed: bool = False
