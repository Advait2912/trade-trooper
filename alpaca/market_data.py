"""Alpaca market data — snapshot, daily bars, and indicator computation."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, time, timezone
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from alpaca.client import AlpacaClient
from schemas.common import Trend
from schemas.market import MarketData
from tools.historical import indicators as ind

log = logging.getLogger("market_intel_agent.market_data")

try:
    _NY: Optional[ZoneInfo] = ZoneInfo("America/New_York")
except ZoneInfoNotFoundError:  # pragma: no cover - no tz database (e.g. bare Windows)
    _NY = None

_MARKET_OPEN = time(9, 30)
_MARKET_CLOSE = time(16, 0)

# A quote/trade older than this (hours) is considered stale.
_STALE_HOURS = 24


async def get_snapshot(client: AlpacaClient, symbol: str) -> Optional[Dict[str, Any]]:
    payload = await client.get_json(f"/v2/stocks/{symbol}/snapshot")
    if not isinstance(payload, dict) or not payload:
        return None
    return payload


async def get_daily_bars(
    client: AlpacaClient, symbol: str, limit: int = 100, feed: str = "iex"
) -> List[Dict[str, Any]]:
    params = {
        "timeframe": "1Day",
        "limit": limit,
        "adjustment": "raw",
        "feed": feed,
        "sort": "asc",
    }
    payload = await client.get_json(f"/v2/stocks/{symbol}/bars", params=params)
    bars = payload.get("bars", []) if isinstance(payload, dict) else []
    return [b for b in bars if isinstance(b, dict)]


def compute_market_data(
    snapshot: Optional[Dict[str, Any]],
    bars: List[Dict[str, Any]],
    symbol: str,
    now: Optional[datetime] = None,
) -> MarketData:
    """Compute MarketData deterministically from snapshot + daily bars."""
    now = now or datetime.now(timezone.utc)

    md = MarketData()

    closes = [float(b["c"]) for b in bars if _num(b.get("c"))]
    highs = [float(b["h"]) for b in bars if _num(b.get("h"))]
    lows = [float(b["l"]) for b in bars if _num(b.get("l"))]
    volumes = [float(b.get("v") or 0) for b in bars]

    # Latest price / volume / previous close.
    latest_trade = snapshot.get("latestTrade") if snapshot else None
    prev_daily = snapshot.get("prevDailyBar") if snapshot else None
    daily_bar = snapshot.get("dailyBar") if snapshot else None

    if latest_trade and _num(latest_trade.get("p")):
        md.price = float(latest_trade["p"])
    elif closes:
        md.price = closes[-1]

    if latest_trade and _num(latest_trade.get("s")):
        md.volume = float(latest_trade["s"])
    elif volumes:
        md.volume = volumes[-1]

    if prev_daily and _num(prev_daily.get("c")):
        md.previous_close = float(prev_daily["c"])
    elif len(closes) >= 2:
        md.previous_close = closes[-2]

    as_of = ""
    if latest_trade and latest_trade.get("t"):
        as_of = latest_trade["t"]
    elif daily_bar and daily_bar.get("t"):
        as_of = daily_bar["t"]
    elif bars:
        as_of = bars[-1].get("t", "")
    md.as_of = as_of

    # Indicators (all require sufficient history; None -> 0.0).
    md.return_1d = ind.return_over(closes, 1) or 0.0
    md.return_5d = ind.return_over(closes, 5) or 0.0
    md.return_20d = ind.return_over(closes, 20) or 0.0
    md.sma20 = ind.sma(closes, 20) or 0.0
    md.sma50 = ind.sma(closes, 50) or 0.0
    md.rsi14 = ind.rsi(closes, 14) or 0.0
    md.atr14 = ind.atr(highs, lows, closes, 14) or 0.0
    md.volume_vs_average = ind.volume_vs_average(volumes, 20) or 0.0
    md.volatility = ind.volatility(closes, 20) or 0.0

    # Freshness / market state.
    as_of_dt = _parse_time(as_of)
    if as_of_dt is not None:
        age_hours = (now - as_of_dt).total_seconds() / 3600.0
        md.stale = age_hours > _STALE_HOURS
        md.market_closed = _market_closed_now(as_of_dt)

    log.debug("market data computed for %s: %s", symbol, md.model_dump())
    return md


def suggest_trend(md: MarketData) -> Trend:
    """Deterministic default trend from price vs moving averages and RSI.

    Used as a fallback; the LLM may refine this in the final synthesis.
    """
    if md.price <= 0:
        return Trend.UNCERTAIN
    if md.sma20 and md.sma50:
        above = md.price > md.sma20 > md.sma50
        below = md.price < md.sma20 < md.sma50
    elif md.sma20:
        above = md.price > md.sma20
        below = md.price < md.sma20
    else:
        above = below = False

    if above and md.rsi14 >= 50:
        return Trend.BULLISH
    if below and md.rsi14 <= 50:
        return Trend.BEARISH
    return Trend.NEUTRAL


async def fetch_market_data(
    client: AlpacaClient, symbol: str, feed: str = "iex"
) -> MarketData:
    """Fetch snapshot and daily bars concurrently, then compute indicators."""
    snapshot, bars = await asyncio.gather(
        get_snapshot(client, symbol),
        get_daily_bars(client, symbol, limit=100, feed=feed),
    )
    return compute_market_data(snapshot, bars, symbol)


def _num(value: Any) -> bool:
    try:
        return value is not None and float(value) == float(value)  # excludes NaN
    except (TypeError, ValueError):
        return False


def _parse_time(value: str) -> Optional[datetime]:
    if not value:
        return None
    try:
        normalized = value
        if normalized.endswith("Z"):
            normalized = normalized[:-1] + "+00:00"
        dt = datetime.fromisoformat(normalized)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except (ValueError, TypeError):
        return None


def _market_closed_now(as_of: datetime) -> bool:
    if _NY is not None:
        local = as_of.astimezone(_NY)
        if local.weekday() >= 5:  # weekend
            return True
        t = local.time()
        return t < _MARKET_OPEN or t > _MARKET_CLOSE

    # Fallback without a tz database: approximate the US session in UTC.
    # 9:30–16:00 ET maps to 13:30–20:00 UTC (EST) / 14:30–21:00 UTC (EDT).
    local = as_of.astimezone(timezone.utc)
    if local.weekday() >= 5:
        return True
    t = local.time()
    return not (time(13, 30) <= t <= time(21, 0))
