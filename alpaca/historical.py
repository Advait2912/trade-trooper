"""Alpaca data layer for the Historical Data Agent.

Wraps the raw ``AlpacaClient.get_json`` to provide:

- ``get_price_history``  — multi-timeframe OHLCV bars (``/v2/stocks/{symbol}/bars``)
- ``get_dividends_history`` — cash dividends via the corporate-actions endpoint
- ``get_earnings_history`` — Alpaca-unavailable stub (returns [])
- ``get_volatility_history`` — rolling realized volatility computed from bars

All functions return typed Pydantic models and raise ``AlpacaError`` subclasses
on API failures (callers degrade gracefully).
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from alpaca.client import AlpacaClient
from schemas.historical import (
    DividendRecord,
    EarningsRecord,
    PriceBar,
    VolatilityPoint,
)

log = logging.getLogger("market_intel_agent.alpaca.historical")

INTERVAL_TIMEframe = {
    "1m": "1Min",
    "5m": "5Min",
    "15m": "15Min",
    "1h": "1Hour",
    "1d": "1Day",
    "1w": "1Week",
    "1mo": "1Month",
}

_BARS_PER_DAY = {
    "1m": 390,
    "5m": 78,
    "15m": 26,
    "1h": 7,
    "1d": 1,
    "1w": 1,
    "1mo": 1,
}

_MAX_BARS = 10000


def _limit_for(interval: str, days_back: int) -> int:
    """Map (interval, days_back) to a safe bars limit."""
    if interval == "1w":
        return max(1, min(_MAX_BARS, days_back // 5 + 2))
    if interval == "1mo":
        return max(1, min(_MAX_BARS, days_back // 21 + 2))
    return max(1, min(_MAX_BARS, days_back * _BARS_PER_DAY.get(interval, 1)))


async def get_price_history(
    client: AlpacaClient,
    symbol: str,
    days_back: int = 60,
    interval: str = "1d",
    feed: str = "iex",
    end_date: date | None = None,
    start_date: date | None = None,
) -> list[PriceBar]:
    """Retrieve historical OHLCV bars for a symbol.

    ``end_date`` overrides the "now" reference so a historical window (e.g. a
    2024-only backtest) can be fetched; ``start_date`` sets the fetch start
    explicitly (when omitted, ``end - days_back*2`` calendar days is used).
    Bars are returned ascending in time.

    Raises ``KeyError`` for an unsupported interval.
    """
    if interval not in INTERVAL_TIMEframe:
        raise KeyError(
            f"Unsupported interval {interval!r}; "
            f"expected one of {sorted(INTERVAL_TIMEframe)}"
        )
    end = end_date or date.today()
    if start_date is not None:
        start = start_date
    else:
        start = end - timedelta(days=max(days_back * 2, 10))
    params = {
        "timeframe": INTERVAL_TIMEframe[interval],
        "limit": _limit_for(interval, days_back),
        "adjustment": "split",
        "feed": feed,
        "sort": "desc",
        "start": start.isoformat(),
        "end": end.isoformat(),
    }
    payload = await client.get_json(f"/v2/stocks/{symbol}/bars", params=params)
    bars = (payload.get("bars") or []) if isinstance(payload, dict) else []
    parsed = [bar for bar in (parse_price_bar(b) for b in bars) if bar is not None]
    return list(reversed(parsed))  # ascending chronological order


def parse_price_bar(row: Any) -> PriceBar | None:
    """Normalize one raw bar dict into a PriceBar (None if unparseable)."""
    if not isinstance(row, dict):
        return None
    keys = {k.lower(): v for k, v in row.items()}

    def num(key: str) -> float | None:
        val = keys.get(key)
        try:
            return float(val) if val is not None else None
        except (TypeError, ValueError):
            return None

    c = num("c")
    o, h, l = num("o"), num("h"), num("l")
    if c is None:
        return None
    v = num("v") or 0
    ts = keys.get("t", "")
    if isinstance(ts, (int, float)):
        ts = pd.to_datetime(ts, unit="ms").isoformat()
    elif isinstance(ts, str):
        ts = ts.replace("Z", "+00:00")
    return PriceBar(
        date=str(ts),
        open=o if o is not None else c,
        high=h if h is not None else c,
        low=l if l is not None else c,
        close=c,
        volume=int(v),
    )


async def get_dividends_history(
    client: AlpacaClient,
    symbol: str,
    years_back: int = 5,
    feed: str = "iex",
) -> list[DividendRecord]:
    """Retrieve cash dividends from the corporate-actions endpoint."""
    today = date.today()
    since = (today - timedelta(days=years_back * 366)).isoformat()
    params = {
        "types": "CASH_DIVIDEND",
        "since": since,
        "until": today.isoformat(),
        "feed": feed,
    }
    payload = await client.get_json(
        f"/v2/stocks/{symbol}/corporate_actions", params=params
    )
    return [d for d in (parse_dividend(e) for e in _iter_actions(payload)) if d]


def _iter_actions(
    payload: Any,
) -> Any:
    """Yield action entries from the (deeply nested) corporate-actions payload.

    The response shape nests symbol -> asset_id -> [actions]; walker yields
    any dict entry that looks like an action (has a 'type' key).
    """
    if not isinstance(payload, dict):
        return iter([])
    outer = payload.get("corporate_actions", payload)
    if not isinstance(outer, dict):
        return iter([])

    def walk(node: Any) -> Any:
        if isinstance(node, list):
            for item in node:
                yield from walk(item)
        elif isinstance(node, dict):
            if "type" in node and any(k in node for k in ("amount", "ex_date")):
                yield node
            else:
                for value in node.values():
                    yield from walk(value)

    return walk(outer)


def parse_dividend(entry: Any) -> DividendRecord | None:
    """Normalize a raw corporate-action entry into a DividendRecord."""
    if not isinstance(entry, dict):
        return None
    kind = str(entry.get("type", "")).lower()
    if "dividend" not in kind:
        return None
    amount = entry.get("amount") or entry.get("dividend_amount")
    try:
        amount_f = float(amount) if amount is not None else None
    except (TypeError, ValueError):
        amount_f = None
    if amount_f is None:
        return None
    ex_date = str(entry.get("ex_date") or "")
    return DividendRecord(
        date=str(entry.get("payable_date") or entry.get("declared_date") or ex_date),
        dividend_amount=round(amount_f, 4),
        ex_date=ex_date,
    )


async def get_earnings_history(
    client: AlpacaClient,
    symbol: str,
    quarters: int = 8,
) -> list[EarningsRecord]:
    """Earnings history.

    Alpaca provides no earnings/fundamentals data, so this is a documented
    stub that always returns []. Earnings-*like* signals (gaps, volume
    surges) are inferred from bars by ``analyze_gaps`` and
    ``identify_trading_events`` instead.
    """
    return []


async def get_volatility_history(
    client: AlpacaClient,
    symbol: str,
    days_back: int = 252,
    period: int = 20,
    feed: str = "iex",
) -> list[VolatilityPoint]:
    """Compute rolling realized volatility from daily bars.

    Returns one point per trading day: annualized std of log returns over
    ``period`` days (``realized_vol``/``rolling_vol_20d``) and a 60-day
    window (``rolling_vol_60d``).
    """
    bars = await get_price_history(
        client, symbol, days_back=days_back + 60, interval="1d", feed=feed
    )
    if len(bars) < 2:
        return []

    closes = pd.Series([b.close for b in bars], dtype="float64")
    logs = np_log_returns(closes)
    realized = (logs.rolling(window=period).std() * _SQRT_252) * 100
    vol_60 = (logs.rolling(window=60).std() * _SQRT_252) * 100

    points: list[VolatilityPoint] = []
    for i, bar in enumerate(bars):
        rv = realized.iloc[i]
        points.append(
            VolatilityPoint(
                date=bar.date,
                realized_vol=_none_to_zero(rv),
                rolling_vol_20d=_none_to_zero(rv),
                rolling_vol_60d=_none_to_zero(vol_60.iloc[i]),
            )
        )
    return points


_SQRT_252 = 252.0 ** 0.5


def np_log_returns(prices: pd.Series) -> pd.Series:
    """Log returns of a price series (NaN for the first point / non-positive prices)."""
    shifted = prices.shift(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        logr = np.log(prices / shifted)
    return logr.where((prices > 0) & (shifted > 0))


def _none_to_zero(value: float) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if pd.isna(f) else round(f, 4)
