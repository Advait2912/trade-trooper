"""Historical data-fetch tools (Phase 1 - Historical Agent).

Contract layer: returns the exact JSON shapes declared in the tool specs
(plain dictionaries, no Pydantic). Fetching/parsing lives in
``alpaca.historical``.
"""

from __future__ import annotations

from typing import Any

from alpaca.client import AlpacaClient
from alpaca.historical import (
    get_dividends_history as _fetch_dividends,
)
from alpaca.historical import (
    get_earnings_history as _fetch_earnings,
)
from alpaca.historical import (
    get_price_history as _fetch_prices,
)
from alpaca.historical import (
    get_volatility_history as _fetch_volatility,
)


async def get_price_history(
    client: AlpacaClient,
    symbol: str,
    days_back: int = 60,
    interval: str = "1d",
    feed: str = "iex",
) -> dict[str, Any]:
    """Retrieve historical OHLCV bars (contract: ``{symbol, prices: [...]}``)."""
    bars = await _fetch_prices(client, symbol, days_back, interval, feed)
    return {
        "symbol": symbol,
        "prices": [
            {
                "date": b.date,
                "open": b.open,
                "high": b.high,
                "low": b.low,
                "close": b.close,
                "volume": b.volume,
            }
            for b in bars
        ],
    }


async def get_dividends_history(
    client: AlpacaClient,
    symbol: str,
    years_back: int = 5,
    feed: str = "iex",
) -> list[dict[str, Any]]:
    """Retrieve historical dividend data (contract: list of records)."""
    records = await _fetch_dividends(client, symbol, years_back, feed)
    return [
        {
            "date": r.date,
            "dividend_amount": r.dividend_amount,
            "ex_date": r.ex_date,
        }
        for r in records
    ]


async def get_earnings_history(
    client: AlpacaClient,
    symbol: str,
    quarters: int = 8,
) -> list[dict[str, Any]]:
    """Get historical earnings data (Alpaca-unavailable stub: always [])."""
    records = await _fetch_earnings(client, symbol, quarters)
    return [
        {
            "date": r.date,
            "eps_actual": r.eps_actual,
            "eps_estimate": r.eps_estimate,
            "revenue": r.revenue,
            "surprise_percent": r.surprise_percent,
        }
        for r in records
    ]


async def get_volatility_history(
    client: AlpacaClient,
    symbol: str,
    days_back: int = 252,
    period: int = 20,
    feed: str = "iex",
) -> list[dict[str, Any]]:
    """Get historical realized volatility over time (contract: list of points)."""
    points = await _fetch_volatility(client, symbol, days_back, period, feed)
    return [
        {
            "date": p.date,
            "realized_vol": p.realized_vol,
            "rolling_vol_20d": p.rolling_vol_20d,
            "rolling_vol_60d": p.rolling_vol_60d,
        }
        for p in points
    ]
