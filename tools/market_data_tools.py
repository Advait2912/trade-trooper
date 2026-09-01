"""Deterministic market-data tools (Phase 1 - Market Data Agent)."""

from __future__ import annotations

from alpaca.client import AlpacaClient
from alpaca.market_data import fetch_market_data
from schemas.market import MarketData


async def get_market_data(client: AlpacaClient, symbol: str, feed: str = "iex") -> MarketData:
    """Fetch snapshot + daily bars and compute deterministic indicators."""
    return await fetch_market_data(client, symbol, feed)


def get_current_price(md: MarketData) -> float:
    """Extract the current price from computed market data."""
    return float(md.price)


def get_volatility(md: MarketData) -> float:
    """Extract the realized annualized volatility from computed market data."""
    return float(md.volatility)
