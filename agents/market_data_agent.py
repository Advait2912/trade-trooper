"""Phase 1 - Market Data Agent.

Fetches the live snapshot + recent daily bars and computes deterministic
market indicators (price, returns, SMA, RSI, ATR, volatility).
"""

from __future__ import annotations

import logging

from agents.base import BaseAgent
from alpaca.client import AlpacaClient, AlpacaError
from alpaca.market_data import fetch_market_data
from schemas.market import MarketData
from utils.config import Settings
from utils.logging import StageTimer

log = logging.getLogger("market_intel_agent.market_agent")


class MarketDataAgent(BaseAgent):
    """Collects live market data + volatility for a ticker."""

    name = "market_data"
    description = "Fetches current price, quotes and computed volatility."
    phase = 1
    tools = ["get_current_price", "get_volatility", "get_market_data"]

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def run(self, ticker: str) -> MarketData:
        with StageTimer("Market data retrieved", log):
            try:
                async with AlpacaClient(self.settings) as client:
                    return await fetch_market_data(
                        client, ticker, self.settings.alpaca_data_feed
                    )
            except AlpacaError as exc:
                log.warning("Market data fetch failed: %s", exc)
                return MarketData()

    async def fetch(self, client: AlpacaClient, ticker: str) -> MarketData:
        """Fetch with a caller-provided client (for parallel collection)."""
        return await fetch_market_data(client, ticker, self.settings.alpaca_data_feed)
