"""Phase 1 - Historical Data Agent (stub; full implementation arrives with
``tools.historical``).

Calls (future):
    - get_price_history()
    - get_dividends_history()
    - get_earnings_history()
    - get_volatility_history()
Returns: historical_trends, volatility_history
"""

from __future__ import annotations

import logging

from agents.base import BaseAgent
from schemas.historical import HistoricalAgentResult
from utils.config import Settings

log = logging.getLogger("market_intel_agent.historical_agent")


class HistoricalAgent(BaseAgent):
    name = "historical"
    description = "Collects price/dividend history and computes historical trends."
    phase = 1
    tools = [
        "get_price_history",
        "get_dividends_history",
        "get_earnings_history",
        "get_volatility_history",
    ]

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def run(self, ticker: str) -> HistoricalAgentResult:
        return HistoricalAgentResult(
            symbol=ticker,
            status="partial",
            errors=["Historical agent pending implementation."],
        )
