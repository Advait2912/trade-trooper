"""Agent implementations.

The suite is arranged by pipeline phase:
- Phase 1 (parallel): ``NewsCollectionAgent``, ``MarketDataAgent``, ``HistoricalAgent``
- Phase 2 (sequential): ``PredictionAgent``
- Phase 3 (sequential): ``RiskAgent``
- Phase 4 (sequential): ``DecisionAgent``
"""

from agents.base import BaseAgent
from agents.decision_agent import DecisionAgent
from agents.historical_agent import HistoricalAgent
from agents.market_data_agent import MarketDataAgent
from agents.news_agent import NewsCollectionAgent
from agents.prediction_agent import PredictionAgent
from agents.risk_agent import RiskAgent

__all__ = [
    "BaseAgent",
    "NewsCollectionAgent",
    "MarketDataAgent",
    "HistoricalAgent",
    "PredictionAgent",
    "RiskAgent",
    "DecisionAgent",
]
