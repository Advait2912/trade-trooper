"""Shared Pydantic schemas package.

Split by domain:
- ``common``: enums shared across agents
- ``news``: news articles
- ``market``: market data / indicators
- ``historical``: historical data + the historical agent result
- ``pipeline``: analysis/report models + phase placeholder results
"""

from schemas.common import (
    Actionability,
    Bias,
    EvidenceQuality,
    Materiality,
    Sentiment,
    SourceRelevance,
    TimeHorizon,
    Trend,
)
from schemas.decision import DecisionGate, DecisionResult, Opportunity
from schemas.historical import (
    DividendRecord,
    EarningsRecord,
    HistoricalAgentResult,
    PriceBar,
    VolatilityPoint,
)
from schemas.market import MarketData
from schemas.news import InitialAnalysis, NewsArticle, NewsCollectionResult
from schemas.pipeline import (
    Analysis,
    CouncilInput,
    EventAssessment,
    Evidence,
    FetchedPage,
    FinalReport,
    FinalSynthesis,
    MarketContext,
    NewsRef,
    PhaseResult,
    SearchResult,
    WebResearch,
    WebSource,
)
from schemas.risk import (
    EquityPosition,
    Greeks,
    OptionPosition,
    PositionRecommendation,
    RiskResult,
)

__all__ = [
    "Actionability",
    "Analysis",
    "Bias",
    "CouncilInput",
    "DecisionGate",
    "DecisionResult",
    "DividendRecord",
    "EarningsRecord",
    "EventAssessment",
    "Evidence",
    "EvidenceQuality",
    "FetchedPage",
    "FinalReport",
    "FinalSynthesis",
    "Greeks",
    "EquityPosition",
    "Opportunity",
    "OptionPosition",
    "PositionRecommendation",
    "HistoricalAgentResult",
    "InitialAnalysis",
    "MarketContext",
    "MarketData",
    "Materiality",
    "NewsArticle",
    "NewsCollectionResult",
    "NewsRef",
    "PhaseResult",
    "PriceBar",
    "RiskResult",
    "SearchResult",
    "Sentiment",
    "SourceRelevance",
    "TimeHorizon",
    "Trend",
    "VolatilityPoint",
    "WebResearch",
    "WebSource",
]
