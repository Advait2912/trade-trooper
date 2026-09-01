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

__all__ = [
    "Actionability",
    "Analysis",
    "Bias",
    "CouncilInput",
    "DividendRecord",
    "EarningsRecord",
    "EventAssessment",
    "Evidence",
    "EvidenceQuality",
    "FetchedPage",
    "FinalReport",
    "FinalSynthesis",
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
    "SearchResult",
    "Sentiment",
    "SourceRelevance",
    "TimeHorizon",
    "Trend",
    "VolatilityPoint",
    "WebResearch",
    "WebSource",
]
