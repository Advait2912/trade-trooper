"""Pydantic schemas for every structured piece of data in the pipeline.

Everything the LLM returns is validated against these models so that a
malformed or hallucinated response fails loudly instead of silently
propagating.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class Materiality(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Sentiment(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    UNCERTAIN = "uncertain"


class EvidenceQuality(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Trend(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    UNCERTAIN = "uncertain"


class Actionability(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class TimeHorizon(str, Enum):
    IMMEDIATE = "immediate"
    ONE_TO_FIVE_DAYS = "1-5_days"
    ONE_TO_FOUR_WEEKS = "1-4_weeks"
    LONG_TERM = "long_term"
    UNCERTAIN = "uncertain"


class Bias(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    UNCERTAIN = "uncertain"


class SourceRelevance(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# ---------------------------------------------------------------------------
# Stage 1/2 — News
# ---------------------------------------------------------------------------
class NewsArticle(BaseModel):
    """A normalized Alpaca news article (headline/summary only)."""

    id: int
    headline: str
    summary: str = ""
    source: str = ""
    url: str = ""
    created_at: str = ""
    updated_at: str = ""
    symbols: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Stage 3 — Initial analysis
# ---------------------------------------------------------------------------
class InitialAnalysis(BaseModel):
    """Structured output of the first (cheap) LLM pass."""

    ticker: str
    event: str
    relevance: float = Field(ge=0.0, le=1.0)
    materiality: Materiality = Materiality.LOW
    sentiment: Sentiment = Sentiment.UNCERTAIN
    evidence_quality: EvidenceQuality = EvidenceQuality.LOW
    needs_web_research: bool = False
    research_questions: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Stage 4/5 — Web research
# ---------------------------------------------------------------------------
class SearchResult(BaseModel):
    title: str = ""
    url: str = ""
    content: str = ""


class FetchedPage(BaseModel):
    url: str
    title: str = ""
    content: str = ""
    links: List[str] = Field(default_factory=list)


class WebSource(BaseModel):
    title: str = ""
    source: str = ""
    url: str = ""
    relevance: SourceRelevance = SourceRelevance.MEDIUM


# ---------------------------------------------------------------------------
# Stage 6 — Market data / indicators
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Stage 7 — Final report (section 11 schema)
# ---------------------------------------------------------------------------
class NewsRef(BaseModel):
    headline: str = ""
    source: str = ""
    published_at: str = ""
    url: str = ""


class EventAssessment(BaseModel):
    description: str = ""
    relevance: float = Field(default=0.0, ge=0.0, le=1.0)
    materiality: Materiality = Materiality.LOW
    sentiment: Sentiment = Sentiment.UNCERTAIN


class Evidence(BaseModel):
    quality: EvidenceQuality = EvidenceQuality.LOW
    facts: List[str] = Field(default_factory=list)
    inferences: List[str] = Field(default_factory=list)
    uncertainties: List[str] = Field(default_factory=list)


class WebResearch(BaseModel):
    performed: bool = False
    key_findings: List[str] = Field(default_factory=list)
    sources: List[WebSource] = Field(default_factory=list)


class MarketContext(BaseModel):
    price: float = 0.0
    return_1d: float = 0.0
    return_5d: float = 0.0
    sma20: float = 0.0
    sma50: float = 0.0
    rsi14: float = 0.0
    volume_vs_average: float = 0.0
    trend: Trend = Trend.UNCERTAIN


class Analysis(BaseModel):
    news_impact: float = Field(default=0.0, ge=-1.0, le=1.0)
    actionability: Actionability = Actionability.NONE
    time_horizon: TimeHorizon = TimeHorizon.UNCERTAIN
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    summary: str = ""


class CouncilInput(BaseModel):
    recommended_bias: Bias = Bias.NEUTRAL
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    key_reason: str = ""
    should_council_consider: bool = False


class FinalReport(BaseModel):
    """The full structured intelligence report (section 11 schema)."""

    ticker: str
    timestamp: str
    news: NewsRef = Field(default_factory=NewsRef)
    event: EventAssessment = Field(default_factory=lambda: EventAssessment())
    evidence: Evidence = Field(default_factory=Evidence)
    web_research: WebResearch = Field(default_factory=WebResearch)
    market_context: MarketContext = Field(default_factory=MarketContext)
    analysis: Analysis = Field(default_factory=lambda: Analysis())
    council_input: CouncilInput = Field(default_factory=lambda: CouncilInput())


class FinalSynthesis(BaseModel):
    """The LLM-produced portion of the final report.

    Market-context numbers are computed deterministically elsewhere and merged
    in; the model only contributes qualitative sections plus `market_trend`.
    """

    event: EventAssessment = Field(default_factory=lambda: EventAssessment())
    evidence: Evidence = Field(default_factory=Evidence)
    web_research: WebResearch = Field(default_factory=WebResearch)
    market_trend: Trend = Trend.UNCERTAIN
    analysis: Analysis = Field(default_factory=lambda: Analysis())
    council_input: CouncilInput = Field(default_factory=lambda: CouncilInput())
