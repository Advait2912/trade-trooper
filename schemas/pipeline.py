"""Analysis / report schemas and phase placeholder models.

Final report models (Phases 1-4 feeding the final synthesis) live here.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

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
from schemas.historical import HistoricalAgentResult


# ---------------------------------------------------------------------------
# Web research
# ---------------------------------------------------------------------------
class SearchResult(BaseModel):
    title: str = ""
    url: str = ""
    content: str = ""


class FetchedPage(BaseModel):
    url: str
    title: str = ""
    content: str = ""
    links: list[str] = Field(default_factory=list)


class WebSource(BaseModel):
    title: str = ""
    source: str = ""
    url: str = ""
    relevance: SourceRelevance = SourceRelevance.MEDIUM


# ---------------------------------------------------------------------------
# Final report
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
    facts: list[str] = Field(default_factory=list)
    inferences: list[str] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)


class WebResearch(BaseModel):
    performed: bool = False
    key_findings: list[str] = Field(default_factory=list)
    sources: list[WebSource] = Field(default_factory=list)


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


class PhaseResult(BaseModel):
    """Shape returned by Phase 2/3/4 agents while they are not implemented."""

    phase: str = ""
    status: str = "not_implemented"  # not_implemented | ok | error
    summary: str = ""
    data: dict = Field(default_factory=dict)


class FinalReport(BaseModel):
    """The full structured intelligence report."""

    ticker: str
    timestamp: str
    news: NewsRef = Field(default_factory=NewsRef)
    event: EventAssessment = Field(default_factory=lambda: EventAssessment())
    evidence: Evidence = Field(default_factory=Evidence)
    web_research: WebResearch = Field(default_factory=WebResearch)
    market_context: MarketContext = Field(default_factory=MarketContext)
    historical: HistoricalAgentResult = Field(
        default_factory=lambda: HistoricalAgentResult()
    )
    prediction: PhaseResult = Field(default_factory=lambda: PhaseResult(phase="prediction"))
    risk: PhaseResult = Field(default_factory=lambda: PhaseResult(phase="risk"))
    decision: PhaseResult = Field(default_factory=lambda: PhaseResult(phase="decision"))
    analysis: Analysis = Field(default_factory=lambda: Analysis())
    council_input: CouncilInput = Field(default_factory=lambda: CouncilInput())
