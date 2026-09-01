"""News schemas (Phase 1 - News Collection Agent)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from schemas.common import EvidenceQuality, Materiality, Sentiment


class NewsArticle(BaseModel):
    """A normalized Alpaca news article (headline/summary only)."""

    id: int
    headline: str
    summary: str = ""
    source: str = ""
    url: str = ""
    created_at: str = ""
    updated_at: str = ""
    symbols: list[str] = Field(default_factory=list)


class InitialAnalysis(BaseModel):
    """Structured output of the per-article LLM sentiment pass."""

    ticker: str
    event: str
    relevance: float = Field(ge=0.0, le=1.0)
    materiality: Materiality = Materiality.LOW
    sentiment: Sentiment = Sentiment.UNCERTAIN
    evidence_quality: EvidenceQuality = EvidenceQuality.LOW
    needs_web_research: bool = False
    research_questions: list[str] = Field(default_factory=list)


class NewsCollectionResult(BaseModel):
    """Output of the News Collection Agent (Phase 1)."""

    ticker: str = ""
    articles: list[NewsArticle] = Field(default_factory=list)
    analyses: list[InitialAnalysis] = Field(default_factory=list)
    primary_article: NewsArticle | None = None
    primary_analysis: InitialAnalysis | None = None
    sentiment_score: float = 0.0
    errors: list[str] = Field(default_factory=list)
