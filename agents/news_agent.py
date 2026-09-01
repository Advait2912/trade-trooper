"""Phase 1 - News Collection Agent.

Collects Alpaca news for a ticker, applies deterministic filtering (in
``alpaca.news``), runs the per-article sentiment LLM pass, and returns a
structured result: ``news_signals`` (article -> analysis) plus a deterministic
``sentiment_score``.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from agents.base import BaseAgent
from agents.llm import OllamaClient, OllamaError
from alpaca.client import AlpacaClient, AlpacaError
from alpaca.news import fetch_news
from schemas.news import InitialAnalysis, NewsArticle, NewsCollectionResult
from tools.news_tools import sentiment_analysis
from utils.config import Settings
from utils.logging import StageTimer

log = logging.getLogger("market_intel_agent.news_agent")


class NewsCollectionAgent(BaseAgent):
    """Fetches + filters Alpaca news and scores sentiment per article."""

    name = "news_collection"
    description = "Fetches relevant Alpaca news and computes a sentiment score."
    phase = 1
    tools = ["fetch_news", "sentiment_analysis"]

    def __init__(
        self,
        settings: Settings,
        verbose: bool = False,
        company_names: Optional[list[str]] = None,
    ) -> None:
        self.settings = settings
        self.verbose = verbose
        self.company_names = company_names or []

    async def run(self, ticker: str) -> NewsCollectionResult:
        result = NewsCollectionResult(ticker=ticker)

        with StageTimer("Alpaca news fetched", log):
            try:
                async with AlpacaClient(self.settings) as client:
                    articles = await fetch_news(
                        client,
                        ticker,
                        self.settings.news_limit,
                        self.settings.lookback_hours,
                        self.company_names,
                    )
            except AlpacaError as exc:
                log.warning("Alpaca news fetch failed: %s", exc)
                result.errors.append(f"news fetch failed: {exc}")
                return result

        result.articles = articles
        if not articles:
            return result

        if self.verbose:
            for a in articles:
                log.debug("selected headline: %s", a.headline)

        analyses = await self._analyze_all(ticker, articles)
        result.analyses = analyses

        primary = self._pick_primary(list(zip(articles, analyses)))
        if primary is not None:
            result.primary_article, result.primary_analysis = primary

        result.sentiment_score = float(sentiment_analysis(analyses)["sentiment_score"])
        return result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    async def _analyze_all(
        self, ticker: str, articles: list[NewsArticle]
    ) -> list[InitialAnalysis]:
        results: list[InitialAnalysis] = []
        async with OllamaClient(self.settings) as ollama:
            tasks = [ollama.analyze_initial(ticker, a) for a in articles]
            outcomes = await asyncio.gather(*tasks, return_exceptions=True)

        for article, outcome in zip(articles, outcomes):
            if isinstance(outcome, BaseException):
                if isinstance(outcome, OllamaError):
                    log.warning(
                        "Initial analysis failed for %r: %s", article.headline, outcome
                    )
                results.append(_fallback_initial(ticker, article))
            else:
                results.append(outcome)
        return results

    @staticmethod
    def _pick_primary(
        pairs: list[tuple[NewsArticle, InitialAnalysis]],
    ) -> Optional[tuple[NewsArticle, InitialAnalysis]]:
        """Choose the article that matters most (relevance, then materiality)."""
        if not pairs:
            return None

        def score(pair: tuple[NewsArticle, InitialAnalysis]) -> tuple[float, int]:
            _, a = pair
            materiality = {"low": 0, "medium": 1, "high": 2}.get(
                a.materiality.value, 0
            )
            return (a.relevance, materiality)

        return max(pairs, key=score)


def _fallback_initial(ticker: str, article: NewsArticle) -> InitialAnalysis:
    from schemas.common import EvidenceQuality, Materiality, Sentiment

    return InitialAnalysis(
        ticker=ticker,
        event=article.headline or "No event identified.",
        relevance=0.0,
        materiality=Materiality.LOW,
        sentiment=Sentiment.UNCERTAIN,
        evidence_quality=EvidenceQuality.LOW,
        needs_web_research=False,
        research_questions=[],
    )
