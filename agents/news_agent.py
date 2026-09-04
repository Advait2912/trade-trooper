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

import httpx

from agents.base import BaseAgent
from agents.llm import OllamaClient, OllamaError
from alpaca.client import AlpacaClient, AlpacaError
from alpaca.news import fetch_news
from schemas.news import InitialAnalysis, NewsArticle, NewsCollectionResult
from tools.news_tools import sentiment_analysis
from utils.config import Settings
from utils.logging import StageTimer

log = logging.getLogger("market_intel_agent.news_agent")

_local_finbert: "FinBertSentiment | None" = None
_local_finbert_unavailable = False


def _score_local_finbert(texts: list[str]) -> list[dict] | None:
    """Score texts with the local FinBERT scorer (module-level singleton).

    Returns None when torch/transformers are not installed or the model
    cannot be loaded, so callers degrade to Ollama / neutral.
    """
    global _local_finbert, _local_finbert_unavailable
    if _local_finbert_unavailable:
        return None
    if _local_finbert is None:
        try:
            from tools.finbert_sentiment import FinBertSentiment

            _local_finbert = FinBertSentiment(device="cuda")
        except Exception as exc:  # noqa: BLE001 - optional dependency
            log.warning("Local FinBERT unavailable: %s", exc)
            _local_finbert_unavailable = True
            return None
    return _local_finbert.score_batch(texts, batch_size=32)


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
        # 1. Prioritize shared FinBERT service when configured
        if self.settings.finbert_url:
            try:
                texts = [f"{a.headline}. {a.summary}".strip() for a in articles]
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.post(
                        f"{self.settings.finbert_url.rstrip('/')}/score_batch",
                        json={"texts": texts},
                    )
                    resp.raise_for_status()
                    scores = resp.json()
                    log.info("Scored %d articles via shared FinBERT service", len(articles))
                    return [
                        _finbert_score_to_initial(
                            ticker,
                            a,
                            str(s.get("label", "neutral")),
                            float(s.get("score", 0.0) or 0.0),
                        )
                        for a, s in zip(articles, scores)
                    ]
            except Exception as exc:
                log.warning(
                    "FinBERT scoring via %s failed: %s; falling back",
                    self.settings.finbert_url,
                    exc,
                )

        # 2. Local FinBERT scorer (GPU) — same model + scoring as the
        #    backtest news cache, so live sentiment matches what the tuned
        #    weights were optimized against.  Loaded once per process.
        try:
            local_scores = _score_local_finbert(
                [f"{a.headline}. {a.summary}".strip() for a in articles]
            )
            if local_scores is not None:
                log.info("Scored %d articles via local FinBERT (GPU)", len(articles))
                return [
                    _finbert_score_to_initial(
                        ticker, a, str(s.get("label", "neutral")), float(s.get("score", 0.0) or 0.0)
                    )
                    for a, s in zip(articles, local_scores)
                ]
        except Exception as exc:  # noqa: BLE001
            log.warning("Local FinBERT scoring failed: %s; falling back", exc)

        # 3. Try Ollama if LLM synthesis is enabled
        if self.settings.enable_llm_synthesis:
            try:
                results: list[InitialAnalysis] = []
                async with OllamaClient(self.settings) as ollama:
                    tasks = [ollama.analyze_initial(ticker, a) for a in articles]
                    outcomes = await asyncio.gather(*tasks, return_exceptions=True)

                for article, outcome in zip(articles, outcomes):
                    if isinstance(outcome, BaseException):
                        results.append(_fallback_initial(ticker, article))
                    else:
                        results.append(outcome)
                return results
            except Exception as exc:
                log.warning("Ollama analysis failed: %s; using neutral fallback", exc)

        # 3. Deterministic neutral fallback
        return [_fallback_initial(ticker, a) for a in articles]

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


def _finbert_score_to_initial(
    ticker: str,
    article: NewsArticle,
    label: str,
    score: float,
) -> InitialAnalysis:
    from schemas.common import EvidenceQuality, Materiality, Sentiment

    label_lower = label.lower()
    if "pos" in label_lower:
        sentiment = Sentiment.BULLISH
    elif "neg" in label_lower:
        sentiment = Sentiment.BEARISH
    else:
        sentiment = Sentiment.NEUTRAL

    relevance = min(0.9, max(0.5, 0.5 + score * 0.4))
    materiality = (
        Materiality.HIGH
        if score > 0.75
        else (Materiality.MEDIUM if score > 0.50 else Materiality.LOW)
    )

    return InitialAnalysis(
        ticker=ticker,
        event=article.headline or "No event identified.",
        relevance=relevance,
        materiality=materiality,
        sentiment=sentiment,
        evidence_quality=EvidenceQuality.MEDIUM,
        needs_web_research=False,
        research_questions=[],
    )


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
