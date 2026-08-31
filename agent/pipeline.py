"""The async controller that orchestrates the full intelligence pipeline.

Stages:
  1. Fetch Alpaca news
  2. Deterministic filtering/selection (already applied in alpaca.news)
  3. Initial Gemma analysis (per selected article, in parallel)
  4. Web research (only when warranted)
  5. Web fetch (extract text from chosen pages)
  6. Market data + deterministic indicators
  7. Final Gemma synthesis

Every stage is timed and independently degradable: a failure in one stage
(e.g. Ollama down) produces a partial but still-valid report rather than
crashing the whole run.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import List, Optional, Tuple, cast

from agent.analyst import OllamaClient, OllamaError
from agent.schemas import (
    Actionability,
    Bias,
    EvidenceQuality,
    FetchedPage,
    FinalReport,
    FinalSynthesis,
    InitialAnalysis,
    MarketContext,
    MarketData,
    Materiality,
    NewsArticle,
    NewsRef,
    Sentiment,
    SourceRelevance,
    TimeHorizon,
    Trend,
    WebResearch,
    WebSource,
)
from alpaca.client import AlpacaClient, AlpacaError
from alpaca.market_data import fetch_market_data, suggest_trend
from alpaca.news import fetch_news
from utils.config import ConfigError, Settings
from utils.logging import PipelineClock, StageTimer
from web.search import WebResearcher, generate_queries

log = logging.getLogger("market_intel_agent.pipeline")


class Pipeline:
    """Orchestrates news -> analyze -> research -> market -> synthesize."""

    def __init__(
        self,
        settings: Settings,
        verbose: bool = False,
        company_names: Optional[List[str]] = None,
    ) -> None:
        self.settings = settings
        self.verbose = verbose
        self.company_names = company_names or []
        self.clock = PipelineClock()

    async def run(self, ticker: str) -> FinalReport:
        if not self.settings.has_alpaca_credentials:
            raise ConfigError(
                "Missing Alpaca credentials. Set ALPACA_API_KEY and "
                "ALPACA_API_SECRET in your environment or .env file."
            )

        self.clock.mark("Starting")

        # ---------------------------------------------------------------
        # Stage 1 + 2 — Alpaca news (fetch + deterministic filter)
        # ---------------------------------------------------------------
        articles: List[NewsArticle] = []
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

        self.clock.mark(f"{len(articles)} articles selected")

        if not articles:
            return self._empty_report(
                ticker, "No relevant news found for the lookback window."
            )

        if self.verbose:
            for a in articles:
                log.debug("selected headline: %s", a.headline)

        # ---------------------------------------------------------------
        # Stage 3 — Initial Gemma analysis (parallel per article)
        # ---------------------------------------------------------------
        analyses: List[Tuple[NewsArticle, InitialAnalysis]] = []
        with StageTimer("Initial Gemma analysis complete", log):
            analyses = await self._analyze_all(ticker, articles)

        primary_article, primary_analysis = self._pick_primary(analyses)

        # ---------------------------------------------------------------
        # Stage 4/5 — Web research (parallel with market data)
        # ---------------------------------------------------------------
        web_task = self._research(
            ticker, primary_article, primary_analysis
        ) if primary_analysis.needs_web_research else None
        market_task = self._market_data(ticker)

        if web_task is not None:
            web_result, market_data = await asyncio.gather(
                web_task, market_task, return_exceptions=True
            )
        else:
            web_result = None
            try:
                market_data = await market_task
            except AlpacaError as exc:
                log.warning("Market data fetch failed: %s", exc)
                market_data = MarketData()

        if isinstance(web_result, BaseException):
            log.warning("Web research failed: %s", web_result)
            web_result = None
        if isinstance(market_data, BaseException):
            log.warning("Market data fetch failed: %s", market_data)
            market_data = MarketData()

        web = cast(Optional[WebResearch], web_result)
        market = cast(MarketData, market_data)

        # ---------------------------------------------------------------
        # Stage 7 — Final Gemma synthesis
        # ---------------------------------------------------------------
        synthesis: Optional[FinalSynthesis] = None
        with StageTimer("Final analysis complete", log):
            try:
                synthesis = await self._synthesize(
                    ticker,
                    primary_article,
                    primary_analysis,
                    web,
                    market,
                )
            except OllamaError as exc:
                log.warning("Final synthesis failed: %s", exc)

        report = self._build_report(
            ticker,
            primary_article,
            primary_analysis,
            web,
            market,
            synthesis,
        )
        self.clock.mark("Done")
        return report

    # ------------------------------------------------------------------
    # Stage helpers
    # ------------------------------------------------------------------
    async def _analyze_all(
        self, ticker: str, articles: List[NewsArticle]
    ) -> List[Tuple[NewsArticle, InitialAnalysis]]:
        results: List[Tuple[NewsArticle, InitialAnalysis]] = []
        async with OllamaClient(self.settings) as ollama:
            tasks = [ollama.analyze_initial(ticker, a) for a in articles]
            outcomes = await asyncio.gather(*tasks, return_exceptions=True)

        for article, outcome in zip(articles, outcomes):
            if isinstance(outcome, BaseException):
                log.warning(
                    "Initial analysis failed for %r: %s", article.headline, outcome
                )
                analysis = _fallback_initial(ticker, article)
            else:
                analysis = cast(InitialAnalysis, outcome)
            results.append((article, analysis))
        return results

    @staticmethod
    def _pick_primary(
        analyses: List[Tuple[NewsArticle, InitialAnalysis]]
    ) -> Tuple[NewsArticle, InitialAnalysis]:
        """Choose the article that matters most for the final report."""

        def score(pair: Tuple[NewsArticle, InitialAnalysis]) -> Tuple[float, int]:
            _, a = pair
            materiality = {"low": 0, "medium": 1, "high": 2}[a.materiality.value]
            return (a.relevance, materiality)

        return max(analyses, key=score)

    async def _research(
        self,
        ticker: str,
        article: NewsArticle,
        analysis: InitialAnalysis,
    ) -> Optional[WebResearch]:
        researcher = WebResearcher(self.settings)
        queries = generate_queries(
            ticker,
            article.headline,
            analysis.research_questions,
            self.company_names,
        )
        # Cap the number of search rounds.
        queries = queries[: self.settings.max_search_rounds]

        if self.verbose:
            log.debug("search queries: %s", queries)

        with StageTimer("Web research complete", log):
            search_lists = await asyncio.gather(
                *(researcher.search(q) for q in queries), return_exceptions=True
            )

        results = []
        for item in search_lists:
            if isinstance(item, list):
                results.extend(item)

        results = _dedupe_results(results)

        # Choose pages to fetch (prefer primary sources), then fetch in parallel.
        to_fetch = _rank_results(results)[: self.settings.max_fetch_pages]
        fetched = await asyncio.gather(
            *(researcher.fetch(r.url) for r in to_fetch), return_exceptions=True
        )

        pages = []
        fetched_urls = set()
        for page in fetched:
            if isinstance(page, BaseException) or page is None:
                continue
            p = cast(FetchedPage, page)
            fetched_urls.add(p.url)
            pages.append(p)

        if self.verbose:
            for p in pages:
                log.debug("fetched page: %s (%d chars)", p.url, len(p.content))

        return self._build_web_research(results, pages, fetched_urls)

    def _build_web_research(
        self,
        results: List,
        pages: List,
        fetched_urls: set,
    ) -> WebResearch:
        sources: List[WebSource] = []
        for r in results:
            relevance = SourceRelevance.HIGH if r.url in fetched_urls else SourceRelevance.MEDIUM
            sources.append(
                WebSource(
                    title=r.title,
                    source=_domain_of(r.url),
                    url=r.url,
                    relevance=relevance,
                )
            )
        # De-duplicate sources by URL.
        seen = set()
        unique_sources: List[WebSource] = []
        for s in sources:
            if s.url and s.url not in seen:
                seen.add(s.url)
                unique_sources.append(s)

        # Deterministic findings from fetched page titles/lead text.
        key_findings = [
            f"{p.title or p.url}: {p.content[:200].strip()}" for p in pages if p.content
        ][:5]

        return WebResearch(
            performed=True,
            key_findings=key_findings,
            sources=unique_sources,
        )

    async def _market_data(self, ticker: str) -> MarketData:
        with StageTimer("Market data retrieved", log):
            async with AlpacaClient(self.settings) as client:
                return await fetch_market_data(
                    client, ticker, self.settings.alpaca_data_feed
                )

    async def _synthesize(
        self,
        ticker: str,
        article: NewsArticle,
        initial: InitialAnalysis,
        web: Optional[WebResearch],
        market: MarketData,
    ) -> FinalSynthesis:
        async with OllamaClient(self.settings) as ollama:
            return await ollama.synthesize(
                ticker=ticker,
                news_block=_news_block(article),
                initial_block=_initial_block(initial),
                web_block=_web_block(web),
                market_block=_market_block(market),
                performed=web is not None and web.performed,
            )

    # ------------------------------------------------------------------
    # Report assembly
    # ------------------------------------------------------------------
    def _build_report(
        self,
        ticker: str,
        article: NewsArticle,
        initial: InitialAnalysis,
        web: Optional[WebResearch],
        market: MarketData,
        synthesis: Optional[FinalSynthesis],
    ) -> FinalReport:
        now = datetime.now(timezone.utc).isoformat()

        if synthesis is None:
            synthesis = _fallback_synthesis(initial, web)

        # Market context numbers are always deterministic.
        market_context = MarketContext(
            price=round(market.price, 2),
            return_1d=round(market.return_1d, 4),
            return_5d=round(market.return_5d, 4),
            sma20=round(market.sma20, 2),
            sma50=round(market.sma50, 2),
            rsi14=round(market.rsi14, 2),
            volume_vs_average=round(market.volume_vs_average, 2),
            trend=synthesis.market_trend if synthesis.market_trend != Trend.UNCERTAIN
            else suggest_trend(market),
        )

        # Web research: `performed` and `sources` are deterministic facts
        # (never the LLM's assertions); `key_findings` reflect the LLM's
        # interpretation when available, else the deterministic excerpts.
        web_research = synthesis.web_research
        if web is not None:
            web_research = WebResearch(
                performed=web.performed,
                key_findings=web_research.key_findings or web.key_findings,
                sources=web.sources,
            )
        else:
            web_research = WebResearch(performed=False)

        return FinalReport(
            ticker=ticker,
            timestamp=now,
            news=NewsRef(
                headline=article.headline,
                source=article.source,
                published_at=article.created_at or article.updated_at,
                url=article.url,
            ),
            event=synthesis.event,
            evidence=synthesis.evidence,
            web_research=web_research,
            market_context=market_context,
            analysis=synthesis.analysis,
            council_input=synthesis.council_input,
        )

    def _empty_report(self, ticker: str, message: str) -> FinalReport:
        self.clock.mark("Done")
        now = datetime.now(timezone.utc).isoformat()
        synthesis = _fallback_synthesis(_fallback_initial(ticker, _blank_article(ticker)), None)
        synthesis.analysis.summary = f"{message} Insufficient evidence."
        synthesis.evidence.uncertainties.append(message)
        return FinalReport(
            ticker=ticker,
            timestamp=now,
            news=NewsRef(),
            event=synthesis.event,
            evidence=synthesis.evidence,
            web_research=WebResearch(performed=False),
            market_context=MarketContext(),
            analysis=synthesis.analysis,
            council_input=synthesis.council_input,
        )


# ---------------------------------------------------------------------------
# Fallbacks (graceful degradation)
# ---------------------------------------------------------------------------
def _blank_article(ticker: str) -> NewsArticle:
    return NewsArticle(id=0, headline="", symbols=[ticker])


def _fallback_initial(ticker: str, article: NewsArticle) -> InitialAnalysis:
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


def _fallback_synthesis(
    initial: InitialAnalysis, web: Optional[WebResearch]
) -> FinalSynthesis:
    from agent.schemas import Analysis, CouncilInput, EventAssessment, Evidence

    return FinalSynthesis(
        event=EventAssessment(
            description=initial.event,
            relevance=initial.relevance,
            materiality=initial.materiality,
            sentiment=initial.sentiment,
        ),
        evidence=Evidence(
            quality=initial.evidence_quality,
            facts=[],
            inferences=[],
            uncertainties=["Insufficient evidence."],
        ),
        web_research=web or WebResearch(performed=False),
        market_trend=Trend.UNCERTAIN,
        analysis=Analysis(
            news_impact=0.0,
            actionability=Actionability.NONE,
            time_horizon=TimeHorizon.UNCERTAIN,
            confidence=0.0,
            summary="Insufficient evidence.",
        ),
        council_input=CouncilInput(
            recommended_bias=Bias.NEUTRAL,
            confidence=0.0,
            key_reason="Insufficient evidence.",
            should_council_consider=False,
        ),
    )


# ---------------------------------------------------------------------------
# Block formatters (concise context for the LLM)
# ---------------------------------------------------------------------------
def _news_block(article: NewsArticle) -> str:
    return (
        f"Headline: {article.headline}\n"
        f"Source: {article.source or 'unknown'}\n"
        f"Summary: {article.summary or '(none)'}\n"
        f"URL: {article.url or '(none)'}\n"
        f"Published: {article.created_at or article.updated_at or '(unknown)'}"
    )


def _initial_block(analysis: InitialAnalysis) -> str:
    return json.dumps(analysis.model_dump(), indent=2)


def _web_block(web: Optional[WebResearch]) -> str:
    if web is None or not web.performed:
        return "No web research was performed."
    return json.dumps(web.model_dump(), indent=2)


def _market_block(market: MarketData) -> str:
    return json.dumps(market.model_dump(), indent=2)


# ---------------------------------------------------------------------------
# Search result ranking helpers
# ---------------------------------------------------------------------------
def _dedupe_results(results: List) -> List:
    seen = set()
    out = []
    for r in results:
        if not r.url:
            continue
        if r.url in seen:
            continue
        seen.add(r.url)
        out.append(r)
    return out


def _rank_results(results: List) -> List:
    """Prefer primary sources (SEC, company IR, reputable publications)."""
    return sorted(results, key=lambda r: _source_priority(r.url))


def _source_priority(url: str) -> int:
    u = url.lower()
    if "sec.gov" in u:
        return 0
    if "gov" in u:
        return 1
    if any(k in u for k in ("ir.", "investor", "press", "newsroom")):
        return 2
    if any(k in u for k in ("reuters", "bloomberg", "wsj", "ft.com", "cnbc")):
        return 3
    if any(k in u for k in ("techcrunch", "theverge", "arstechnica", "wired")):
        return 4
    return 5


def _domain_of(url: str) -> str:
    try:
        from urllib.parse import urlparse

        host = urlparse(url).netloc
        return host.removeprefix("www.")
    except Exception:
        return url
