"""The async orchestrator running the phased trading-intelligence pipeline.

Phases (per the phase architecture):
  PHASE 1 (parallel collection, ~4s): News Collection Agent, Market Data
      Agent, Historical Data Agent run concurrently.
  PHASE 2 (sequential, ~3s): Prediction Agent (deterministic).
  PHASE 3 (sequential, ~2s): Risk Agent (options chain + deterministic sizing).
  PHASE 4 (sequential, ~1s): Decision Agent (deterministic signal synthesis +
      opportunity ranking into a trade decision).

  Followed by: web research (when warranted) and final LLM synthesis into the
  report. Every stage is timed and independently degradable: a failure in one
  stage produces a partial but still-valid report rather than crashing.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, cast

from agents.decision_agent import DecisionAgent
from agents.historical_agent import HistoricalAgent
from agents.llm import OllamaClient, OllamaError
from agents.market_data_agent import MarketDataAgent
from agents.news_agent import NewsCollectionAgent
from agents.prediction_agent import PredictionAgent
from agents.risk_agent import RiskAgent
from schemas.common import (
    Actionability,
    Bias,
    EvidenceQuality,
    Materiality,
    Sentiment,
    TimeHorizon,
    Trend,
)
from schemas.decision import DecisionResult
from schemas.historical import HistoricalAgentResult
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
    SourceRelevance,
    WebResearch,
    WebSource,
)
from schemas.prediction import PredictionResult
from schemas.risk import RiskResult
from utils.config import ConfigError, Settings
from utils.logging import PipelineClock, StageTimer
from utils.paths import data_path
from web.search import WebResearcher, generate_queries
from weights_db import load_weights_db, resolve_config, resolve_industry

log = logging.getLogger("market_intel_agent.pipeline")


class Pipeline:
    """Orchestrates Phase 1 collection -> Phase 2-4 -> synthesis."""

    def __init__(
        self,
        settings: Settings,
        verbose: bool = False,
        company_names: list[str] | None = None,
    ) -> None:
        self.settings = settings
        self.verbose = verbose
        self.company_names = company_names or []
        self.clock = PipelineClock()
        _db_path = data_path("weights_db.json")
        self._db = load_weights_db(_db_path) if _db_path.exists() else {}

    async def run(self, ticker: str) -> FinalReport:
        import time

        t_start = time.perf_counter()
        if not self.settings.has_alpaca_credentials:
            raise ConfigError(
                "Missing Alpaca credentials. Set ALPACA_API_KEY and "
                "ALPACA_API_SECRET in your environment or .env file."
            )

        self.clock.mark("Starting")

        # Resolve industry-specific settings and tuning for this ticker
        settings_i, tuning_i = resolve_config(ticker, self._db, self.settings)

        # ---------------------------------------------------------------
        # PHASE 1 — parallel data collection (news + market + historical)
        # ---------------------------------------------------------------
        t_phase1 = time.perf_counter()
        with StageTimer("Phase 1 - parallel data collection", log):
            phase1 = await self._phase1(ticker)
        phase1_ms = (time.perf_counter() - t_phase1) * 1000

        news_result = phase1["news"]
        market = phase1["market"]
        historical = phase1["historical"]

        self.clock.mark(
            f"Phase 1 complete: {len(news_result.articles)} articles, "
            f"{historical.bars_count} bars"
        )

        if not news_result.articles:
            return self._empty_report(
                ticker, "No relevant news found for the lookback window.", historical
            )

        article = news_result.primary_article
        analysis = news_result.primary_analysis
        if article is None or analysis is None:  # pragma: no cover - defensive
            article, analysis = news_result.articles[0], _fallback_initial(
                ticker, news_result.articles[0]
            )

        # ---------------------------------------------------------------
        # Web research (gated behind enable_web_research)
        # ---------------------------------------------------------------
        if self.settings.enable_web_research and analysis.needs_web_research:
            web = await self._research(ticker, article, analysis)
        else:
            web = WebResearch(performed=False)

        # ---------------------------------------------------------------
        # PHASE 2 — sequential prediction (deterministic, with tuning)
        # ---------------------------------------------------------------
        t_phase2 = time.perf_counter()
        prediction_agent = PredictionAgent()
        with StageTimer("Phase 2 - prediction", log):
            prediction = await prediction_agent.run(phase1, tuning=tuning_i)
        phase2_ms = (time.perf_counter() - t_phase2) * 1000

        # ---------------------------------------------------------------
        # PHASE 3 — sequential risk (options chain + deterministic sizing)
        # ---------------------------------------------------------------
        t_phase3 = time.perf_counter()
        risk_agent = RiskAgent(settings_i)
        with StageTimer("Phase 3 - risk", log):
            risk = await risk_agent.run(phase1, prediction, tuning=tuning_i)
        phase3_ms = (time.perf_counter() - t_phase3) * 1000

        # ---------------------------------------------------------------
        # PHASE 4 — sequential decision (deterministic, with tuning)
        # ---------------------------------------------------------------
        t_phase4 = time.perf_counter()
        decision_agent = DecisionAgent(settings_i)
        with StageTimer("Phase 4 - decision", log):
            decision = await decision_agent.run(
                phase1, prediction, risk, tuning=tuning_i
            )
        phase4_ms = (time.perf_counter() - t_phase4) * 1000

        # ---------------------------------------------------------------
        # Final LLM synthesis (gated behind enable_llm_synthesis)
        # ---------------------------------------------------------------
        synthesis: FinalSynthesis | None = None
        if self.settings.enable_llm_synthesis:
            with StageTimer("Final analysis complete", log):
                try:
                    synthesis = await self._synthesize(
                        ticker,
                        article,
                        analysis,
                        web,
                        market,
                        historical,
                        prediction,
                        risk,
                        decision,
                    )
                except OllamaError as exc:
                    log.warning("Final synthesis failed: %s", exc)

        report = self._build_report(
            ticker,
            article,
            analysis,
            web,
            market,
            historical,
            synthesis,
            prediction,
            risk,
            decision,
        )

        total_ms = (time.perf_counter() - t_start) * 1000
        report.benchmark = {
            "phase1_ms": round(phase1_ms, 1),
            "phase2_ms": round(phase2_ms, 1),
            "phase3_ms": round(phase3_ms, 1),
            "phase4_ms": round(phase4_ms, 1),
            "total_ms": round(total_ms, 1),
            "industry": resolve_industry(ticker),
            "weights_db_active": bool(self._db),
        }

        self.clock.mark("Done")
        return report

    # ------------------------------------------------------------------
    # Phase helpers
    # ------------------------------------------------------------------
    async def _phase1(self, ticker: str) -> dict[str, Any]:
        news_agent = NewsCollectionAgent(
            self.settings, verbose=self.verbose, company_names=self.company_names
        )
        market_agent = MarketDataAgent(self.settings)
        historical_agent = HistoricalAgent(self.settings)

        with StageTimer("Phase 1 agents fetched", log):
            news, market, historical = await asyncio.gather(
                news_agent.run(ticker),
                market_agent.run(ticker),
                historical_agent.run(ticker),
                return_exceptions=True,
            )

        if isinstance(news, BaseException):
            log.warning("News agent failed: %s", news)
            news = NewsCollectionResult(ticker=ticker, errors=[str(news)])
        if isinstance(market, BaseException):
            log.warning("Market agent failed: %s", market)
            market = MarketData()
        if isinstance(historical, BaseException):
            log.warning("Historical agent failed: %s", historical)
            historical = HistoricalAgentResult(symbol=ticker, status="partial")

        return {"news": news, "market": market, "historical": historical}

    async def _research(
        self,
        ticker: str,
        article: NewsArticle,
        analysis: InitialAnalysis,
    ) -> WebResearch:
        researcher = WebResearcher(self.settings)
        queries = generate_queries(
            ticker,
            article.headline,
            analysis.research_questions,
            self.company_names,
        )
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

        to_fetch = _rank_results(results)[: self.settings.max_fetch_pages]
        fetched = await asyncio.gather(
            *(researcher.fetch(r.url) for r in to_fetch), return_exceptions=True
        )

        pages: list[FetchedPage] = []
        fetched_urls: set[str] = set()
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
        results: list,
        pages: list,
        fetched_urls: set[str],
    ) -> WebResearch:
        sources: list[WebSource] = []
        for r in results:
            relevance = SourceRelevance.HIGH if r.url in fetched_urls else SourceRelevance.MEDIUM
            sources.append(
                WebSource(title=r.title, source=_domain_of(r.url), url=r.url, relevance=relevance)
            )
        seen = set()
        unique_sources: list[WebSource] = []
        for s in sources:
            if s.url and s.url not in seen:
                seen.add(s.url)
                unique_sources.append(s)

        key_findings = [
            f"{p.title or p.url}: {p.content[:200].strip()}" for p in pages if p.content
        ][:5]

        return WebResearch(performed=True, key_findings=key_findings, sources=unique_sources)

    async def _synthesize(
        self,
        ticker: str,
        article: NewsArticle,
        initial: InitialAnalysis,
        web: WebResearch,
        market: MarketData,
        historical: HistoricalAgentResult,
        prediction: PredictionResult,
        risk: RiskResult,
        decision: DecisionResult,
    ) -> FinalSynthesis:
        async with OllamaClient(self.settings) as ollama:
            return await ollama.synthesize(
                ticker=ticker,
                news_block=_news_block(article),
                initial_block=_initial_block(initial),
                web_block=_web_block(web),
                market_block=_market_block(market),
                historical_block=_historical_block(historical),
                decision_block=_decision_block(prediction, risk, decision),
                performed=web.performed,
            )

    # ------------------------------------------------------------------
    # Report assembly
    # ------------------------------------------------------------------
    def _build_report(
        self,
        ticker: str,
        article: NewsArticle,
        initial: InitialAnalysis,
        web: WebResearch,
        market: MarketData,
        historical: HistoricalAgentResult,
        synthesis: FinalSynthesis | None,
        prediction: PredictionResult,
        risk: RiskResult,
        decision: DecisionResult,
    ) -> FinalReport:
        now = datetime.now(timezone.utc).isoformat()

        if synthesis is None:
            synthesis = _fallback_synthesis(initial, web)

        market_context = MarketContext(
            price=round(market.price, 2),
            return_1d=round(market.return_1d, 4),
            return_5d=round(market.return_5d, 4),
            sma20=round(market.sma20, 2),
            sma50=round(market.sma50, 2),
            rsi14=round(market.rsi14, 2),
            volume_vs_average=round(market.volume_vs_average, 2),
            trend=(
                synthesis.market_trend if synthesis.market_trend != Trend.UNCERTAIN else _suggest_trend(market)
            ),
        )

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
            historical=historical,
            analysis=synthesis.analysis,
            council_input=synthesis.council_input,
            prediction=prediction,
            risk=risk,
            decision=decision,
        )

    def _empty_report(
        self, ticker: str, message: str, historical: HistoricalAgentResult | None = None
    ) -> FinalReport:
        self.clock.mark("Done")
        now = datetime.now(timezone.utc).isoformat()
        synthesis = _fallback_synthesis(
            _fallback_initial(ticker, _blank_article(ticker)), None
        )
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
            historical=historical or HistoricalAgentResult(),
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
    initial: InitialAnalysis, web: WebResearch | None
) -> FinalSynthesis:
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


def _web_block(web: WebResearch) -> str:
    if web is None or not web.performed:
        return "No web research was performed."
    return json.dumps(web.model_dump(), indent=2)


def _market_block(market: MarketData) -> str:
    return json.dumps(market.model_dump(), indent=2)


def _historical_block(historical: HistoricalAgentResult) -> str:
    if historical is None or not historical.bars_count:
        return "No historical data available."

    def summarize(d: Any) -> str:
        return json.dumps(d, default=str) if d else "(none)"

    parts = [
        f"Historical trends: {summarize(historical.historical_trends)}",
        f"Technical indicator bundle: {summarize(historical.technical)}",
        f"Volatility analysis: {summarize(historical.volatility)}",
        f"Risk statistics: {summarize(historical.risk)}",
        f"Technical summary: {summarize(historical.summary)}",
    ]
    if historical.patterns:
        parts.append(f"Chart patterns: {summarize(historical.patterns)}")
    return "\n".join(parts)


def _decision_block(
    prediction: PredictionResult, risk: RiskResult, decision: DecisionResult
) -> str:
    """Concise, deterministic context of Phases 2-4 for the final synthesis."""
    pred = {
        "price_forecast": prediction.price_forecast,
        "price_forecast_high": prediction.price_forecast_high,
        "price_forecast_low": prediction.price_forecast_low,
        "expected_move_pct": prediction.expected_move_pct,
        "composite_signal": prediction.composite_signal,
        "momentum_score": prediction.momentum_score,
        "adjusted_momentum": prediction.adjusted_momentum,
        "iv_forecast": prediction.iv_forecast,
        "vol_regime": prediction.vol_regime,
        "confidence": prediction.confidence,
    }
    risk_slim = {
        "risk_score": risk.risk_score,
        "risk_level": risk.risk_level,
        "risk_reward_ratio": risk.risk_reward_ratio,
        "stop_loss_level": risk.stop_loss_level,
        "take_profit_level": risk.take_profit_level,
        "greeks_source": risk.greeks_source,
        "iv_source": risk.iv_source,
        "spread_pct": risk.spread_pct,
    }
    decision_slim = {
        "trade_decision": decision.trade_decision,
        "confidence_score": decision.confidence_score,
        "composite_bias": decision.composite_bias,
        "agreement_score": decision.agreement_score,
        "instrument": decision.instrument,
        "option_type": decision.option_type,
        "entry_price": decision.entry_price,
        "stop_loss": decision.stop_loss,
        "take_profit": decision.take_profit,
        "rationale": decision.rationale,
    }
    return json.dumps(
        {
            "Phase 2 Prediction": pred,
            "Phase 3 Risk": risk_slim,
            "Phase 4 Decision": decision_slim,
        },
        default=str,
    )


def _suggest_trend(md: MarketData) -> Trend:
    """Deterministic default trend from price vs moving averages and RSI."""
    if md.price <= 0:
        return Trend.UNCERTAIN
    if md.sma20 and md.sma50:
        above = md.price > md.sma20 > md.sma50
        below = md.price < md.sma20 < md.sma50
    elif md.sma20:
        above = md.price > md.sma20
        below = md.price < md.sma20
    else:
        above = below = False

    if above and md.rsi14 >= 50:
        return Trend.BULLISH
    if below and md.rsi14 <= 50:
        return Trend.BEARISH
    return Trend.NEUTRAL


# ---------------------------------------------------------------------------
# Search result ranking helpers
# ---------------------------------------------------------------------------
def _dedupe_results(results: list) -> list:
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


def _rank_results(results: list) -> list:
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
