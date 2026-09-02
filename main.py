"""Command-line entry point for the market intelligence agent.

Usage:
    python main.py NVDA                       # full 4-phase pipeline
    python main.py NVDA --historical          # Phase 1 historical agent only
    python main.py NVDA --news-limit 5 --lookback-hours 24 --verbose
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from agents.historical_agent import HistoricalAgent
from orchestrator.pipeline import Pipeline
from schemas.historical import HistoricalAgentResult
from schemas.pipeline import FinalReport
from schemas.risk import RiskResult
from utils.config import ConfigError, load_settings, validate_ticker
from utils.logging import setup_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="market_intel_agent",
        description="Agentic market intelligence (Phase 1 collection + Phase 2-4 pipeline).",
    )
    parser.add_argument("ticker", help="Stock ticker, e.g. NVDA")
    parser.add_argument(
        "--historical",
        action="store_true",
        help="Run only the Phase 1 Historical Data Agent and print its result.",
    )
    parser.add_argument(
        "--news-limit",
        type=int,
        default=None,
        help="Max news articles to consider (default from env or 5).",
    )
    parser.add_argument(
        "--lookback-hours",
        type=int,
        default=None,
        help="Only consider news newer than this many hours (default 24).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show selected headlines, queries, URLs and raw model output.",
    )
    return parser


def render_historical(result: HistoricalAgentResult) -> str:
    """Human-readable historical agent result."""
    lines = ["=" * 40, "HISTORICAL DATA AGENT — PHASE 1", "=" * 40]
    lines.append(f"Symbol: {result.symbol}")
    lines.append(f"Status: {result.status}")
    lines.append(f"Bars: {result.bars_count}")
    lines.append("")
    tr = result.historical_trends
    if tr:
        lines.append("HISTORICAL TRENDS")
        lines.append("-" * 16)
        for key, value in tr.items():
            lines.append(f"{key}: {value}")
        lines.append("")
    if result.volatility_history:
        lines.append("VOLATILITY HISTORY (latest)")
        lines.append("-" * 27)
        for pt in result.volatility_history[-5:]:
            lines.append(
                f"{pt.date}  rvol={pt.realized_vol:.2f}%  "
                f"20d={pt.rolling_vol_20d:.2f}%  60d={pt.rolling_vol_60d:.2f}%"
            )
    summary = result.summary
    if summary:
        lines.append("")
        lines.append("TECHNICAL SUMMARY")
        lines.append("-" * 17)
        lines.append(f"Overall signal: {summary.get('overall_signal', 'n/a')}")
        lines.append(summary.get("summary_text", ""))
    if result.errors:
        lines.append("")
        lines.append("ERRORS")
        lines.append("-" * 6)
        for e in result.errors:
            lines.append(f"- {e}")
    return "\n".join(lines)


def _render_risk(risk: RiskResult) -> list[str]:
    """Human-readable PHASE 3 risk block."""
    lines = ["PHASE 3 — RISK", "-" * 13, f"Status: {risk.status}"]
    lines.append(f"Risk level: {risk.risk_level} (score {risk.risk_score:.0f}/100)")
    lines.append(f"IV: {risk.iv_used:.1f}% ({risk.iv_source}; greeks via {risk.greeks_source})")
    if risk.spread_pct:
        lines.append(f"Bid/ask spread: {risk.spread_pct * 100:.1f}%")
    if risk.implied_move_pct:
        lines.append(f"Market-implied move: {risk.implied_move_pct * 100:.2f}%")
    if risk.theta_per_day:
        lines.append(f"Theta/day: {risk.theta_per_day:.4f}")
    lines.append(f"Stop: ${risk.stop_loss_level:.2f}")
    lines.append(f"Target: ${risk.take_profit_level:.2f}")
    lines.append(f"Risk/reward: {risk.risk_reward_ratio:.2f}")
    rec = risk.position_recommendation
    lines.append(
        f"Position: {rec.equity.shares:.0f} shares (${rec.equity.dollar_value:,.0f}) / "
        f"{rec.option.contracts:.0f} contracts (premium risk ${rec.option.premium_risk:,.0f})"
    )
    lines.append(f"Capital at risk: {risk.capital_at_risk_pct * 100:.2f}%")
    lines.append(
        f"Max loss: ${risk.max_loss_dollars:,.0f} ({risk.max_loss_pct * 100:.2f}%)"
    )
    if risk.tail_var_dollars:
        lines.append(
            f"Tail: VaR ${risk.tail_var_dollars:,.0f} / CVaR ${risk.tail_cvar_dollars:,.0f}"
        )
    lines.append("")
    return lines


def render_report(report: FinalReport, verbose: bool = False) -> str:
    """Human-readable console report."""
    def pct(x: float) -> str:
        return f"{x * 100:+.2f}%"

    lines: list[str] = []
    sep = "=" * 40
    lines.append(sep)
    lines.append(f"MARKET INTELLIGENCE REPORT — {report.ticker}")
    lines.append(sep)
    lines.append("")

    news = report.news
    lines.append("LATEST RELEVANT NEWS")
    lines.append("-" * 20)
    lines.append(f"Headline: {news.headline or '(none)'}")
    lines.append(f"Source: {news.source or '(unknown)'}")
    lines.append(f"Published: {news.published_at or '(unknown)'}")
    lines.append("")

    ev = report.event
    lines.append("EVENT")
    lines.append("-" * 5)
    lines.append(ev.description or "(none)")
    lines.append("")
    lines.append("NEWS SENTIMENT")
    lines.append("-" * 14)
    lines.append(ev.sentiment.value.capitalize())
    lines.append("")
    lines.append("NEWS IMPACT")
    lines.append("-" * 11)
    lines.append(f"{report.analysis.news_impact:+.2f}")
    lines.append("")

    wr = report.web_research
    lines.append("WEB RESEARCH")
    lines.append("-" * 12)
    if wr.performed:
        lines.append("✓ Additional sources found")
        n = len(wr.sources)
        lines.append(f"✓ {n} source(s) reviewed")
        findings = wr.key_findings or []
        lines.append(
            "✓ " + ("Event partially verified" if findings else "No strong findings")
        )
        if findings and verbose:
            for f in findings:
                lines.append(f"  • {f}")
    else:
        lines.append("• Not performed (not warranted)")
    lines.append("")

    mc = report.market_context
    lines.append("MARKET")
    lines.append("-" * 6)
    lines.append(f"Price: ${mc.price:.2f}")
    lines.append(f"1D: {pct(mc.return_1d)}")
    lines.append(f"5D: {pct(mc.return_5d)}")
    lines.append(f"RSI: {mc.rsi14:.1f}")
    lines.append(f"SMA20: {mc.sma20:.2f}")
    lines.append(f"SMA50: {mc.sma50:.2f}")
    lines.append(f"Volume vs average: {mc.volume_vs_average:.2f}x")
    lines.append(f"Trend: {mc.trend.value.capitalize()}")
    lines.append("")

    hist = report.historical
    if hist.status != "not_implemented" and hist.bars_count:
        lines.append("HISTORICAL CONTEXT")
        lines.append("-" * 19)
        trend = hist.historical_trends.get("trend", "n/a")
        lines.append(f"Trend: {trend}")
        summary = hist.summary.get("overall_signal", "n/a")
        lines.append(f"Technical signal: {summary}")
        lines.append("")

    phs = [("PHASE 2", report.prediction), ("PHASE 4", report.decision)]
    for label, ph in phs:
        lines.append(label)
        lines.append("-" * len(label))
        lines.append(f"Status: {ph.status}")
        if ph.summary:
            lines.append(ph.summary)
        lines.append("")

    lines.extend(_render_risk(report.risk))

    a = report.analysis
    lines.append("ASSESSMENT")
    lines.append("-" * 10)
    lines.append(a.summary or "(none)")
    lines.append("")
    lines.append("ACTIONABILITY")
    lines.append("-" * 12)
    lines.append(a.actionability.value.capitalize())
    lines.append("")
    lines.append("CONFIDENCE")
    lines.append("-" * 10)
    lines.append(f"{a.confidence:.2f}")
    lines.append("")

    ci = report.council_input
    lines.append("COUNCIL INPUT")
    lines.append("-" * 13)
    lines.append(f"{ci.recommended_bias.value.capitalize()} bias")
    lines.append(f"Confidence: {ci.confidence:.2f}")
    if ci.key_reason:
        lines.append(f"Key reason: {ci.key_reason}")
    lines.append("")

    lines.append(sep)
    return "\n".join(lines)


async def _run_historical(ticker: str) -> HistoricalAgentResult:
    settings = load_settings()
    agent = HistoricalAgent(settings)
    return await agent.run(ticker)


async def _run(ticker: str, args: argparse.Namespace) -> FinalReport:
    settings = load_settings()
    if args.news_limit is not None:
        settings.news_limit = args.news_limit
    if args.lookback_hours is not None:
        settings.lookback_hours = args.lookback_hours

    pipeline = Pipeline(settings, verbose=args.verbose)
    return await pipeline.run(ticker)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    setup_logging(verbose=args.verbose)

    try:
        ticker = validate_ticker(args.ticker)
    except ConfigError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    try:
        if args.historical:
            result = asyncio.run(_run_historical(ticker))
            print(render_historical(result))
            print("\nMACHINE-READABLE JSON:")
            print(json.dumps(result.model_dump(), indent=2, default=str))
            return 0
        report = asyncio.run(_run(ticker, args))
    except ConfigError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130

    print(render_report(report, verbose=args.verbose))
    print("\nMACHINE-READABLE JSON:")
    print(json.dumps(report.model_dump(), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
