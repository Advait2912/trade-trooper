"""Command-line entry point for the market intelligence agent.

Usage:
    python main.py NVDA
    python main.py NVDA --news-limit 5 --lookback-hours 24 --verbose
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Optional

from agent.pipeline import Pipeline
from agent.schemas import FinalReport
from utils.config import ConfigError, load_settings, validate_ticker
from utils.logging import setup_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="market_intel_agent",
        description="Research-only market intelligence agent (never trades).",
    )
    parser.add_argument("ticker", help="Stock ticker, e.g. NVDA")
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


async def _run(ticker: str, args: argparse.Namespace) -> FinalReport:
    settings = load_settings()
    if args.news_limit is not None:
        settings.news_limit = args.news_limit
    if args.lookback_hours is not None:
        settings.lookback_hours = args.lookback_hours

    pipeline = Pipeline(settings, verbose=args.verbose)
    return await pipeline.run(ticker)


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    setup_logging(verbose=args.verbose)

    try:
        ticker = validate_ticker(args.ticker)
    except ConfigError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    try:
        report = asyncio.run(_run(ticker, args))
    except ConfigError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130

    print(render_report(report, verbose=args.verbose))
    print("\nMACHINE-READABLE JSON:")
    print(json.dumps(report.model_dump(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
