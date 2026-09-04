"""Build the historical news-sentiment cache (FinBERT on GPU).

Usage (from repo root):

    python scripts/build_news_cache.py NVDA,AMD,SPY --start 2025-01-01 --end 2026-01-01

Fetches each ticker's news over [start, end], scores every article once with
FinBERT (batched, GPU when available), and aggregates a per-trading-day
sentiment into the cache DB consumed by the backtest.  Re-running is idempotent
and only scores newly-added articles.

Requires the optional ML stack: `pip install -r requirements-ml.txt`.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from alpaca.client import AlpacaClient
from trading.news_cache import NewsCache, aggregate_daily, fetch_news_range
from utils.config import load_settings
from utils.paths import data_path


def _load_scorer(device: str | None):
    try:
        from tools.finbert_sentiment import FinBertSentiment
    except ImportError as exc:  # pragma: no cover - optional dependency
        print(
            "FinBERT requires the optional ML stack. Run `pip install -r requirements-ml.txt`.",
            file=sys.stderr,
        )
        raise SystemExit(2) from exc
    return FinBertSentiment(device=device)


async def build_one(
    cache: NewsCache, scorer, ticker: str, start: str, end: str, batch_size: int
) -> dict:
    async with AlpacaClient(load_settings()) as client:
        articles = await fetch_news_range(client, ticker, start, end)

    cache.add_articles(ticker, articles)

    unscored = cache.unscored()
    ids = [a["id"] for a in unscored]
    texts = [f"{a['headline']} {a['summary']}".strip() for a in unscored]
    if texts:
        for aid, res in zip(ids, scorer.score_batch(texts, batch_size=batch_size)):
            cache.set_sentiment(aid, res["score"], res["label"])

    daily = aggregate_daily(cache.dated_scores(ticker))
    cache.set_daily(ticker, daily)

    return {"ticker": ticker, "articles": len(articles), "scored": len(texts),
            "days": len(daily)}


async def run(args: argparse.Namespace) -> int:
    cache = NewsCache(args.cache)
    scorer = _load_scorer(args.device)
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    print(f"Building news cache ({args.start}..{args.end}) for {tickers} -> {args.cache}")
    try:
        for ticker in tickers:
            result = await build_one(cache, scorer, ticker, args.start, args.end, args.batch_size)
            print(f"  {result['ticker']}: {result['articles']} articles, "
                  f"{result['scored']} scored, {result['days']} days")
    finally:
        cache.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="build-news-cache")
    parser.add_argument("tickers", help="comma-separated tickers")
    parser.add_argument("--start", required=True, help="start date YYYY-MM-DD")
    parser.add_argument("--end", default=date.today().isoformat(), help="end date YYYY-MM-DD")
    parser.add_argument("--cache", default=str(data_path("news_cache.db")),
                        help="SQLite cache path (default data/news_cache.db)")
    parser.add_argument("--batch-size", type=int, default=64, help="FinBERT batch size")
    parser.add_argument("--device", default=None, help="torch device (default: cuda if available)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
