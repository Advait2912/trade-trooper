"""Historical news sentiment cache (FinBERT-scored) for the backtest.

Stores, per ticker:

- ``news_articles`` — raw articles fetched from Alpaca.
- ``article_sentiment`` — a continuous score ``P(pos) - P(neg)`` in [-1, 1]
  plus a label, one row per article (written once, idempotent).
- ``daily_sentiment`` — the mean article score aggregated per trading day.

The ``decision_date`` of an article is the trading day whose close it precedes,
using a fixed 21:00 UTC close proxy: the calendar date of ``created_at + 3h``.
This is a 24-hour lookback ending at the close, with no look-ahead into the
next bar.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from alpaca.client import AlpacaClient
from alpaca.news import NEWS_ENDPOINT, normalize_headline, parse_alpaca_news
from schemas.news import NewsArticle

_CLOSE_HOUR_SHIFT = 3  # 21:00 UTC close -> +3h turns the boundary into midnight

_SCHEMA = """
CREATE TABLE IF NOT EXISTS news_articles (
    id INTEGER PRIMARY KEY,
    ticker TEXT NOT NULL,
    created_at TEXT NOT NULL,
    headline TEXT,
    summary TEXT,
    source TEXT,
    symbols TEXT
);
CREATE TABLE IF NOT EXISTS article_sentiment (
    article_id INTEGER PRIMARY KEY,
    score REAL,
    label TEXT
);
CREATE TABLE IF NOT EXISTS daily_sentiment (
    ticker TEXT,
    date TEXT,
    score REAL,
    label TEXT,
    article_count INTEGER,
    PRIMARY KEY (ticker, date)
);
"""


def _parse_dt(value: str) -> datetime | None:
    if not value:
        return None
    try:
        normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
        dt = datetime.fromisoformat(normalized)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except (ValueError, TypeError):
        return None


def decision_date(created_at: str) -> str:
    """Return the trading day (YYYY-MM-DD) whose close this article precedes."""
    dt = _parse_dt(created_at)
    if dt is None:
        return ""
    return (dt + timedelta(hours=_CLOSE_HOUR_SHIFT)).date().isoformat()


def _label(score: float) -> str:
    if score > 0.15:
        return "bullish"
    if score < -0.15:
        return "bearish"
    return "neutral"


def aggregate_daily(dated_scores: list[tuple[str, float]]) -> dict[str, dict[str, Any]]:
    """Aggregate (decision_date, score) pairs into per-day means."""
    buckets: dict[str, list[float]] = {}
    for date, score in dated_scores:
        if date:
            buckets.setdefault(date, []).append(score)

    result: dict[str, dict[str, Any]] = {}
    for date, scores in buckets.items():
        score = sum(scores) / len(scores)
        result[date] = {
            "score": round(score, 4),
            "label": _label(score),
            "article_count": len(scores),
        }
    return result


class NewsCache:
    """SQLite-backed news sentiment cache."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path))
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    @property
    def path(self) -> Path:
        return self._path

    # ---- writes ----
    def add_articles(self, ticker: str, articles: list[NewsArticle]) -> int:
        """Insert articles (idempotent by id); returns the count of new rows."""
        added = 0
        for a in articles:
            cur = self._conn.execute(
                "INSERT OR IGNORE INTO news_articles "
                "(id, ticker, created_at, headline, summary, source, symbols) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (a.id, ticker, a.created_at, a.headline, a.summary, a.source,
                 ",".join(a.symbols)),
            )
            added += cur.rowcount
        self._conn.commit()
        return added

    def unscored(self) -> list[dict[str, Any]]:
        """Return articles that still need a sentiment score."""
        rows = self._conn.execute(
            "SELECT a.id, a.headline, a.summary FROM news_articles a "
            "LEFT JOIN article_sentiment s ON s.article_id = a.id "
            "WHERE s.article_id IS NULL"
        ).fetchall()
        return [{"id": r[0], "headline": r[1] or "", "summary": r[2] or ""} for r in rows]

    def set_sentiment(self, article_id: int, score: float, label: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO article_sentiment (article_id, score, label) "
            "VALUES (?, ?, ?)",
            (article_id, score, label),
        )
        self._conn.commit()

    def set_daily(self, ticker: str, daily: dict[str, dict[str, Any]]) -> None:
        for date, d in daily.items():
            self._conn.execute(
                "INSERT OR REPLACE INTO daily_sentiment "
                "(ticker, date, score, label, article_count) VALUES (?, ?, ?, ?, ?)",
                (ticker, date, d["score"], d["label"], d["article_count"]),
            )
        self._conn.commit()

    def dated_scores(self, ticker: str) -> list[tuple[str, float]]:
        """Return (decision_date, score) for every scored article of a ticker."""
        rows = self._conn.execute(
            "SELECT a.created_at, s.score FROM news_articles a "
            "JOIN article_sentiment s ON s.article_id = a.id WHERE a.ticker = ?",
            (ticker,),
        ).fetchall()
        return [(decision_date(r[0]), float(r[1])) for r in rows]

    # ---- reads ----
    def load_daily_map(self, ticker: str) -> dict[str, dict[str, Any]]:
        """Return {date: {score, label, article_count}} for a ticker."""
        rows = self._conn.execute(
            "SELECT date, score, label, article_count FROM daily_sentiment WHERE ticker = ?",
            (ticker,),
        ).fetchall()
        return {
            r[0]: {"score": float(r[1]), "label": r[2], "article_count": int(r[3])}
            for r in rows
        }


async def fetch_news_range(
    client: AlpacaClient, ticker: str, start: str, end: str
) -> list[NewsArticle]:
    """Fetch all (paginated) news for a ticker over [start, end], deduplicated."""
    articles: list[NewsArticle] = []
    page_token: str | None = None
    while True:
        params: dict[str, Any] = {
            "symbols": ticker,
            "start": start,
            "end": end,
            "limit": 50,
            "sort": "asc",
        }
        if page_token:
            params["page_token"] = page_token
        payload = await client.get_json(NEWS_ENDPOINT, params=params)
        articles.extend(parse_alpaca_news(payload))
        page_token = payload.get("next_page_token") if isinstance(payload, dict) else None
        if not page_token:
            break

    seen: set[str] = set()
    deduped: list[NewsArticle] = []
    for a in articles:
        key = normalize_headline(a.headline)
        if key and key not in seen:
            seen.add(key)
            deduped.append(a)
    return deduped
