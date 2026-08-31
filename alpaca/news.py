"""Alpaca News API — fetching and deterministic pre-filtering."""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from agent.schemas import NewsArticle
from alpaca.client import AlpacaClient

NEWS_ENDPOINT = "/v1beta1/news"

# Normalization helpers for duplicate detection.
_WS_RE = re.compile(r"\s+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")

# Entertainment / memorabilia / clearly non-material keywords. Used only as a
# weak negative signal in combination with ticker/company matching; never the
# sole reason to drop an article.
_SOFT_IRRELEVANT_KEYWORDS = (
    "autograph",
    "memorabilia",
    "collectible",
    "celebrity",
    "fan art",
    "merch",
    "merchandise",
    "hall of fame",
    "nostalgia",
    "auction",
    "giveaway",
)


def normalize_headline(headline: str) -> str:
    """Lowercase, strip punctuation/whitespace for duplicate comparison."""
    text = headline.lower().strip()
    text = _WS_RE.sub(" ", text)
    text = _NON_ALNUM_RE.sub("", text)
    return text


def parse_alpaca_news(payload: Optional[dict]) -> List[NewsArticle]:
    """Normalize raw Alpaca news JSON into NewsArticle models.

    Returns an empty list on missing/malformed payloads rather than raising.
    """
    items = payload.get("news", []) if isinstance(payload, dict) else []
    articles: List[NewsArticle] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            articles.append(
                NewsArticle(
                    id=int(item.get("id", 0)),
                    headline=(item.get("headline") or "").strip(),
                    summary=(item.get("summary") or "").strip(),
                    source=(item.get("source") or "").strip(),
                    url=(item.get("url") or "").strip(),
                    created_at=item.get("created_at") or "",
                    updated_at=item.get("updated_at") or "",
                    symbols=[s for s in (item.get("symbols") or []) if s],
                )
            )
        except (ValueError, TypeError):
            continue
    return articles


def filter_articles(
    articles: List[NewsArticle],
    ticker: str,
    company_names: List[str],
    lookback_hours: int = 24,
    now: Optional[datetime] = None,
) -> List[NewsArticle]:
    """Deterministic, ticker-agnostic relevance filtering.

    Drops articles that are:
      - duplicates (normalized headline match)
      - older than the lookback window
      - not actually tagged with the ticker or an obvious company/entity name
      - weakly tagged and matching soft-irrelevant keyword signals

    Returns articles ordered newest-first.
    """
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=lookback_hours)
    ticker = ticker.upper()
    names = [n.lower() for n in company_names if n]

    seen_headlines = set()
    results: List[NewsArticle] = []

    for article in sorted(
        articles,
        key=lambda a: _parse_time(a.created_at) or _parse_time(a.updated_at) or now,
        reverse=True,
    ):
        if not article.headline:
            continue

        # Recency filter.
        pub = _parse_time(article.created_at) or _parse_time(article.updated_at)
        if pub is not None and pub < cutoff:
            continue

        # Duplicate detection on normalized headline.
        key = normalize_headline(article.headline)
        if not key:
            continue
        if key in seen_headlines:
            continue
        seen_headlines.add(key)

        # Ticker / entity match.
        tagged_with_ticker = ticker in {s.upper() for s in article.symbols}
        headline_l = article.headline.lower()
        summary_l = article.summary.lower()
        mentions_ticker = _mentions(headline_l + " " + summary_l, ticker)
        mentions_company = any(name in headline_l for name in names) or any(
            name in summary_l for name in names
        )

        if not (tagged_with_ticker or mentions_ticker or mentions_company):
            continue

        # Soft-irrelevant signal: only drop when it is NOT clearly financial.
        if _is_soft_irrelevant(headline_l + " " + summary_l):
            continue

        results.append(article)

    return results


def _mentions(text: str, ticker: str) -> bool:
    # Whole-word match so "APP" doesn't match "APPLE".
    return bool(re.search(rf"\b{re.escape(ticker.lower())}\b", text))


def _is_soft_irrelevant(text: str) -> bool:
    has_soft = any(kw in text for kw in _SOFT_IRRELEVANT_KEYWORDS)
    if not has_soft:
        return False
    # If the article also talks about money/market signals, keep it.
    financial = (
        "revenue",
        "earnings",
        "guidance",
        "supply",
        "agreement",
        "contract",
        "deal",
        "acquisition",
        "merger",
        "financing",
        "stake",
        "shares",
        "stock",
        "profit",
        "loss",
        "investment",
        "chips",
        "gpu",
        "data center",
        "datacenter",
        "partnership",
    )
    return not any(kw in text for kw in financial)


def _parse_time(value: str) -> Optional[datetime]:
    if not value:
        return None
    try:
        # Handle trailing 'Z' -> UTC and offset-aware ISO strings.
        normalized = value
        if normalized.endswith("Z"):
            normalized = normalized[:-1] + "+00:00"
        dt = datetime.fromisoformat(normalized)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except (ValueError, TypeError):
        return None


async def fetch_news(
    client: AlpacaClient,
    ticker: str,
    limit: int,
    lookback_hours: int,
    company_names: List[str],
) -> List[NewsArticle]:
    """Fetch recent news for `ticker` and apply deterministic filtering."""
    params = {
        "symbols": ticker,
        "limit": max(limit * 5, 50),  # over-fetch, then filter down
        "sort": "desc",
    }
    payload = await client.get_json(NEWS_ENDPOINT, params=params)
    articles = parse_alpaca_news(payload)
    filtered = filter_articles(articles, ticker, company_names, lookback_hours)
    return filtered[:limit]
