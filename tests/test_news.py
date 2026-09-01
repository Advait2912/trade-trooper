"""Deterministic news filtering and normalization."""

from datetime import datetime, timedelta, timezone

from alpaca.news import (
    filter_articles,
    normalize_headline,
    parse_alpaca_news,
)
from schemas.news import NewsArticle


def _article(headline: str, symbols=None, created_at=None, summary="", source="x"):
    return NewsArticle(
        id=abs(hash(headline)) % 10_000_000,
        headline=headline,
        summary=summary,
        source=source,
        url=f"https://example.com/{abs(hash(headline))}",
        created_at=created_at or datetime.now(timezone.utc).isoformat(),
        updated_at="",
        symbols=symbols or ["NVDA"],
    )


def test_normalize_headline_dedup():
    a = normalize_headline("Nvidia signs GPU deal!")
    b = normalize_headline("Nvidia  Signs GPU deal")
    assert a == b


def test_parse_alpaca_news_normalizes():
    payload = {
        "news": [
            {
                "id": 1,
                "headline": "Nvidia signs deal",
                "summary": "Summary",
                "source": "s",
                "url": "https://e.com",
                "created_at": "2026-01-01T00:00:00Z",
                "updated_at": "",
                "symbols": ["NVDA", "AMD"],
            }
        ]
    }
    articles = parse_alpaca_news(payload)
    assert len(articles) == 1
    assert articles[0].symbols == ["NVDA", "AMD"]


def test_parse_malformed_returns_empty():
    assert parse_alpaca_news({"news": [{"id": "not-an-int", "headline": "x"}]}) == []
    assert parse_alpaca_news(None) == []
    assert parse_alpaca_news({"news": [1, 2, 3]}) == []


def test_highly_relevant_kept():
    now = datetime.now(timezone.utc)
    articles = [
        _article("Nvidia signs a major new GPU supply agreement"),
        _article("Jensen Huang autograph auction", summary="celebrity memorabilia"),
    ]
    out = filter_articles(articles, "NVDA", [], lookback_hours=24, now=now)
    assert len(out) == 1
    assert "GPU supply" in out[0].headline


def test_irrelevant_ticker_mention_dropped():
    # Not tagged with the ticker AND doesn't mention it in text.
    articles = [
        _article("Completely unrelated story about coffee", symbols=["SBUX"]),
    ]
    out = filter_articles(articles, "NVDA", [], 24)
    assert out == []


def test_duplicate_headlines_collapsed():
    articles = [
        _article("Nvidia signs a major new GPU supply agreement"),
        _article("Nvidia signs a major new GPU supply agreement"),
    ]
    out = filter_articles(articles, "NVDA", [], 24)
    assert len(out) == 1


def test_old_article_dropped():
    old = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
    articles = [_article("Nvidia signs a major new GPU supply agreement", created_at=old)]
    out = filter_articles(articles, "NVDA", [], lookback_hours=24)
    assert out == []


def test_empty_feed():
    assert filter_articles([], "NVDA", [], 24) == []
