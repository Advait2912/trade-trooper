"""Shared test fixtures and respx mock helpers.

Tests never require real API credentials or a running Ollama server — all HTTP
traffic is intercepted by respx.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

import httpx
import pytest
import respx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config import Settings

NEWS_URL_RE = re.compile(r"https://data\.alpaca\.markets/v1beta1/news.*")
SNAPSHOT_URL = "https://data.alpaca.markets/v2/stocks/{symbol}/snapshot"
BARS_URL_RE = re.compile(r"https://data\.alpaca\.markets/v2/stocks/.*/bars.*")
CHAT_URL = "http://localhost:11434/api/chat"
WEB_SEARCH_URL = "http://localhost:11434/api/experimental/web_search"
WEB_FETCH_URL = "http://localhost:11434/api/experimental/web_fetch"


@pytest.fixture
def settings() -> Settings:
    return Settings(
        alpaca_api_key="test_key",
        alpaca_api_secret="test_secret",
        alpaca_data_feed="iex",
        ollama_base_url="http://localhost:11434",
        ollama_model="gemma4:e4b",
        ollama_api_key="",
        ollama_web_search_url=WEB_SEARCH_URL,
        ollama_web_fetch_url=WEB_FETCH_URL,
    )


# ---------------------------------------------------------------------------
# Alpaca mock helpers
# ---------------------------------------------------------------------------
def _news_items(headlines: List[str]) -> dict:
    now = datetime.now(timezone.utc)
    items = []
    for i, h in enumerate(headlines):
        items.append(
            {
                "id": 1000 + i,
                "headline": h,
                "summary": f"Summary for: {h}",
                "source": "test-source",
                "url": f"https://example.com/{i}",
                "created_at": now.isoformat(),
                "updated_at": now.isoformat(),
                "symbols": ["NVDA"],
            }
        )
    return {"news": items}


def mock_news(headlines: List[str]) -> None:
    respx.get(NEWS_URL_RE).mock(
        return_value=httpx.Response(200, json=_news_items(headlines))
    )


def mock_news_empty() -> None:
    respx.get(NEWS_URL_RE).mock(
        return_value=httpx.Response(200, json={"news": []})
    )


def mock_news_error(status: int = 500) -> None:
    respx.get(NEWS_URL_RE).mock(return_value=httpx.Response(status))


def _bars(symbol: str, n: int = 100) -> List[dict]:
    base = datetime(2026, 1, 1, 13, 0, tzinfo=timezone.utc)
    out = []
    price = 100.0
    for i in range(n):
        day = base + timedelta(days=i)
        o = price
        c = price * 1.01
        h = max(o, c) * 1.005
        l = min(o, c) * 0.995
        v = 1_000_000 + i * 1000
        out.append(
            {
                "t": day.isoformat(),
                "o": round(o, 2),
                "h": round(h, 2),
                "l": round(l, 2),
                "c": round(c, 2),
                "v": v,
            }
        )
        price = c
    return out


def mock_market_data(symbol: str = "NVDA") -> None:
    bars = _bars(symbol)
    last = bars[-1]
    snapshot = {
        "symbol": symbol,
        "latestTrade": {"t": last["t"], "p": last["c"], "s": 123456},
        "latestQuote": {"ap": last["c"], "bp": last["c"]},
        "dailyBar": last,
        "prevDailyBar": bars[-2],
    }
    respx.get(SNAPSHOT_URL.format(symbol=symbol)).mock(
        return_value=httpx.Response(200, json=snapshot)
    )
    respx.get(BARS_URL_RE).mock(
        return_value=httpx.Response(200, json={"bars": bars, "symbol": symbol})
    )


def mock_market_data_error(status: int = 500) -> None:
    respx.get(SNAPSHOT_URL.format(symbol="NVDA")).mock(
        return_value=httpx.Response(status)
    )
    respx.get(BARS_URL_RE).mock(return_value=httpx.Response(status))


# ---------------------------------------------------------------------------
# Ollama mock helpers
# ---------------------------------------------------------------------------
INITIAL_JSON = {
    "ticker": "NVDA",
    "event": "Nvidia signs a major GPU supply agreement.",
    "relevance": 0.9,
    "materiality": "high",
    "sentiment": "bullish",
    "evidence_quality": "medium",
    "needs_web_research": True,
    "research_questions": ["What is the size of the deal?", "Who is the buyer?"],
}

FINAL_JSON = {
    "event": {
        "description": "Nvidia signs a major GPU supply agreement.",
        "relevance": 0.9,
        "materiality": "high",
        "sentiment": "bullish",
    },
    "evidence": {
        "quality": "medium",
        "facts": ["A supply agreement was reported."],
        "inferences": ["Higher future GPU revenue is plausible."],
        "uncertainties": ["Exact contract size is unconfirmed."],
    },
    "web_research": {
        "performed": True,
        "key_findings": ["Multiple outlets corroborate the agreement."],
        "sources": [
            {
                "title": "Press release",
                "source": "example.com",
                "url": "https://example.com/0",
                "relevance": "high",
            }
        ],
    },
    "market_trend": "bullish",
    "analysis": {
        "news_impact": 0.55,
        "actionability": "medium",
        "time_horizon": "1-5_days",
        "confidence": 0.72,
        "summary": "Positive supply news with medium confidence.",
    },
    "council_input": {
        "recommended_bias": "bullish",
        "confidence": 0.68,
        "key_reason": "Material supply agreement reported.",
        "should_council_consider": True,
    },
}


def _ollama_responder(initial: dict, final: dict):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        text = json.dumps(body)
        # "council_input" only appears in the final-synthesis prompt.
        content = json.dumps(final if "council_input" in text else initial)
        return httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": content}, "done": True},
        )

    return handler


def mock_ollama(initial: Optional[dict] = None, final: Optional[dict] = None) -> None:
    respx.post(CHAT_URL).mock(
        side_effect=_ollama_responder(initial or INITIAL_JSON, final or FINAL_JSON)
    )


def mock_ollama_unavailable(status: int = 500) -> None:
    respx.post(CHAT_URL).mock(return_value=httpx.Response(status))


# ---------------------------------------------------------------------------
# Web research mock helpers
# ---------------------------------------------------------------------------
def mock_web_search(results: Optional[List[dict]] = None) -> None:
    data = {"results": results or []}
    respx.post(WEB_SEARCH_URL).mock(return_value=httpx.Response(200, json=data))


def mock_web_search_error(status: int = 500) -> None:
    respx.post(WEB_SEARCH_URL).mock(return_value=httpx.Response(status))


def mock_web_fetch(content: str = "page text") -> None:
    respx.post(WEB_FETCH_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "title": "Fetched page",
                "content": content,
                "links": ["https://example.com/0"],
            },
        )
    )


def mock_web_research_full() -> None:
    mock_web_search(
        [
            {
                "title": "Corroborating report",
                "url": "https://example.com/0",
                "content": "Confirms the supply agreement.",
            }
        ]
    )
    mock_web_fetch("Confirms the supply agreement with details.")
