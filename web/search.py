"""Web search via Ollama's local experimental search endpoint.

The controller decides when and what to search; this module only exposes the
low-level primitives plus deterministic query generation.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import httpx

from schemas.pipeline import FetchedPage, SearchResult
from utils.config import Settings
from web import fetch as web_fetch

log = logging.getLogger("market_intel_agent.web.search")


class WebResearcher:
    """Search and fetch primitives backed by Ollama's web search API."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        """Perform a single web search. Returns [] on any failure."""
        headers = {}
        if self._settings.ollama_api_key:
            headers["Authorization"] = f"Bearer {self._settings.ollama_api_key}"
        payload = {"query": query, "max_results": max_results}

        try:
            async with httpx.AsyncClient(
                timeout=self._settings.http_timeout
            ) as client:
                resp = await client.post(
                    self._settings.ollama_web_search_url,
                    json=payload,
                    headers=headers,
                )
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError as exc:
            log.warning("Web search failed for query %r: %s", query, exc)
            return []
        except ValueError:
            log.warning("Web search returned malformed JSON for query %r", query)
            return []

        results: List[SearchResult] = []
        for item in data.get("results", []) or []:
            if not isinstance(item, dict):
                continue
            results.append(
                SearchResult(
                    title=item.get("title") or "",
                    url=item.get("url") or "",
                    content=(item.get("content") or "")[
                        : self._settings.max_web_chars_per_page
                    ],
                )
            )
        return results

    async def fetch(self, url: str) -> Optional[FetchedPage]:
        """Fetch a page, preferring Ollama's web_fetch and falling back to direct."""
        page = await web_fetch.fetch_via_ollama(url, self._settings)
        if page is not None and page.content:
            return page
        return await web_fetch.fetch_direct(
            url,
            self._settings.http_timeout,
            self._settings.max_web_chars_per_page,
        )


def generate_queries(
    ticker: str,
    headline: str,
    research_questions: List[str],
    company_names: List[str],
) -> List[str]:
    """Deterministically generate a small, targeted set of search queries.

    Combines the LLM's research questions with templated primary-source
    queries (company IR, SEC) and a headline-derived natural query. Ticker is
    always included to keep results on-topic.
    """
    queries: List[str] = []
    ticker = ticker.upper()
    headline = (headline or "").strip()

    # LLM-provided research questions (highest priority, most specific).
    for q in research_questions:
        q = q.strip()
        if q and len(q) < 200:
            queries.append(q if ticker.lower() in q.lower() else f"{ticker} {q}")

    # Headline-derived natural query.
    if headline:
        queries.append(f"{ticker} {headline[:80]}")

    # Primary-source queries.
    queries.append(f"site:sec.gov {ticker}")
    for name in company_names:
        domain = _domainify(name)
        if domain:
            queries.append(f"site:{domain} {ticker}")

    # De-duplicate while preserving order; cap to a reasonable set.
    seen = set()
    out: List[str] = []
    for q in queries:
        key = q.lower().strip()
        if key and key not in seen:
            seen.add(key)
            out.append(q)
    return out[:6]


def _domainify(company_name: str) -> str:
    """Naive company-name -> domain guess for site: queries (best effort)."""
    name = company_name.lower().strip()
    if not name or " " not in name:
        return ""
    tokens = [t for t in name.split() if t.isalnum()]
    if not tokens:
        return ""
    return tokens[0] + ".com"
