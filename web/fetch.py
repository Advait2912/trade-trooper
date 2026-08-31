"""Web page fetching and HTML text extraction.

Primary path is Ollama's local ``/api/experimental/web_fetch`` endpoint (which
already returns cleaned text). A direct httpx fetch with a small stdlib HTML
extractor is used as a fallback, with hard caps on how much text is retained
so the local model never receives giant pages.
"""

from __future__ import annotations

import logging
from html.parser import HTMLParser
from typing import Optional

import httpx

from agent.schemas import FetchedPage
from utils.config import Settings

log = logging.getLogger("market_intel_agent.web.fetch")

_SKIP_TAGS = {
    "script",
    "style",
    "noscript",
    "svg",
    "nav",
    "header",
    "footer",
    "aside",
    "form",
    "iframe",
    "template",
}


class _TextExtractor(HTMLParser):
    """Collects visible text while skipping obvious navigation/boilerplate."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_depth = 0
        self.title_parts: list[str] = []
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list) -> None:
        tag = tag.lower()
        if tag == "title":
            self._in_title = True
            self.title_parts = []
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
        elif tag in {"p", "br", "li", "h1", "h2", "h3", "h4", "div", "tr"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag == "title":
            self._in_title = False
        if tag in _SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title_parts.append(data)
        elif self._skip_depth == 0:
            self.parts.append(data)

    def title(self) -> str:
        return " ".join(" ".join(self.title_parts).split())

    def text(self, max_chars: int) -> str:
        raw = " ".join("".join(self.parts).split())
        return raw[:max_chars]


def extract_text(html: str, max_chars: int = 4000) -> tuple[str, str]:
    """Return (title, body_text) extracted from raw HTML, body capped."""
    parser = _TextExtractor()
    try:
        parser.feed(html or "")
    except Exception:  # pragma: no cover - defensive
        return "", ""
    return parser.title(), parser.text(max_chars)


async def fetch_direct(
    url: str, timeout: float, max_chars: int = 4000
) -> Optional[FetchedPage]:
    """Fetch a page directly with httpx and extract its text."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; market-intel-agent/1.0; research)"
        )
    }
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
    except httpx.HTTPError as exc:
        log.warning("Direct fetch failed for %s: %s", url, exc)
        return None

    content_type = resp.headers.get("content-type", "")
    if "text/html" not in content_type and "text/plain" not in content_type:
        log.debug("Skipping non-HTML content at %s", url)
        return None

    html = resp.text
    title, text = extract_text(html, max_chars)
    if not text and "text/plain" in content_type:
        text = html[:max_chars]
    return FetchedPage(url=url, title=title, content=text)


async def fetch_via_ollama(
    url: str, settings: Settings
) -> Optional[FetchedPage]:
    """Fetch a page through Ollama's local web_fetch endpoint."""
    payload = {"url": url}
    headers = {}
    if settings.ollama_api_key:
        headers["Authorization"] = f"Bearer {settings.ollama_api_key}"

    try:
        async with httpx.AsyncClient(timeout=settings.http_timeout) as client:
            resp = await client.post(
                settings.ollama_web_fetch_url, json=payload, headers=headers
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPError as exc:
        log.warning("Ollama web_fetch failed for %s: %s", url, exc)
        return None
    except ValueError:
        log.warning("Ollama web_fetch returned malformed JSON for %s", url)
        return None

    content = (data.get("content") or "")[: settings.max_web_chars_per_page]
    links = data.get("links") or []
    return FetchedPage(
        url=url,
        title=data.get("title") or "",
        content=content,
        links=list(links),
    )
