"""Async HTTP client for the Alpaca Market Data API.

Handles authentication, retries with backoff, rate limits, network failures
and malformed responses. No order/trading endpoints are ever called.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

import httpx

from utils.config import Settings

log = logging.getLogger("market_intel_agent.alpaca")

DATA_BASE_URL = "https://data.alpaca.markets"

_MAX_RETRIES = 8
_RETRY_STATUS_CODES = {429, 500, 502, 503, 504}


class AlpacaError(RuntimeError):
    """Base class for Alpaca client errors."""


class AlpacaAuthError(AlpacaError):
    """Authentication / permission failure (401, 403)."""


class AlpacaRateLimitError(AlpacaError):
    """Rate limit exceeded (429) after exhausting retries."""


class AlpacaAPIError(AlpacaError):
    """Unexpected API or malformed-response error."""


class AlpacaNetworkError(AlpacaError):
    """Network / connection failure."""


class AlpacaClient:
    """Small wrapper around httpx.AsyncClient for the Alpaca data API."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._headers = {
            "APCA-API-KEY-ID": settings.alpaca_api_key,
            "APCA-API-SECRET-KEY": settings.alpaca_api_secret,
        }
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self) -> "AlpacaClient":
        self._client = httpx.AsyncClient(
            base_url=DATA_BASE_URL,
            headers=self._headers,
            timeout=self._settings.http_timeout,
        )
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _require_client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise AlpacaError("AlpacaClient used outside of async context manager.")
        return self._client

    async def get_json(self, url: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """GET `url` and return parsed JSON, with retry/backoff on transient errors."""
        client = self._require_client()
        last_exc: Optional[Any] = None

        for attempt in range(_MAX_RETRIES):
            try:
                resp = await client.get(url, params=params)
            except httpx.HTTPError as exc:
                last_exc = exc
                log.warning("Alpaca request failed (network): %s", exc)
                await asyncio.sleep(_backoff(attempt))
                continue

            if resp.status_code in (401, 403):
                raise AlpacaAuthError(
                    f"Alpaca authentication failed ({resp.status_code}). "
                    "Check ALPACA_API_KEY / ALPACA_API_SECRET."
                )

            if resp.status_code in _RETRY_STATUS_CODES and attempt < _MAX_RETRIES - 1:
                last_exc = resp
                log.warning(
                    "Alpaca transient error %s, retrying (%d/%d)",
                    resp.status_code,
                    attempt + 1,
                    _MAX_RETRIES,
                )
                await asyncio.sleep(_backoff(attempt))
                continue

            if resp.status_code >= 400:
                if resp.status_code == 429:
                    raise AlpacaRateLimitError("Alpaca rate limit exceeded (429).")
                raise AlpacaAPIError(
                    f"Alpaca API error {resp.status_code}: {resp.text[:200]}"
                )

            try:
                return resp.json()
            except (ValueError, TypeError) as exc:
                raise AlpacaAPIError("Alpaca returned malformed JSON.") from exc

        raise AlpacaNetworkError(f"Alpaca request failed: {last_exc!r}")


def _backoff(attempt: int) -> float:
    """Exponential backoff: 1s, 2s, 4s, 8s, ... (rate-limit friendly)."""
    return 1.0 * (2 ** attempt)
