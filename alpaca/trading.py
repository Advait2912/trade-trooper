"""Paper-trading execution client.

Talks **only** to Alpaca's PAPER trading API (``https://paper-api.alpaca.markets``),
so simulated money is used and real funds can never be touched.  This is the
single place in the codebase that can place/modify/cancel orders and read the
account/positions/portfolio history.

The data client (``alpaca/client.py``) remains strictly data-only.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from utils.config import Settings

log = logging.getLogger("market_intel_agent.trading")

PAPER_TRADING_BASE_URL = "https://paper-api.alpaca.markets"

_MAX_RETRIES = 3
_RETRY_STATUS_CODES = {429, 500, 502, 503, 504}


class TradingError(RuntimeError):
    """Base class for paper-trading errors."""


class TradingAuthError(TradingError):
    """Authentication/permission failure (401, 403)."""


class TradingAPIError(TradingError):
    """Unexpected API or malformed-response error."""


class TradingNetworkError(TradingError):
    """Network / connection failure after retries."""


class TradingClient:
    """Async wrapper for the Alpaca paper-trading REST API."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._headers = {
            "APCA-API-KEY-ID": settings.alpaca_api_key,
            "APCA-API-SECRET-KEY": settings.alpaca_api_secret,
        }
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "TradingClient":
        self._client = httpx.AsyncClient(
            base_url=PAPER_TRADING_BASE_URL,
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
            raise TradingError("TradingClient used outside of async context manager.")
        return self._client

    # ------------------------------------------------------------------
    # Read endpoints
    # ------------------------------------------------------------------
    async def get_account(self) -> dict[str, Any]:
        """Return the paper account (equity, buying_power, cash, etc.)."""
        return await self._request("GET", "/v2/account")

    async def get_positions(self) -> list[dict[str, Any]]:
        """Return all open positions."""
        payload = await self._request("GET", "/v2/positions")
        return payload if isinstance(payload, list) else []

    async def get_position(self, symbol: str) -> dict[str, Any] | None:
        """Return a single position or None if not held."""
        try:
            return await self._request("GET", f"/v2/positions/{symbol}")
        except TradingAPIError as exc:
            if "404" in str(exc):
                return None
            raise

    async def get_order(self, order_id: str) -> dict[str, Any]:
        """Return an order by id."""
        return await self._request("GET", f"/v2/orders/{order_id}")

    async def list_orders(
        self, status: str = "open", limit: int = 100
    ) -> list[dict[str, Any]]:
        """List orders by status."""
        payload = await self._request(
            "GET", "/v2/orders", params={"status": status, "limit": limit}
        )
        return payload if isinstance(payload, list) else []

    async def portfolio_history(
        self, period: str = "1D", timeframe: str = "1H"
    ) -> dict[str, Any]:
        """Return the portfolio equity/timestamps history."""
        return await self._request(
            "GET", "/v2/account/portfolio/history",
            params={"period": period, "timeframe": timeframe},
        )

    # ------------------------------------------------------------------
    # Write endpoints
    # ------------------------------------------------------------------
    async def submit_order(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Place an order; returns the order object.

        ``payload`` is the documented Alpaca order body (symbol, qty, side,
        type, time_in_force, plus optional limit_price / stop_price /
        order_class / stop_loss / take_profit / client_order_id).
        """
        return await self._request("POST", "/v2/orders", json=payload)

    async def cancel_order(self, order_id: str) -> dict[str, Any]:
        """Cancel an open order."""
        return await self._request("DELETE", f"/v2/orders/{order_id}")

    async def close_position(self, symbol: str) -> list[dict[str, Any]]:
        """Liquidate an open position (returns the closing order(s))."""
        payload = await self._request("DELETE", f"/v2/positions/{symbol}")
        return payload if isinstance(payload, list) else [payload]

    # ------------------------------------------------------------------
    # Low-level request with retry/backoff
    # ------------------------------------------------------------------
    async def _request(
        self,
        method: str,
        url: str,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
    ) -> Any:
        client = self._require_client()
        last_exc: Any = None

        for attempt in range(_MAX_RETRIES):
            try:
                resp = await client.request(method, url, params=params, json=json)
            except httpx.HTTPError as exc:
                last_exc = exc
                log.warning("Trading request failed (network): %s", exc)
                await asyncio.sleep(_backoff(attempt))
                continue

            if resp.status_code in (401, 403):
                raise TradingAuthError(
                    f"Paper-trading authentication failed ({resp.status_code}). "
                    "Check ALPACA_API_KEY / ALPACA_API_SECRET."
                )

            if resp.status_code in _RETRY_STATUS_CODES and attempt < _MAX_RETRIES - 1:
                last_exc = resp
                log.warning(
                    "Trading transient error %s, retrying (%d/%d)",
                    resp.status_code, attempt + 1, _MAX_RETRIES,
                )
                await asyncio.sleep(_backoff(attempt))
                continue

            if resp.status_code >= 400:
                raise TradingAPIError(
                    f"Paper-trading API error {resp.status_code}: {resp.text[:300]}"
                )

            if method in ("DELETE",) and resp.status_code == 204:
                return {}

            try:
                return resp.json()
            except (ValueError, TypeError) as exc:
                raise TradingAPIError("Paper-trading returned malformed JSON.") from exc

        raise TradingNetworkError(f"Paper-trading request failed: {last_exc!r}")


def _backoff(attempt: int) -> float:
    return 0.5 * (2 ** attempt)
