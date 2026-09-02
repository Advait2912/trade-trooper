"""Alpaca options data layer — option chain snapshots (greeks + IV + quote).

Phase 3 Risk Agent fetches the option chain via the REST endpoint
``GET /v1beta1/options/snapshots/{underlying_symbol}``, which returns — for each
contract — Black-Scholes greeks, market-implied volatility and the latest
bid/ask.  This is the same data the WebSocket option stream carries, but a
single REST request fits the ~2s phase budget and reuses the existing
``AlpacaClient``.

The free ``indicative`` feed returns delayed/modified quotes; ``opra`` requires
a paid subscription.  If the account has no options subscription the endpoint
returns 401/403, which the Risk Agent degrades on.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any

from alpaca.client import AlpacaClient

log = logging.getLogger("market_intel_agent.options")


async def get_option_chain(
    client: AlpacaClient,
    underlying_symbol: str,
    spot: float,
    *,
    feed: str = "indicative",
    strike_band: float = 0.10,
    horizon_days: int = 5,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Fetch a slice of the option chain around the ATM strike.

    Parameters
    ----------
    client:
        Active ``AlpacaClient`` (inside its async context manager).
    underlying_symbol:
        Equity ticker (e.g. ``NVDA``).
    spot:
        Current price — used to bracket strikes around the money.
    feed:
        Options feed: ``opra`` (paid) or ``indicative`` (free).
    strike_band:
        Fractional band around ``spot`` for strike filtering (e.g. 0.10 = ±10 %).
    horizon_days:
        Minimum days-to-expiry filter.
    limit:
        Maximum snapshots to return.

    Returns
    -------
    List of snapshot dicts, each with a ``symbol`` key plus the snapshot fields
    (``greeks``, ``impliedVolatility``, ``latestQuote``, ``latestTrade``).
    Empty on any failure (caller degrades gracefully).
    """
    if spot <= 0:
        return []

    expiry_gte = (date.today() + timedelta(days=max(1, horizon_days))).isoformat()
    params: dict[str, Any] = {
        "feed": feed,
        "strike_price_gte": round(spot * (1 - strike_band), 2),
        "strike_price_lte": round(spot * (1 + strike_band), 2),
        "expiration_date_gte": expiry_gte,
        "limit": limit,
    }

    payload = await client.get_json(
        f"/v1beta1/options/snapshots/{underlying_symbol}", params=params
    )

    snapshots = payload.get("snapshots", {}) if isinstance(payload, dict) else {}
    out: list[dict[str, Any]] = []
    for symbol, snap in snapshots.items():
        if isinstance(snap, dict):
            out.append({"symbol": symbol, **snap})
    return out
