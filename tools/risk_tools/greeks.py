"""Phase 3 - Options greeks tool.

Produces Black-Scholes greeks for the at-the-money call and put nearest the
forecast horizon.  Two sources, in priority order:

1. ``alpaca_option_chain`` — greeks + implied volatility + bid/ask returned
   directly by Alpaca's option chain endpoint (``/v1beta1/options/snapshots``).
   Alpaca computes these using the Black-Scholes model and also returns the
   market-implied volatility, so this is the preferred, market-anchored source.
2. ``black_scholes_estimated`` — local Black-Scholes greeks derived from
   Phase 2's *estimated* forward volatility (``iv_forecast``) when no options
   chain is available.  No scipy is required; the normal CDF uses ``math.erf``.

The tool is a pure, deterministic function: it receives the (already-fetched)
chain data and always returns a normalized dict, degrading gracefully to the
local estimate when the chain is empty or unparseable.
"""

from __future__ import annotations

import math
from typing import Any

_RFR = 0.045          # default risk-free rate
_BS_KEYS: tuple[str, ...] = ("delta", "gamma", "theta", "vega", "rho")


def parse_occ_symbol(symbol: str) -> dict[str, Any] | None:
    """Parse a standard equity option OCC symbol.

    Format: ``ROOT`` + ``YYMMDD`` + ``C|P`` + 8-digit strike (strike × 1000).
    Parsing is right-to-left, so it is robust to variable-length roots.

    Returns ``{root, expiry, type, strike}`` or ``None`` when unparseable.
    """
    s = str(symbol).upper().strip()
    if len(s) < 16:
        return None
    strike_str = s[-8:]
    opt_type = s[-9]
    date_str = s[-15:-9]
    root = s[:-15]
    if opt_type not in ("C", "P"):
        return None
    if not date_str.isdigit() or not strike_str.isdigit():
        return None
    return {
        "root": root,
        "expiry": date_str,  # YYMMDD
        "type": "call" if opt_type == "C" else "put",
        "strike": int(strike_str) / 1000.0,
    }


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _normal_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _bs_price(
    spot: float, strike: float, t: float, r: float, sigma: float, is_call: bool
) -> float:
    d1 = (math.log(spot / strike) + (r + 0.5 * sigma * sigma) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)
    if is_call:
        return spot * _normal_cdf(d1) - strike * math.exp(-r * t) * _normal_cdf(d2)
    return strike * math.exp(-r * t) * _normal_cdf(-d2) - spot * _normal_cdf(-d1)


def _bs_greeks(
    spot: float, strike: float, t: float, r: float, sigma: float, is_call: bool
) -> dict[str, float]:
    """Theoretical greeks (theta annualized; vega per 1.0 change in sigma)."""
    zero = dict.fromkeys(_BS_KEYS, 0.0)
    if spot <= 0 or strike <= 0 or t <= 0 or sigma <= 0:
        return zero
    sqrt_t = math.sqrt(t)
    d1 = (math.log(spot / strike) + (r + 0.5 * sigma * sigma) * t) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    pdf_d1 = _normal_pdf(d1)

    gamma = pdf_d1 / (spot * sigma * sqrt_t)
    vega = spot * pdf_d1 * sqrt_t
    if is_call:
        delta = _normal_cdf(d1)
        theta = -(spot * pdf_d1 * sigma) / (2 * sqrt_t) - r * strike * math.exp(-r * t) * _normal_cdf(d2)
        rho = strike * t * math.exp(-r * t) * _normal_cdf(d2)
    else:
        delta = _normal_cdf(d1) - 1.0
        theta = -(spot * pdf_d1 * sigma) / (2 * sqrt_t) + r * strike * math.exp(-r * t) * _normal_cdf(-d2)
        rho = -strike * t * math.exp(-r * t) * _normal_cdf(-d2)

    return {"delta": delta, "gamma": gamma, "theta": theta, "vega": vega, "rho": rho}


def _mid(quote: dict[str, Any] | None) -> tuple[float, float, float]:
    """Return (bid, ask, mid) from an Alpaca option quote dict."""
    if not quote:
        return 0.0, 0.0, 0.0
    bid = float(quote.get("bp") or 0.0)
    ask = float(quote.get("ap") or 0.0)
    mid = (bid + ask) / 2.0 if bid > 0 and ask > 0 else (ask or bid)
    return bid, ask, mid


def calculate_greeks(
    chain: list[dict[str, Any]] | None,
    spot: float,
    iv_forecast: float,
    horizon_days: int = 5,
    risk_free_rate: float = _RFR,
) -> dict[str, Any]:
    """Select ATM call & put and produce greeks.

    Parameters
    ----------
    chain:
        List of raw option-chain snapshots (each with a ``symbol`` and optional
        ``greeks`` / ``impliedVolatility`` / ``latestQuote`` keys).  ``None`` or
        ``[]`` triggers the local Black-Scholes fallback.
    spot:
        Current underlying price (for ATM strike selection).
    iv_forecast:
        Phase 2 estimated forward volatility (annualized %), used only in the
        fallback.
    horizon_days:
        Forecast horizon in trading days (used for fallback time-to-expiry).
    risk_free_rate:
        Risk-free rate for the fallback Black-Scholes model.

    Returns
    -------
    dict with greeks, source, IV, spread, implied move, theta and premiums.
    """
    errors: list[str] = []

    if chain:
        calls: list[tuple[dict[str, Any], dict[str, Any]]] = []
        puts: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for snap in chain:
            if not isinstance(snap, dict):
                continue
            parsed = parse_occ_symbol(str(snap.get("symbol", "")))
            if parsed is None:
                continue
            if parsed["type"] == "call":
                calls.append((parsed, snap))
            else:
                puts.append((parsed, snap))

        if calls and puts and spot > 0:
            expiries = {p["expiry"] for p, _ in calls} | {p["expiry"] for p, _ in puts}
            expiry = min(expiries)
            calls_at_exp = [(p, s) for p, s in calls if p["expiry"] == expiry]
            puts_at_exp = [(p, s) for p, s in puts if p["expiry"] == expiry]
            _, call_snap = min(calls_at_exp, key=lambda x: abs(x[0]["strike"] - spot))
            _, put_snap = min(puts_at_exp, key=lambda x: abs(x[0]["strike"] - spot))

            call_greeks = {k: float(call_snap.get("greeks", {}).get(k) or 0.0) for k in _BS_KEYS}
            put_greeks = {k: float(put_snap.get("greeks", {}).get(k) or 0.0) for k in _BS_KEYS}
            iv = float(call_snap.get("impliedVolatility") or 0.0)

            call_bid, call_ask, call_mid = _mid(call_snap.get("latestQuote"))
            put_bid, put_ask, put_mid = _mid(put_snap.get("latestQuote"))

            spread = (call_ask - call_bid) / call_mid if call_mid > 0 else 0.0
            straddle = (call_ask or call_mid) + (put_ask or put_mid)
            implied_move = straddle / spot if spot > 0 else 0.0
            theta_per_day = abs(call_greeks["theta"])

            return {
                "greeks": {"call": call_greeks, "put": put_greeks},
                "greeks_source": "alpaca_option_chain",
                "iv_used": round(iv * 100.0, 4),
                "iv_source": "market_implied",
                "strike": round(min(calls_at_exp, key=lambda x: abs(x[0]["strike"] - spot))[0]["strike"], 4),
                "expiry": expiry,
                "spread_pct": round(spread, 6),
                "implied_move_pct": round(implied_move, 6),
                "theta_per_day": round(theta_per_day, 6),
                "call_premium": round(call_mid, 4),
                "put_premium": round(put_mid, 4),
                "call_bid": round(call_bid, 4),
                "call_ask": round(call_ask, 4),
                "put_bid": round(put_bid, 4),
                "put_ask": round(put_ask, 4),
                "errors": errors,
            }
        errors.append("Options chain present but no usable ATM call/put; using Black-Scholes fallback.")

    # ---- Local Black-Scholes fallback from estimated IV ----
    if spot <= 0:
        return {
            "greeks": {"call": dict.fromkeys(_BS_KEYS, 0.0), "put": dict.fromkeys(_BS_KEYS, 0.0)},
            "greeks_source": "black_scholes_estimated",
            "iv_used": 0.0,
            "iv_source": "estimated",
            "strike": 0.0,
            "expiry": "",
            "spread_pct": 0.0,
            "implied_move_pct": 0.0,
            "theta_per_day": 0.0,
            "call_premium": 0.0,
            "put_premium": 0.0,
            "call_bid": 0.0,
            "call_ask": 0.0,
            "put_bid": 0.0,
            "put_ask": 0.0,
            "errors": ["Invalid or zero spot price — cannot compute greeks."],
        }

    sigma = iv_forecast / 100.0
    if sigma <= 0:
        sigma = 0.25  # 25 % floor fallback
        errors.append("IV forecast unavailable; used 25 % fallback.")
    strike = round(spot)
    t = max(horizon_days, 1) / 365.0

    call_g = _bs_greeks(spot, strike, t, risk_free_rate, sigma, True)
    put_g = _bs_greeks(spot, strike, t, risk_free_rate, sigma, False)
    call_premium = _bs_price(spot, strike, t, risk_free_rate, sigma, True)
    put_premium = _bs_price(spot, strike, t, risk_free_rate, sigma, False)

    call_g["theta"] = call_g["theta"] / 365.0
    put_g["theta"] = put_g["theta"] / 365.0
    theta_per_day = abs(call_g["theta"])

    implied_move = sigma * math.sqrt(t)

    return {
        "greeks": {
            "call": {k: round(v, 6) for k, v in call_g.items()},
            "put": {k: round(v, 6) for k, v in put_g.items()},
        },
        "greeks_source": "black_scholes_estimated",
        "iv_used": round(sigma * 100.0, 4),
        "iv_source": "estimated",
        "strike": strike,
        "expiry": "",
        "spread_pct": 0.0,
        "implied_move_pct": round(implied_move, 6),
        "theta_per_day": round(theta_per_day, 6),
        "call_premium": round(call_premium, 4),
        "put_premium": round(put_premium, 4),
        "call_bid": 0.0,
        "call_ask": 0.0,
        "put_bid": 0.0,
        "put_ask": 0.0,
        "errors": errors,
    }
