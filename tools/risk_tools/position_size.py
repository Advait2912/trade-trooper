"""Phase 3 - Position sizing tool.

Fixed-fractional sizing: risk a fixed fraction of capital per trade, where the
stop distance converts that dollar risk into a share count.  The base size is
then scaled by four quality factors:

    confidence  : prediction confidence            (lower confidence -> smaller)
    iv_quality  : trust in the IV/greeks source    (estimated -> smaller)
    spread_pct  : option bid/ask spread            (wider -> smaller)
    drawdown    : current drawdown regime          (deeper -> smaller)

The result is capped at ``max_position_pct`` of capital.  Both equity (shares)
and option (long-only, defined-risk = premium) views are produced so the
product model is explicit.  All math is deterministic.
"""

from __future__ import annotations

from typing import Any

_DRAWDOWN_FACTOR: dict[str, float] = {
    "low": 1.0,
    "moderate": 0.8,
    "high": 0.6,
    "extreme": 0.4,
}


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def calculate_position_size(
    capital: float,
    risk_per_trade_pct: float,
    entry: float,
    stop: float,
    premium: float = 0.0,
    delta: float = 1.0,
    confidence: float = 0.5,
    iv_quality: float = 1.0,
    spread_pct: float = 0.0,
    drawdown_risk: str = "low",
    max_position_pct: float = 0.05,
) -> dict[str, Any]:
    """Compute equity and option position sizes.

    Returns
    -------
    dict with equity_shares, equity_dollar_value, option_contracts,
    option_premium_risk, delta_exposure, capital_at_risk[_pct] and the
    quality factors for auditability.
    """
    errors: list[str] = []

    if capital <= 0:
        return {
            "equity_shares": 0.0,
            "equity_dollar_value": 0.0,
            "option_contracts": 0.0,
            "option_premium_risk": 0.0,
            "delta_exposure": 0.0,
            "capital_at_risk": 0.0,
            "capital_at_risk_pct": 0.0,
            "risk_per_trade_pct": risk_per_trade_pct,
            "confidence_factor": 0.0,
            "iv_quality_factor": iv_quality,
            "spread_factor": 1.0,
            "drawdown_factor": 1.0,
            "sizing_method": "fixed_fractional",
            "errors": ["Invalid or zero capital — cannot size position."],
        }

    stop_distance = entry - stop
    if entry <= 0 or stop_distance <= 0:
        errors.append("Invalid entry/stop: stop must be below entry for a long position.")
        stop_distance = 0.0

    risk_amount = capital * risk_per_trade_pct

    confidence_factor = _clamp(confidence, 0.25, 1.0)
    iv_quality_factor = _clamp(iv_quality, 0.5, 1.0)
    spread_factor = _clamp(1.0 - spread_pct * 6.0, 0.3, 1.0)
    drawdown_factor = _DRAWDOWN_FACTOR.get(drawdown_risk, 1.0)
    total_factor = confidence_factor * iv_quality_factor * spread_factor * drawdown_factor

    if stop_distance > 0:
        base_shares = risk_amount / stop_distance
    else:
        base_shares = 0.0
    shares = base_shares * total_factor
    dollar_value = shares * entry

    max_dollar = capital * max_position_pct
    if dollar_value > max_dollar:
        shares = max_dollar / entry
        dollar_value = max_dollar

    capital_at_risk = shares * stop_distance
    capital_at_risk_pct = capital_at_risk / capital

    # Option view (long-only, defined risk = premium paid).
    contracts = 0.0
    premium_risk = 0.0
    delta_exposure = 0.0
    if premium > 0:
        per_contract_risk = premium * 100.0
        contracts = (risk_amount / per_contract_risk) * total_factor if per_contract_risk > 0 else 0.0
        premium_risk = contracts * per_contract_risk
        if premium_risk > max_dollar:
            contracts = max_dollar / per_contract_risk
            premium_risk = max_dollar
        delta_exposure = contracts * 100.0 * delta

    return {
        "equity_shares": round(shares, 4),
        "equity_dollar_value": round(dollar_value, 2),
        "option_contracts": round(contracts, 2),
        "option_premium_risk": round(premium_risk, 2),
        "delta_exposure": round(delta_exposure, 2),
        "capital_at_risk": round(capital_at_risk, 2),
        "capital_at_risk_pct": round(capital_at_risk_pct, 6),
        "risk_per_trade_pct": risk_per_trade_pct,
        "confidence_factor": round(confidence_factor, 4),
        "iv_quality_factor": round(iv_quality_factor, 4),
        "spread_factor": round(spread_factor, 4),
        "drawdown_factor": round(drawdown_factor, 4),
        "sizing_method": "fixed_fractional",
        "errors": errors,
    }
