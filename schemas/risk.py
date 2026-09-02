"""Phase 3 – Risk Agent schemas.

``RiskResult`` is the typed output produced by the Risk Agent and consumed by
the Phase 4 Decision Agent.  All numbers are computed by deterministic tools
(plus an optional live options-chain fetch); no LLM is involved.

The nested ``PositionRecommendation`` makes the product model explicit:
equity sizing (risk/stop distance) is computed alongside option sizing
(long-only, defined-risk = premium, delta-adjusted exposure).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Greeks(BaseModel):
    """Black-Scholes greeks for a single contract (per unit sigma / per share)."""

    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    rho: float = 0.0


class EquityPosition(BaseModel):
    """Equity (shares) sizing recommendation."""

    shares: float = 0.0
    dollar_value: float = 0.0


class OptionPosition(BaseModel):
    """Option (contracts) sizing recommendation — long-only, defined risk."""

    contracts: float = 0.0
    premium_risk: float = 0.0      # total premium paid (max loss on long)
    delta_exposure: float = 0.0    # contracts * 100 * delta (share-equivalent)


class PositionRecommendation(BaseModel):
    """Both equity and option sizing views."""

    equity: EquityPosition = Field(default_factory=EquityPosition)
    option: OptionPosition = Field(default_factory=OptionPosition)


class RiskResult(BaseModel):
    """Deterministic risk assessment produced by Phase 3."""

    # ---- meta ----
    status: str = "ok"             # ok | partial | error | insufficient_data
    summary: str = ""
    errors: list[str] = Field(default_factory=list)

    # ---- composite risk ----
    risk_score: float = 0.0        # 0–100 (higher = riskier)
    risk_level: str = "moderate"   # low | moderate | high | very_high

    # ---- greeks / IV ----
    greeks_source: str = "black_scholes_estimated"  # alpaca_option_chain | black_scholes_estimated
    call_greeks: Greeks = Field(default_factory=Greeks)
    put_greeks: Greeks = Field(default_factory=Greeks)
    iv_used: float = 0.0           # annualized % used for greeks
    iv_source: str = "estimated"   # market_implied | estimated
    spread_pct: float = 0.0        # bid/ask spread as a fraction of mid
    implied_move_pct: float = 0.0  # ATM straddle cost / spot (market move)
    theta_per_day: float = 0.0     # absolute daily theta (surfaced, non-gating)

    # ---- levels ----
    stop_loss_level: float = 0.0
    take_profit_level: float = 0.0
    risk_reward_ratio: float = 0.0

    # ---- position sizing ----
    position_recommendation: PositionRecommendation = Field(
        default_factory=PositionRecommendation
    )
    capital_at_risk_pct: float = 0.0

    # ---- max loss ----
    max_loss_dollars: float = 0.0
    max_loss_pct: float = 0.0
    tail_var_dollars: float = 0.0
    tail_cvar_dollars: float = 0.0

    # ---- auditable nested tool outputs (keyed by tool name) ----
    risk_metrics: dict = Field(default_factory=dict)
