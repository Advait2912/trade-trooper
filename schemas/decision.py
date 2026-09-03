"""Phase 4 – Decision Agent schemas.

``DecisionResult`` is the typed output produced by the Decision Agent and the
final deterministic product of the 4-phase pipeline.  All numbers are computed
by deterministic tools (plus the Phase 2/3 results); no LLM is involved.

The Decision Agent is direction-aware and long-only on premium:
- bullish  -> long call (or long equity)
- bearish  -> long put (defined risk = premium)
- neutral / un-tradable -> hold / avoid
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class DecisionGate(BaseModel):
    """Result of a single deterministic gate for a candidate."""

    name: str
    passed: bool = False
    value: float = 0.0
    detail: str = ""


class Opportunity(BaseModel):
    """A single ranked trade opportunity (long-only/premium)."""

    symbol: str = ""
    direction: str = "long"      # long
    instrument: str = "option"   # equity | option
    option_type: str = "call"    # call | put (empty for equity)
    score: float = 0.0           # 0-100
    rank: int = 1
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    risk_reward_ratio: float = 0.0
    position_shares: float = 0.0
    option_contracts: float = 0.0
    premium_risk: float = 0.0
    rationale: str = ""


class DecisionResult(BaseModel):
    """Deterministic trade decision produced by Phase 4."""

    # ---- meta ----
    status: str = "ok"            # ok | partial | insufficient_data | error
    summary: str = ""
    errors: list[str] = Field(default_factory=list)

    # ---- headline decision ----
    trade_decision: str = "hold"  # long_call | long_put | long_equity | hold | avoid
    confidence_score: float = 0.0  # 0-1
    composite_bias: str = "neutral"  # bullish | bearish | neutral
    agreement_score: float = 0.0     # 0-1 (how aligned the sources are)
    divergences: list[str] = Field(default_factory=list)

    # ---- chosen position ----
    symbol: str = ""
    instrument: str = "none"      # equity | option | none
    option_type: str = ""         # call | put (empty for equity/none)
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    risk_reward_ratio: float = 0.0
    position_shares: float = 0.0
    option_contracts: float = 0.0
    premium_risk: float = 0.0
    rationale: str = ""

    # ---- ranked opportunities (future multi-symbol support) ----
    opportunities: list[Opportunity] = Field(default_factory=list)

    # ---- auditable nested tool outputs (keyed by tool name) ----
    decision_metrics: dict = Field(default_factory=dict)


# Re-export for convenience.
Gates = dict[str, Any]
