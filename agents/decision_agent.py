"""Phase 4 - Decision Agent (placeholder).

Future implementation:
    - synthesize_signals()
    - rank_opportunities()
    Returns: trade_decision, confidence_score
"""

from __future__ import annotations

from typing import Any

from agents.base import BaseAgent
from schemas.pipeline import PhaseResult


class DecisionAgent(BaseAgent):
    name = "decision"
    description = "Placeholder: synthesizes signals and ranks opportunities (Phase 4)."
    phase = 4
    tools = ["synthesize_signals", "rank_opportunities"]

    async def run(self, phase1: Any, phase2: Any, phase3: Any) -> PhaseResult:
        return PhaseResult(
            phase="decision",
            status="not_implemented",
            summary="Decision Agent is not implemented yet.",
        )
