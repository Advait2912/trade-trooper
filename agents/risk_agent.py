"""Phase 3 - Risk Agent (placeholder).

Future implementation:
    - calculate_greeks()
    - calculate_position_size()
    - calculate_max_loss()
    Returns: risk_metrics, position_recommendation
"""

from __future__ import annotations

from typing import Any

from agents.base import BaseAgent
from schemas.pipeline import PhaseResult


class RiskAgent(BaseAgent):
    name = "risk"
    description = "Placeholder: computes risk metrics and position size (Phase 3)."
    phase = 3
    tools = ["calculate_greeks", "calculate_position_size", "calculate_max_loss"]

    async def run(self, phase1: Any, phase2: Any) -> PhaseResult:
        return PhaseResult(
            phase="risk",
            status="not_implemented",
            summary="Risk Agent is not implemented yet.",
        )
