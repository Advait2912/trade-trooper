"""Phase 4 - Decision tools.

Deterministic trade-decision tools (no LLM):

- ``signals.synthesize_signals``: weighted directional signal synthesis
- ``ranking.rank_opportunities``: gate + score + rank trade candidates

Implements the "decision" product philosophy: a bullish composite selects a
long call / long equity, a bearish composite selects a long put (defined-risk
premium), and anything untradable becomes hold / avoid.
"""

from tools.decision_tools.ranking import (
    calculate_opportunity_score,
    rank_opportunities,
)
from tools.decision_tools.signals import synthesize_signals

__all__ = ["synthesize_signals", "calculate_opportunity_score", "rank_opportunities"]
