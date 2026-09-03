"""Deterministic tool suite.

Each agent's deterministic tools live in a matching subtree:

- ``news_tools``          Phase 1 - News Collection Agent
- ``market_data_tools``   Phase 1 - Market Data Agent
- ``historical``          Phase 1 - Historical Data Agent
- ``prediction_tools``    Phase 2 - Prediction Agent
- ``risk_tools``          Phase 3 - Risk Agent
- ``decision_tools``      Phase 4 - Decision Agent

``registry`` holds the name -> spec (schema + handler) mapping used by the
agents and documentation.
"""

from tools.registry import TOOL_SPECS, get_tool, import_handlers

__all__ = ["TOOL_SPECS", "get_tool", "import_handlers"]
