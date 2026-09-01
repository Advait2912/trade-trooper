"""Phase timing budgets (the ~10s per trading cycle).

Total target across all four phases is ~10 seconds; each agent timing budget
is enforced by the orchestrator's StageTimer instrumentation.
"""

from __future__ import annotations

PHASES = {
    1: {
        "name": "Parallel data collection",
        "agents": ["news_collection", "market_data", "historical"],
        "budget_s": 4,
    },
    2: {"name": "Sequential prediction", "agents": ["prediction"], "budget_s": 3},
    3: {"name": "Sequential risk", "agents": ["risk"], "budget_s": 2},
    4: {"name": "Sequential decision", "agents": ["decision"], "budget_s": 1},
}

TOTAL_BUDGET_S = sum(p["budget_s"] for p in PHASES.values())
