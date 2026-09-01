"""Base agent class.

Agents are thin orchestrators over deterministic tools. Each agent owns its
phase, exposes its tool list, and degrades gracefully: tool failures yield
partial results, never crashes.
"""

from __future__ import annotations

from typing import Any


class BaseAgent:
    """Base class for every agent in the suite."""

    name: str = "base"
    description: str = ""
    phase: int = 0
    tools: list[str] = []

    async def run(self, **kwargs: Any) -> Any:
        """Execute the agent's work for a ticker. Overridden by subclasses."""
        raise NotImplementedError(f"{type(self).__name__}.run() is not implemented")
