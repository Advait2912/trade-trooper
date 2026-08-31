"""Logging and per-stage timing instrumentation.

The agent prints concise, human-readable timing lines as it runs and writes
detailed logs (never containing secrets) for diagnostics.
"""

from __future__ import annotations

import logging
import sys
import time
from typing import Optional

_LOGGER_NAME = "market_intel_agent"

# Secrets that must never appear in logs.
_SECRET_KEYS = (
    "ALPACA_API_KEY",
    "ALPACA_API_SECRET",
    "OLLAMA_API_KEY",
    "authorization",
    "apca-api-key-id",
    "apca-api-secret-key",
)


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure the root logger for the agent and return it."""
    logger = logging.getLogger(_LOGGER_NAME)
    if logger.handlers:
        return logger

    level = logging.DEBUG if verbose else logging.INFO
    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def redact(text: str) -> str:
    """Best-effort removal of secret-looking key/value pairs from a string."""
    out = text
    for key in _SECRET_KEYS:
        # Match "Key=value", "Key: value", "key value" style occurrences.
        out = _redact_key(out, key)
    return out


def _redact_key(text: str, key: str) -> str:
    import re

    pattern = re.compile(rf"({re.escape(key)})([=: ]+)([^\s\",}}]+)", re.IGNORECASE)
    return pattern.sub(r"\1\2***", text)


class StageTimer:
    """Measures and logs the elapsed time of a single pipeline stage."""

    def __init__(
        self,
        label: str,
        logger: Optional[logging.Logger] = None,
        console: bool = True,
    ) -> None:
        self.label = label
        self.logger = logger
        self.console = console
        self._start = 0.0
        self._end: Optional[float] = None

    def __enter__(self) -> "StageTimer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._end = time.perf_counter()
        status = "complete" if exc_type is None else "failed"
        elapsed = self.elapsed
        if self.logger is not None:
            self.logger.info(
                "%s %s in %.2fs", self.label, status, elapsed
            )
        if self.console:
            mark = "" if exc_type is None else " (FAILED)"
            print(
                f"[{elapsed:5.2f}s] {self.label} {status}{mark}",
                file=sys.stdout,
                flush=True,
            )

    @property
    def elapsed(self) -> float:
        end = self._end if self._end is not None else time.perf_counter()
        return end - self._start


class PipelineClock:
    """Tracks cumulative elapsed time for the [N.NNs] console prefix."""

    def __init__(self) -> None:
        self._t0 = time.perf_counter()

    def mark(self, label: str) -> float:
        elapsed = time.perf_counter() - self._t0
        print(f"[{elapsed:5.2f}s] {label}", file=sys.stdout, flush=True)
        return elapsed
