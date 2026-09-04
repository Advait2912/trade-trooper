"""Async decision and explainability logger for frontend and auditing.

Appends structured JSONL events to data/logs/decisions_YYYY-MM-DD.jsonl
using a non-blocking background queue so disk I/O never blocks the trading loop.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from schemas.pipeline import FinalReport

log = logging.getLogger("market_intel_agent.decision_logger")


class DecisionLogger:
    """Non-blocking JSONL decision logger."""

    def __init__(self, log_dir: str | Path = "data/logs") -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._worker_task: asyncio.Task | None = None
        self._stopping = False

    def _ensure_worker(self) -> None:
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._worker())

    async def _worker(self) -> None:
        while not self._stopping or not self._queue.empty():
            try:
                item = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            try:
                today_str = date.today().isoformat()
                log_file = self.log_dir / f"decisions_{today_str}.jsonl"
                line = json.dumps(item, default=str) + "\n"
                # Append in thread to avoid blocking loop
                await asyncio.to_thread(self._append_line, log_file, line)
            except Exception as exc:  # noqa: BLE001
                log.warning("Failed to write decision log: %s", exc)
            finally:
                self._queue.task_done()

    @staticmethod
    def _append_line(path: Path, line: str) -> None:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)

    async def log(
        self,
        ticker: str,
        report: FinalReport,
        benchmark: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Enqueue a structured decision record for async writing."""
        self._ensure_worker()

        now_iso = datetime.now(timezone.utc).isoformat()
        decision = report.decision
        risk = report.risk
        pred = report.prediction

        # Extract vote weights and signals
        votes: dict[str, Any] = {}
        if pred:
            votes["prediction_signal"] = {
                "signal": pred.composite_signal,
                "momentum": pred.momentum_score,
                "confidence": pred.confidence,
            }
            votes["news_sentiment"] = {
                "sentiment": pred.news_sentiment,
                "score": pred.news_sentiment_score,
            }

        bm = benchmark or report.benchmark or {}

        record: dict[str, Any] = {
            "ts": now_iso,
            "ticker": ticker.upper(),
            "industry": bm.get("industry", "Unknown"),
            "decision": decision.trade_decision,
            "composite_bias": decision.composite_bias,
            "confidence": round(decision.confidence_score, 4),
            "summary": decision.summary or decision.rationale,
            "reason": {
                "rationale": decision.rationale,
                "gates": {
                    "stop_loss": decision.stop_loss,
                    "take_profit": decision.take_profit,
                    "position_shares": decision.position_shares,
                    "option_contracts": decision.option_contracts,
                },
                "signal_votes": votes,
                "weights_db_active": bm.get("weights_db_active", False),
            },
            "benchmark": bm,
        }

        await self._queue.put(record)
        return record

    async def close(self) -> None:
        """Flush the queue and stop background worker."""
        self._stopping = True
        if self._worker_task is not None:
            try:
                await asyncio.wait_for(self._queue.join(), timeout=3.0)
            except asyncio.TimeoutError:
                pass
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None
