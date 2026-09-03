"""Phase 5 - Paper-trading runner.

The persistent loop that turns Phase 1-4 decisions into real (paper) trades:

    while True and market-hours:
        manage open positions (stop/target/flip/horizon)
        run the pipeline -> decision
        journal the cycle + equity
        if room and daily loss not exceeded and decision is a trade:
            place the order (executor) -> journal the order

The loop is kill-switched (``TRADING_ENABLED``), paper-only (via
``TradingClient``), single-position by default, and idempotent per cycle
(``client_order_id``).  It never touches real money.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime, time, timezone
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from alpaca.trading import TradingClient
from orchestrator.pipeline import Pipeline
from trading.executor import execute_decision
from trading.journal import TradeJournal
from trading.position_manager import decide_exit, manage_and_close
from utils.config import Settings
from utils.paths import data_path

log = logging.getLogger("market_intel_agent.runner")

try:
    _NY: ZoneInfo | None = ZoneInfo("America/New_York")
except ZoneInfoNotFoundError:  # pragma: no cover - bare Windows
    _NY = None
_MARKET_OPEN = time(9, 30)
_MARKET_CLOSE = time(16, 0)


def is_market_open(now: datetime | None = None) -> bool:
    """True if the US regular session is open (Mon-Fri 09:30-16:00 ET)."""
    now = now or datetime.now(timezone.utc)
    if _NY is not None:
        local = now.astimezone(_NY)
    else:  # pragma: no cover - approx UTC window
        local = now
    if local.weekday() >= 5:
        return False
    t = local.time()
    return _MARKET_OPEN <= t <= _MARKET_CLOSE


class PaperRunner:
    """Drives one ticker's paper-trading loop."""

    def __init__(
        self,
        settings: Settings,
        ticker: str,
        journal_path: str | None = None,
        verbose: bool = False,
    ) -> None:
        self.settings = settings
        self.ticker = ticker
        self.verbose = verbose
        self.journal = TradeJournal(journal_path or str(data_path("trading_journal.db")))
        # Open-position state (symbol -> opened_ts) is kept in memory and
        # rebuilt from the journal on start.
        self._open_state: dict[str, str] = {}
        self._day_open_equity: float | None = None
        self._day_key: str = ""
        self._day_locked = False

    async def start(self) -> None:
        """Begin the loop; runs until interrupted."""
        self._rebuild_open_state()
        interval = max(1, self.settings.trading_interval_min) * 60
        log.info("Paper runner started for %s (interval %ds)", self.ticker, interval)
        while True:
            await asyncio.sleep(interval)
            if not is_market_open():
                log.debug("market closed - skipping cycle")
                continue
            try:
                await self.run_once()
            except Exception as exc:  # noqa: BLE001 - keep the loop alive
                log.exception("cycle failed: %s", exc)
                await asyncio.sleep(interval)

    def _rebuild_open_state(self) -> None:
        """Rebuild opened timestamps for currently-held symbols from the journal."""
        orders = self.journal.orders()
        for o in reversed(orders):  # newest first
            symbol = o["symbol"]
            if symbol not in self._open_state and o["side"] == "buy":
                self._open_state[symbol] = o.get("cycle_ts", "")

    async def run_once(self) -> dict[str, Any]:
        """Execute one cycle (decide + manage positions + maybe trade)."""
        cycle_ts = datetime.now(timezone.utc).isoformat()
        result: dict[str, Any] = {"cycle_ts": cycle_ts, "ticker": self.ticker}

        async with TradingClient(self.settings) as client:
            account = await client.get_account()
            equity = float(account.get("equity") or 0.0)
            cash = float(account.get("cash") or 0.0)
            self.journal.record_equity(cycle_ts, equity, cash)

            # ---- daily loss limit ----
            self._update_day_open(equity)
            if self._day_locked:
                log.warning("Daily loss limit hit - blocking new entries")
                result["entry_blocked"] = "daily_loss_limit"

            # ---- decide first (so the decision can flip-close positions) ----
            report = await Pipeline(self.settings, verbose=self.verbose).run(self.ticker)
            decision = report.decision
            self.journal.record_cycle(
                cycle_ts, self.ticker, decision.trade_decision,
                decision.composite_bias, decision.confidence_score,
                report.model_dump(),
            )
            result["decision"] = decision.trade_decision
            result["confidence"] = decision.confidence_score

            # ---- manage open positions using the current decision ----
            positions = await client.get_positions()
            await self._manage_positions(client, positions, cycle_ts, decision)

            # ---- re-read positions (may have changed from closes) ----
            positions = await client.get_positions()
            open_count = len(positions)
            result["open_positions"] = open_count

            # ---- maybe enter ----
            if self._day_locked:
                result["entry_blocked"] = "daily_loss_limit"
            elif open_count >= self.settings.max_open_positions:
                result["entry_blocked"] = "max_positions"
            else:
                options = (report.risk.risk_metrics or {}).get("calculate_greeks", {}) or {}
                exec_result = await execute_decision(
                    client, decision, self.settings, options, cycle_ts
                )
                if exec_result["action"] == "order":
                    symbol = exec_result.get("symbol") or self.ticker
                    self._open_state.setdefault(symbol, cycle_ts)
                    self.journal.record_order(
                        cycle_ts=cycle_ts,
                        client_order_id=exec_result.get("client_order_id", ""),
                        symbol=symbol,
                        side=exec_result.get("side", "buy"),
                        qty=float(exec_result.get("qty") or 0.0),
                        order_type=exec_result.get("type", ""),
                        limit_price=None,
                        status=exec_result.get("status", "submitted"),
                        filled_avg_price=None,
                        order_id=exec_result.get("order_id"),
                        reason="entry",
                    )
                    result["entry"] = exec_result
                elif exec_result["action"] == "error":
                    result["entry_error"] = exec_result["reason"]
                    log.warning("entry failed: %s", exec_result["reason"])

        result["equity"] = equity
        result["cash"] = cash
        return result

    async def _manage_positions(
        self,
        client: TradingClient,
        positions: list[dict[str, Any]],
        cycle_ts: str,
        decision: Any,
    ) -> None:
        for pos in positions:
            symbol = pos.get("symbol") or self.ticker
            pos.setdefault("opened_at", self._open_state.get(symbol, cycle_ts))
            out = decide_exit(
                pos, decision, horizon_days=self.settings.trade_horizon_days
            )
            if out["action"] != "hold":
                await manage_and_close(
                    client, pos, self.journal, out["action"], self.ticker, cycle_ts
                )
                self._open_state.pop(symbol, None)

    def _update_day_open(self, equity: float) -> None:
        day = date.today().isoformat()
        if self._day_open_equity is None or self._day_key != day:
            self._day_open_equity = equity
            self._day_key = day
            self._day_locked = False
        if self._day_open_equity and self._day_open_equity > 0:
            loss_pct = (equity - self._day_open_equity) / self._day_open_equity
            if loss_pct <= -self.settings.daily_loss_limit_pct:
                self._day_locked = True
