"""Phase 5 - Paper-trading runner.

Supports:
- ``PortfolioRunner``: Unified multi-ticker runner managing an entire universe
  of stocks in one lean process. Evaluates all tickers in parallel, resolves
  per-ticker sector weights from ``weights_db.json``, executes trades on Alpaca,
  and logs structured JSONL decisions and Optuna retraining checkpoints.
- ``PaperRunner``: Single-ticker runner preserved for backwards compatibility.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time as _time
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from alpaca.trading import TradingClient
from orchestrator.pipeline import Pipeline
from trading.executor import execute_decision
from trading.journal import TradeJournal
from trading.position_manager import decide_exit, manage_and_close
from utils.config import Settings
from utils.decision_logger import DecisionLogger
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


def _get_sym(pos: Any) -> str:
    if isinstance(pos, dict):
        return str(pos.get("symbol", ""))
    return str(getattr(pos, "symbol", ""))


class PortfolioRunner:
    """Unified multi-ticker portfolio paper-trading runner."""

    def __init__(
        self,
        settings: Settings,
        tickers: list[str],
        journal_dir: str | Path | None = None,
        verbose: bool = False,
    ) -> None:
        self.settings = settings
        self.tickers = [t.strip().upper() for t in tickers if t.strip()]
        self.verbose = verbose
        j_dir = Path(journal_dir) if journal_dir else data_path("journals")
        j_dir.mkdir(parents=True, exist_ok=True)
        self.journals = {
            t: TradeJournal(j_dir / f"journal_{t.lower()}.db") for t in self.tickers
        }
        self.decision_logger = DecisionLogger(data_path("logs"))
        self._open_state: dict[str, dict[str, str]] = {t: {} for t in self.tickers}
        self._day_open_equity: dict[str, float] = {}
        self._day_key: str = ""
        self._day_locked: dict[str, bool] = {t: False for t in self.tickers}
        self._portfolio_peak_equity: float = 0.0
        self._portfolio_dd_locked: bool = False

    def _rebuild_open_state(self) -> None:
        for t in self.tickers:
            orders = self.journals[t].orders()
            for o in reversed(orders):
                sym = o["symbol"]
                if sym not in self._open_state[t] and o["side"] == "buy":
                    self._open_state[t][sym] = o.get("cycle_ts", "")

    async def start(self) -> None:
        self._rebuild_open_state()
        interval = max(1, self.settings.trading_interval_min) * 60
        log.info("PortfolioRunner started for %s (interval %ds)", self.tickers, interval)

        # Alpaca CLI Preflight Check
        try:
            from tools import alpaca_cli
            if alpaca_cli.is_available():
                alpaca_cli.ensure_profile_login(
                    self.settings.alpaca_api_key,
                    self.settings.alpaca_api_secret,
                    paper=True,
                )
                doc = alpaca_cli.cli_doctor()
                if doc.get("ok"):
                    log.info("Alpaca CLI doctor: PASSED (connected to paper endpoint)")
                else:
                    log.warning("Alpaca CLI doctor check: %s", doc.get("error") or doc.get("stdout"))
        except Exception as exc:  # noqa: BLE001
            log.warning("Alpaca CLI preflight check bypassed: %s", exc)

        while True:
            if not is_market_open():
                log.debug("market closed - skipping cycle")
            else:
                try:
                    await self.run_once()
                except Exception as exc:  # noqa: BLE001
                    log.exception("portfolio cycle failed: %s", exc)
            await asyncio.sleep(interval)

    async def run_once(self) -> dict[str, Any]:
        cycle_ts = datetime.now(timezone.utc).isoformat()
        t_start = _time.perf_counter()
        results: dict[str, Any] = {"cycle_ts": cycle_ts, "tickers": {}}

        async with TradingClient(self.settings) as client:
            account = await client.get_account()
            equity = float(account.get("equity") or 0.0)
            cash = float(account.get("cash") or 0.0)

            for t in self.tickers:
                self.journals[t].record_equity(cycle_ts, equity, cash)
                self._update_day_open(t, equity)

            # 1. Run Pipeline for all tickers in parallel
            t_pipe = _time.perf_counter()
            pipeline = Pipeline(self.settings, verbose=self.verbose)
            reports = await asyncio.gather(
                *(pipeline.run(t) for t in self.tickers),
                return_exceptions=True,
            )
            pipe_ms = (_time.perf_counter() - t_pipe) * 1000

            # 2. Fetch current positions once
            positions = await client.get_positions()

            # 3. Manage open positions per ticker with fresh decisions
            for t, rep in zip(self.tickers, reports):
                if isinstance(rep, Exception):
                    log.warning("[%s] Pipeline failed: %s", t, rep)
                    continue
                t_positions = [p for p in positions if _get_sym(p).startswith(t)]
                await self._manage_positions(client, t, t_positions, cycle_ts, rep.decision)

            # 4. Re-read positions after closes
            positions = await client.get_positions()

            # 5. Execute new entries (independent per ticker, faithful to backtest)
            t_order_start = _time.perf_counter()
            for t, rep in zip(self.tickers, reports):
                if isinstance(rep, Exception):
                    continue
                d = rep.decision
                t_result: dict[str, Any] = {
                    "decision": d.trade_decision,
                    "confidence": d.confidence_score,
                }

                # Check ticker daily loss limit
                if self._day_locked.get(t, False):
                    t_result["entry_blocked"] = "daily_loss_limit"
                    results["tickers"][t] = t_result
                    continue

                # Portfolio drawdown kill switch (halts all new entries)
                if self._portfolio_dd_locked:
                    t_result["entry_blocked"] = "portfolio_drawdown_limit"
                    results["tickers"][t] = t_result
                    continue

                # Count open positions: per-ticker AND portfolio-wide
                t_positions = [p for p in positions if _get_sym(p).startswith(t)]
                if len(t_positions) >= self.settings.max_open_positions_per_ticker:
                    t_result["entry_blocked"] = "max_positions_per_ticker"
                    results["tickers"][t] = t_result
                    continue
                if len(positions) >= self.settings.max_open_positions:
                    t_result["entry_blocked"] = "max_positions"
                    results["tickers"][t] = t_result
                    continue

                # If entry decision, place order
                if d.trade_decision in ("long_equity", "long_call", "long_put"):
                    opts = (rep.risk.risk_metrics or {}).get("calculate_greeks", {}) or {}
                    exec_result = await execute_decision(
                        client, d, self.settings, opts, cycle_ts
                    )
                    if exec_result["action"] == "order":
                        sym = exec_result.get("symbol") or t
                        self._open_state[t].setdefault(sym, cycle_ts)
                        self.journals[t].record_order(
                            cycle_ts=cycle_ts,
                            client_order_id=exec_result.get("client_order_id", ""),
                            symbol=sym,
                            side=exec_result.get("side", "buy"),
                            qty=float(exec_result.get("qty") or 0.0),
                            order_type=exec_result.get("type", ""),
                            limit_price=None,
                            status=exec_result.get("status", "submitted"),
                            filled_avg_price=None,
                            order_id=exec_result.get("order_id"),
                            reason="entry",
                        )
                        positions.append({"symbol": sym})  # reflect in local count
                        t_result["entry"] = exec_result

                results["tickers"][t] = t_result

            order_ms = (_time.perf_counter() - t_order_start) * 1000
            total_ms = (_time.perf_counter() - t_start) * 1000

            # 6. Journal, Decision Log, and Checkpoints for all tickers
            for t, rep in zip(self.tickers, reports):
                if isinstance(rep, Exception):
                    continue
                d = rep.decision
                bm = {
                    **(rep.benchmark or {}),
                    "order_ms": round(order_ms, 1),
                    "total_ms": round(total_ms, 1),
                }

                # SQLite journal cycle record
                self.journals[t].record_cycle(
                    cycle_ts,
                    t,
                    d.trade_decision,
                    d.composite_bias,
                    d.confidence_score,
                    rep.model_dump(),
                    benchmark=bm,
                )

                # Optuna retraining checkpoint
                w_hash = hashlib.sha256(json.dumps(bm, default=str).encode()).hexdigest()[:12]
                self.journals[t].record_cycle_checkpoint(
                    ticker=t,
                    cycle_ts=cycle_ts,
                    industry=bm.get("industry", ""),
                    weights_hash=w_hash,
                    market_snapshot={"price": getattr(rep.market_context, "price", 0.0)},
                    news_snapshot={"headline": getattr(rep.news, "headline", "")},
                    decision=d.trade_decision,
                    trade_placed="entry" in results["tickers"].get(t, {}),
                    entry_price=d.price,
                    stop_loss=d.stop_loss,
                    take_profit=d.take_profit,
                )

                # Async frontend JSONL decision log
                await self.decision_logger.log(t, rep, bm)

            log.info(
                "[Benchmark] %d tickers | Total %.0fms | Pipeline %.0fms | Order %.0fms | Safe freq: every %ds",
                len(self.tickers),
                total_ms,
                pipe_ms,
                order_ms,
                max(30, int(total_ms / 1000) * 2),
            )

        results["total_ms"] = total_ms
        return results

    async def _manage_positions(
        self,
        client: TradingClient,
        ticker: str,
        positions: list[dict[str, Any]],
        cycle_ts: str,
        decision: Any,
    ) -> None:
        for pos in positions:
            sym = _get_sym(pos) or ticker
            pos.setdefault("opened_at", self._open_state[ticker].get(sym, cycle_ts))
            out = decide_exit(
                pos, decision, horizon_days=self.settings.trade_horizon_days
            )
            if out["action"] != "hold":
                await manage_and_close(
                    client, pos, self.journals[ticker], out["action"], ticker, cycle_ts
                )
                self._open_state[ticker].pop(sym, None)
                # Update checkpoint outcome on trade close
                exit_price = float(pos.get("mark_price") or pos.get("current_price") or 0.0)
                entry_price = float(pos.get("avg_entry_price") or 0.0)
                pnl = round(exit_price - entry_price, 2)
                pnl_pct = round(pnl / entry_price, 4) if entry_price > 0 else 0.0
                self.journals[ticker].update_checkpoint_outcome(
                    ticker=ticker,
                    cycle_ts=cycle_ts,
                    exit_price=exit_price,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    exit_reason=out["action"],
                )

    def _update_day_open(self, ticker: str, equity: float) -> None:
        day = date.today().isoformat()
        if self._day_key != day:
            self._day_key = day
            self._day_open_equity.clear()
            for t in self.tickers:
                self._day_locked[t] = False

        if ticker not in self._day_open_equity:
            self._day_open_equity[ticker] = equity

        open_eq = self._day_open_equity.get(ticker, 0.0)
        if open_eq > 0:
            loss_pct = (equity - open_eq) / open_eq
            if loss_pct <= -self.settings.daily_loss_limit_pct:
                self._day_locked[ticker] = True

        # Portfolio drawdown kill switch: track the equity peak across the
        # whole run (not just today) and lock new entries when drawdown from
        # the peak exceeds the configured threshold.
        peak = self._portfolio_peak_equity
        if equity > peak:
            self._portfolio_peak_equity = equity
        if self._portfolio_peak_equity > 0:
            dd_pct = (self._portfolio_peak_equity - equity) / self._portfolio_peak_equity
            if dd_pct >= self.settings.max_portfolio_drawdown_pct:
                self._portfolio_dd_locked = True


class PaperRunner:
    """Single-ticker paper-trading loop (backward compatibility wrapper)."""

    def __init__(
        self,
        settings: Settings,
        ticker: str,
        journal_path: str | None = None,
        verbose: bool = False,
    ) -> None:
        self.settings = settings
        self.ticker = ticker.strip().upper()
        self.verbose = verbose
        self.journal = TradeJournal(journal_path or str(data_path("trading_journal.db")))
        self.decision_logger = DecisionLogger(data_path("logs"))
        self._open_state: dict[str, str] = {}
        self._day_open_equity: float | None = None
        self._day_key: str = ""
        self._day_locked = False

    async def start(self) -> None:
        self._rebuild_open_state()
        interval = max(1, self.settings.trading_interval_min) * 60
        log.info("Paper runner started for %s (interval %ds)", self.ticker, interval)
        while True:
            if not is_market_open():
                log.debug("market closed - skipping cycle")
            else:
                try:
                    await self.run_once()
                except Exception as exc:  # noqa: BLE001
                    log.exception("cycle failed: %s", exc)
            await asyncio.sleep(interval)

    def _rebuild_open_state(self) -> None:
        orders = self.journal.orders()
        for o in reversed(orders):
            symbol = o["symbol"]
            if symbol not in self._open_state and o["side"] == "buy":
                self._open_state[symbol] = o.get("cycle_ts", "")

    async def run_once(self) -> dict[str, Any]:
        cycle_ts = datetime.now(timezone.utc).isoformat()
        result: dict[str, Any] = {"cycle_ts": cycle_ts, "ticker": self.ticker}

        async with TradingClient(self.settings) as client:
            account = await client.get_account()
            equity = float(account.get("equity") or 0.0)
            cash = float(account.get("cash") or 0.0)
            self.journal.record_equity(cycle_ts, equity, cash)

            self._update_day_open(equity)
            if self._day_locked:
                result["entry_blocked"] = "daily_loss_limit"

            report = await Pipeline(self.settings, verbose=self.verbose).run(self.ticker)
            decision = report.decision
            self.journal.record_cycle(
                cycle_ts,
                self.ticker,
                decision.trade_decision,
                decision.composite_bias,
                decision.confidence_score,
                report.model_dump(),
                benchmark=report.benchmark,
            )
            result["decision"] = decision.trade_decision
            result["confidence"] = decision.confidence_score

            positions = await client.get_positions()
            ticker_positions = [p for p in positions if _get_sym(p).startswith(self.ticker)]
            await self._manage_positions(client, ticker_positions, cycle_ts, decision)

            positions = await client.get_positions()
            ticker_positions = [p for p in positions if _get_sym(p).startswith(self.ticker)]
            open_count = len(ticker_positions)
            result["open_positions"] = open_count

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

            await self.decision_logger.log(self.ticker, report, report.benchmark)

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
            symbol = _get_sym(pos) or self.ticker
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
