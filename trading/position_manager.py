"""Phase 5 - Position management.

Decides, each cycle, what to do with any *open* position:

- **stop hit / target hit**: close (equity brackets are managed server-side by
  Alpaca, but we double-check and close options/reference positions ourselves).
- **horizon expired**: close.
- **decision flipped to ``avoid``** (or a strongly opposing direction): close.

The manager is idempotent (quoting the open position each cycle) so a restart
never double-closes.  For equity, Alpaca bracket orders handle the exit; the
manager is the safety net and the close-on-flip/horizon rule.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from alpaca.trading import TradingClient
from schemas.decision import DecisionResult


def parse_iso(ts: str) -> datetime:
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return datetime.now(timezone.utc)


def decide_exit(
    position: dict[str, Any],
    decision: DecisionResult,
    horizon_days: int,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Return an exit action for an open position.

    ``position`` is an Alpaca position dict (has ``symbol``, ``qty``,
    ``avg_entry_price`` for equity; for options ``symbol`` is the OCC symbol
    and ``qty``/``avg_entry_price`` are per-contract / per-share-equivalent).
    """
    now = now or datetime.now(timezone.utc)
    priority: dict[str, int] = {"hold": 0, "close_horizon": 1, "close_flip": 2,
                                "close_target": 3, "close_stop": 4}
    action = "hold"

    # 1) stop / target (most urgent; brackets handle equity server-side, this is
    #    the safety net and already covers options).
    mark = float(position.get("mark_price") or position.get("current_price") or 0.0)
    avg_entry = float(position.get("avg_entry_price") or 0.0)
    if mark > 0 and avg_entry > 0:
        stop = float(position.get("stop_price") or 0.0)
        target = float(position.get("take_profit_price") or 0.0)
        if stop > 0 and mark <= stop:
            action = "close_stop"
        elif target > 0 and mark >= target:
            action = "close_target"

    # 2) decision flip to avoid
    if decision.trade_decision == "avoid" and priority["close_flip"] > priority[action]:
        action = "close_flip"

    # 3) horizon expired (lowest priority)
    opened = position.get("opened_at") or position.get("entry_ts")
    if opened:
        age_days = (now - parse_iso(str(opened))).days
        if age_days >= max(1, horizon_days) and priority["close_horizon"] > priority[action]:
            action = "close_horizon"

    return {"action": action, "mark_price": mark}


async def manage_and_close(
    client: TradingClient,
    position: dict[str, Any],
    journal: Any,
    reason: str,
    ticker: str,
    cycle_ts: str,
) -> None:
    """Liquidate a position and journal the realized trade."""
    symbol = position.get("symbol") or ticker
    try:
        await client.close_position(symbol)
    except Exception as exc:  # noqa: BLE001 - tolerate close failures
        journal.record_order(
            cycle_ts=cycle_ts, client_order_id=f"{symbol}-close-{cycle_ts}",
            symbol=symbol, side="sell", qty=float(position.get("qty") or 0.0),
            order_type="market", limit_price=None, status=f"close_failed: {exc}",
            filled_avg_price=None, order_id=None, reason=reason,
        )
        return

    journal.record_order(
        cycle_ts=cycle_ts, client_order_id=f"{symbol}-close-{cycle_ts}",
        symbol=symbol, side="sell", qty=float(position.get("qty") or 0.0),
        order_type="market", limit_price=None, status="filled",
        filled_avg_price=float(position.get("avg_entry_price") or 0.0),
        order_id=None, reason=reason,
    )

    closed_ts = datetime.now(timezone.utc).isoformat()
    qty = float(position.get("qty") or 0.0)
    entry = float(position.get("avg_entry_price") or 0.0)
    mark = float(position.get("mark_price") or 0.0)
    is_option = len(symbol) > 6
    multiplier = 100.0 if is_option else 1.0
    pnl = round((mark - entry) * qty * multiplier, 2)
    pnl_pct = round((mark - entry) / entry, 4) if entry > 0 else 0.0
    journal.record_trade(
        opened_ts=str(position.get("opened_at") or cycle_ts),
        closed_ts=closed_ts,
        ticker=ticker,
        instrument="option" if is_option else "equity",
        option_type="call" if "C" in symbol[-8:].upper() else ("put" if "P" in symbol[-8:].upper() else ""),
        symbol=symbol,
        quantity=qty,
        entry_price=entry,
        exit_price=mark,
        pnl=pnl,
        pnl_pct=pnl_pct,
        exit_reason=reason,
        cycle_ts=cycle_ts,
    )
