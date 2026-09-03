"""Phase 5 - Order executor.

Maps a ``DecisionResult`` from Phase 4 into concrete paper orders.

Policies:
- ``long_equity`` -> a server-side bracket order (buy shares with a stop-loss
  and take-profit attached, so Alpaca monitors them even if our loop is down).
- ``long_call`` / ``long_put`` -> a buy-to-open limit order at (mid +/- slippage)
  for the exact ATM contract chosen by Phase 3 (OCC ``call_symbol``/``put_symbol``).
- ``hold`` / ``avoid`` -> no order.

If an options order is rejected by the account (e.g. no options permission), the
executor degrades to the next-ranked opportunity (equity) and journals the
reason instead of failing.
"""

from __future__ import annotations

import hashlib
from typing import Any

from alpaca.trading import TradingClient
from schemas.decision import DecisionResult
from utils.config import Settings


class ExecutorError(RuntimeError):
    pass


def build_client_order_id(ticker: str, cycle_ts: str, nonce: str = "0") -> str:
    """Idempotent, deterministic order id for a given cycle.

    Using a deterministic id means re-running the same cycle (e.g. after a
    crash) can never place a duplicate order.
    """
    raw = f"{ticker}-{cycle_ts}-{nonce}"
    digest = hashlib.sha256(raw.encode()).hexdigest()[:16]
    return f"{ticker}-{digest}"


def build_order_payload(
    decision: DecisionResult,
    settings: Settings,
    cycle_ts: str,
    options: dict[str, Any],
) -> dict[str, Any] | None:
    """Build the Alpaca order payload for a decision, or None for hold/avoid."""
    if decision.trade_decision not in ("long_equity", "long_call", "long_put"):
        return None

    if decision.trade_decision == "long_equity":
        shares = int(decision.position_shares)
        if shares <= 0:
            return None
        limit_price = _limit_price(decision.entry_price, settings)
        payload: dict[str, Any] = {
            "symbol": decision.symbol,
            "qty": str(shares),
            "side": "buy",
            "type": settings.order_type,
            "time_in_force": "day",
            "client_order_id": build_client_order_id(decision.symbol, cycle_ts, "entry"),
        }
        if settings.order_type == "limit":
            payload["limit_price"] = f"{limit_price:.2f}"
        # Attach stop/target bracket so Alpaca manages the exit.
        if decision.stop_loss > 0 and decision.take_profit > 0:
            payload["order_class"] = "bracket"
            payload["stop_loss"] = {"stop_price": f"{decision.stop_loss:.2f}"}
            payload["take_profit"] = {"limit_price": f"{decision.take_profit:.2f}"}
        return payload

    # Options: long call / long put.
    occ = (
        options.get("call_symbol") if decision.option_type == "call" else options.get("put_symbol")
    )
    if not occ:
        return None
    contracts = int(decision.option_contracts)
    if contracts <= 0:
        return None
    # Reference premium from Phase 3 for the requested option type.
    premium = options.get(
        "call_premium" if decision.option_type == "call" else "put_premium", 0.0
    )
    limit_price = _limit_price(float(premium), settings)
    payload = {
        "symbol": occ,
        "qty": str(contracts),
        "side": "buy",
        "type": settings.order_type,
        "time_in_force": "day",
        "client_order_id": build_client_order_id(occ, cycle_ts, "entry"),
        "class": "simple",
    }
    if settings.order_type == "limit":
        payload["limit_price"] = f"{limit_price:.2f}"
    return payload


def _limit_price(reference: float, settings: Settings) -> float:
    """Mid-based limit price with optional slippage away from mid."""
    if settings.order_type == "market":
        return round(reference, 4)
    slip = settings.limit_slippage_pct
    # For buys we pay slightly more than mid to improve fill odds.
    return round(reference * (1.0 + slip), 4)


async def execute_decision(
    client: TradingClient,
    decision: DecisionResult,
    settings: Settings,
    options: dict[str, Any],
    cycle_ts: str,
) -> dict[str, Any]:
    """Submit the order for a decision and return a summary record."""
    payload = build_order_payload(decision, settings, cycle_ts, options)
    if payload is None:
        return {"action": "none", "reason": decision.trade_decision}

    try:
        order = await client.submit_order(payload)
    except Exception as exc:  # noqa: BLE001 - degrade to equity fallback
        if decision.trade_decision in ("long_call", "long_put"):
            equity_decision = _as_equity_decision(decision)
            if equity_decision is not None:
                return await execute_decision(client, equity_decision, settings, options, cycle_ts)
        return {"action": "error", "reason": str(exc), "decision": decision.trade_decision}

    return {
        "action": "order",
        "order_id": order.get("id"),
        "client_order_id": payload["client_order_id"],
        "symbol": payload["symbol"],
        "side": payload["side"],
        "qty": payload["qty"],
        "type": payload["type"],
        "status": order.get("status"),
    }


def _as_equity_decision(decision: DecisionResult) -> DecisionResult | None:
    """Copy a call/put decision into an equity decision (for fallback)."""
    if decision.position_shares <= 0:
        return None
    return DecisionResult(
        status=decision.status,
        summary="Fallback to equity from options",
        trade_decision="long_equity",
        confidence_score=decision.confidence_score,
        composite_bias=decision.composite_bias,
        agreement_score=decision.agreement_score,
        symbol=decision.symbol,
        instrument="equity",
        entry_price=decision.entry_price,
        stop_loss=decision.stop_loss,
        take_profit=decision.take_profit,
        risk_reward_ratio=decision.risk_reward_ratio,
        position_shares=decision.position_shares,
        rationale=f"Options rejected; falling back to equity. {decision.rationale}",
        opportunities=decision.opportunities,
        decision_metrics=decision.decision_metrics,
    )
