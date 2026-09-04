"""Phase 3 - Risk Agent.

Receives the raw Phase 1 bundle and the Phase 2 ``PredictionResult`` and:

1. Fetches the option chain (greeks + IV + bid/ask) from Alpaca — degrading
   gracefully to Phase 2's estimated IV when the options feed is unavailable.
2. Derives a stop from ATR + the nearest support level, and a target from the
   forecast high (or a 2R fallback).
3. Runs the deterministic risk tools (greeks, position size, max loss, risk
   score) in a worker thread.
4. Returns a typed ``RiskResult``.

No LLM is involved; every number is deterministic (or sourced from Alpaca).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from agents.base import BaseAgent
from alpaca.client import AlpacaClient
from alpaca.options import get_option_chain
from schemas.prediction import Phase1Bundle, PredictionResult
from schemas.risk import (
    EquityPosition,
    Greeks,
    OptionPosition,
    PositionRecommendation,
    RiskResult,
)
from tools.risk_tools.greeks import calculate_greeks
from tools.risk_tools.max_loss import calculate_max_loss
from tools.risk_tools.position_size import calculate_position_size
from tools.risk_tools.risk_score import calculate_risk_score, risk_reward_ratio
from tuning import TuningConfig
from utils.config import Settings

log = logging.getLogger("market_intel_agent.risk_agent")


class RiskAgent(BaseAgent):
    """Computes risk metrics and a position recommendation (Phase 3)."""

    name = "risk"
    description = (
        "Fetches the option chain (greeks + IV), derives stop/target levels, "
        "and computes position size, max loss and a composite risk score."
    )
    phase = 3
    tools = [
        "calculate_greeks",
        "calculate_position_size",
        "calculate_max_loss",
        "calculate_risk_score",
    ]

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def run(
        self,
        phase1: Any,
        phase2: Any,
        tuning: TuningConfig | None = None,
    ) -> RiskResult:
        """Run Phase 3 risk assessment.

        Parameters
        ----------
        phase1:
            Either the raw dict from ``Pipeline._phase1()`` or a typed
            ``Phase1Bundle``.
        phase2:
            The ``PredictionResult`` from Phase 2.
        tuning:
            Optional ``TuningConfig`` overriding risk parameters.
        """
        if isinstance(phase1, dict):
            bundle = Phase1Bundle.from_phase1_dict(phase1)
        else:
            bundle = phase1

        prediction = phase2 if isinstance(phase2, PredictionResult) else PredictionResult()

        spot = bundle.market.price
        if spot <= 0:
            return RiskResult(
                status="insufficient_data",
                summary="No valid market price — cannot assess risk.",
                errors=["Invalid or zero market price."],
            )

        ticker = bundle.historical.symbol or bundle.news.ticker or ""
        horizon = max(1, prediction.forecast_horizon_days)

        errors: list[str] = []
        chain: list[dict[str, Any]] = []
        try:
            async with AlpacaClient(self.settings) as client:
                chain = await get_option_chain(
                    client,
                    ticker,
                    spot,
                    feed=self.settings.alpaca_options_feed,
                    horizon_days=horizon,
                )
        except Exception as exc:  # noqa: BLE001 - degrade gracefully on any feed error
            log.warning("Option chain fetch failed: %s", exc)
            errors.append(f"option chain fetch failed: {exc}")

        computed = await asyncio.to_thread(
            _compute_risk, bundle, prediction, chain, self.settings, errors, tuning
        )
        return computed


def _compute_risk(
    bundle: Phase1Bundle,
    prediction: PredictionResult,
    chain: list[dict[str, Any]],
    settings: Settings,
    errors: list[str],
    tuning: "TuningConfig | None" = None,
) -> RiskResult:
    """Synchronous risk computation bundle (runs in a worker thread)."""
    spot = bundle.market.price
    atr = bundle.market.atr14
    historical = bundle.historical

    # ---- Phase 1 risk/level inputs ----
    levels = historical.levels or {}
    support_levels = levels.get("support_levels", []) or []
    risk_bundle = historical.risk or {}
    dd = risk_bundle.get("calculate_drawdown", {}) or {}
    gaps = risk_bundle.get("analyze_gaps", {}) or {}
    var_bundle = risk_bundle.get("calculate_value_at_risk", {}) or {}

    drawdown_risk = str(dd.get("risk_level", "low"))
    gap_frequency = str(gaps.get("gap_frequency", "rare"))
    avg_gap_size = float(gaps.get("avg_gap_size", 0.0) or 0.0)
    var_pct = float(var_bundle.get("var", 0.0) or 0.0)
    cvar_pct = float(var_bundle.get("cvar", 0.0) or 0.0)

    # ---- Stop / target ----
    atr_distance = 1.5 * atr if atr > 0 else 0.0
    support_below = [lv.get("level", 0.0) for lv in support_levels if (lv.get("level") or 0.0) < spot]
    support_distance = spot - max(support_below) if support_below else 0.0
    stop_distance = max(atr_distance, support_distance) or (spot * 0.02)
    stop = spot - stop_distance

    target = prediction.price_forecast_high
    if target <= spot:
        target = spot + 2.0 * stop_distance

    # ---- Greeks ----
    greeks = calculate_greeks(
        chain,
        spot,
        prediction.iv_forecast,
        horizon_days=max(1, prediction.forecast_horizon_days),
    )
    if greeks.get("errors"):
        errors.extend(greeks["errors"])
    greeks_source = greeks["greeks_source"]
    iv_quality = 1.0 if greeks_source == "alpaca_option_chain" else 0.5
    call_delta = float(greeks["greeks"]["call"]["delta"] or 0.0)
    premium = float(greeks["call_premium"] or 0.0)
    spread_pct = float(greeks["spread_pct"] or 0.0)

    # ---- Position sizing ----
    position = calculate_position_size(
        capital=settings.account_capital,
        risk_per_trade_pct=settings.risk_per_trade_pct,
        entry=spot,
        stop=stop,
        premium=premium,
        delta=call_delta if call_delta > 0 else 1.0,
        confidence=prediction.confidence,
        iv_quality=iv_quality,
        spread_pct=spread_pct,
        drawdown_risk=drawdown_risk,
        max_position_pct=settings.max_position_pct,
        tuning=tuning,
    )
    if position.get("errors"):
        errors.extend(position["errors"])

    # ---- Max loss ----
    max_loss = calculate_max_loss(
        entry=spot,
        stop=stop,
        position_size=position["equity_shares"],
        avg_gap_size=avg_gap_size,
        gap_frequency=gap_frequency,
        var_pct=var_pct,
        cvar_pct=cvar_pct,
        tuning=tuning,
    )

    # ---- Risk:reward + composite score ----
    r_r = risk_reward_ratio(spot, stop, target)
    score = calculate_risk_score(
        vol_regime=prediction.vol_regime,
        iv_percentile=prediction.vol_percentile,
        drawdown_risk=drawdown_risk,
        gap_frequency=gap_frequency,
        spread_pct=spread_pct,
        max_loss_pct=max_loss["max_loss_pct"],
        confidence=prediction.confidence,
        tuning=tuning,
    )

    risk_level = score["risk_level"]

    rec = PositionRecommendation(
        equity=EquityPosition(
            shares=position["equity_shares"],
            dollar_value=position["equity_dollar_value"],
        ),
        option=OptionPosition(
            contracts=position["option_contracts"],
            premium_risk=position["option_premium_risk"],
            delta_exposure=position["delta_exposure"],
        ),
    )

    status = "ok" if not errors else "partial"

    summary = (
        f"Risk {risk_level} (score {score['risk_score']:.0f}/100); "
        f"stop ${stop:.2f}, target ${target:.2f}, R:R {r_r:.2f}; "
        f"position ${rec.equity.dollar_value:,.0f} equity / "
        f"{rec.option.contracts:.0f} contracts; max loss "
        f"${max_loss['max_loss_dollars']:,.0f} ({max_loss['max_loss_pct'] * 100:.2f}%)"
    )

    return RiskResult(
        status=status,
        summary=summary,
        errors=errors,
        risk_score=score["risk_score"],
        risk_level=risk_level,
        greeks_source=greeks_source,
        call_greeks=Greeks(**greeks["greeks"]["call"]),
        put_greeks=Greeks(**greeks["greeks"]["put"]),
        iv_used=greeks["iv_used"],
        iv_source=greeks["iv_source"],
        spread_pct=spread_pct,
        implied_move_pct=greeks["implied_move_pct"],
        theta_per_day=greeks["theta_per_day"],
        stop_loss_level=round(stop, 4),
        take_profit_level=round(target, 4),
        risk_reward_ratio=r_r,
        position_recommendation=rec,
        capital_at_risk_pct=position["capital_at_risk_pct"],
        max_loss_dollars=max_loss["max_loss_dollars"],
        max_loss_pct=max_loss["max_loss_pct"],
        tail_var_dollars=max_loss["tail_var_dollars"],
        tail_cvar_dollars=max_loss["tail_cvar_dollars"],
        risk_metrics={
            "calculate_greeks": greeks,
            "calculate_position_size": position,
            "calculate_max_loss": max_loss,
            "calculate_risk_score": score,
            "risk_reward_ratio": r_r,
        },
    )
