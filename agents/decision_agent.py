"""Phase 4 - Decision Agent.

The final deterministic stage.  Receives Phase 1 (bundle), Phase 2
(``PredictionResult``) and Phase 3 (``RiskResult``) and:

1. ``synthesize_signals`` — aggregates every directional signal into a
   composite bias + agreement score.
2. Builds up to three trade candidates (long call / long put / long equity)
   and sizes the put candidate itself with the Phase 3 position-size tool
   (put premium + |put delta| from the Phase 3 greeks) — the Risk Agent only
   sizes the call; the Decision Agent picks the instrument.
3. ``rank_opportunities`` — gates + scores each candidate and returns a
   ``trade_decision`` (long_call | long_put | long_equity | hold | avoid) and
   a ``confidence_score``.

No LLM is involved; every number is deterministic.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from agents.base import BaseAgent
from schemas.decision import DecisionResult, Opportunity
from schemas.prediction import Phase1Bundle, PredictionResult
from schemas.risk import RiskResult
from tools.decision_tools.ranking import rank_opportunities
from tools.decision_tools.signals import synthesize_signals
from tools.risk_tools.position_size import calculate_position_size
from tuning import TuningConfig
from utils.config import Settings

log = logging.getLogger("market_intel_agent.decision_agent")

# stop/target multipliers used to *mirror* the Phase 3 (bullish) levels for a
# long-put candidate without re-deriving levels from scratch.
_PUT_TARGET_MULT = 2.0


class DecisionAgent(BaseAgent):
    """Synthesizes signals and ranks opportunities into a trade decision."""

    name = "decision"
    description = (
        "Aggregates all phase signals, gates + ranks call/put/equity "
        "opportunities and returns a deterministic trade decision."
    )
    phase = 4
    tools = ["synthesize_signals", "rank_opportunities", "calculate_position_size"]

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def run(self, phase1: Any, phase2: Any, phase3: Any) -> DecisionResult:
        """Run Phase 4 decision synthesis.

        ``phase1`` may be a raw dict (from ``Pipeline._phase1``) or a typed
        ``Phase1Bundle``; ``phase2``/``phase3`` are typed phase results.
        """
        if isinstance(phase1, dict):
            bundle = Phase1Bundle.from_phase1_dict(phase1)
        else:
            bundle = phase1

        prediction = phase2 if isinstance(phase2, PredictionResult) else PredictionResult()
        risk = phase3 if isinstance(phase3, RiskResult) else RiskResult()

        # Guard: no price -> no trade.
        if bundle.market.price <= 0:
            return DecisionResult(
                status="insufficient_data",
                summary="No valid market price — cannot decide.",
                errors=["Invalid or zero market price."],
                trade_decision="hold",
                rationale="Insufficient price data.",
            )

        errors: list[str] = []
        try:
            return await asyncio.to_thread(
                _decide, bundle, prediction, risk, self.settings, errors, None
            )
        except Exception as exc:  # noqa: BLE001 - degrade gracefully
            log.exception("DecisionAgent failed: %s", exc)
            return DecisionResult(
                status="error",
                summary=f"Decision error: {exc}",
                errors=[f"DecisionAgent error: {exc}"],
                trade_decision="hold",
                rationale="Decision agent error.",
            )


def _decide(
    bundle: Phase1Bundle,
    prediction: PredictionResult,
    risk: RiskResult,
    settings: Settings,
    errors: list[str],
    tuning: "TuningConfig | None" = None,
) -> DecisionResult:
    """Synchronous decision computation (runs in a worker thread)."""
    spot = bundle.market.price
    symbol = bundle.historical.symbol or bundle.news.ticker or ""

    # ------------------------------------------------------------------
    # Step 1: directional synthesis
    # ------------------------------------------------------------------
    sig = synthesize_signals(bundle, prediction, risk, tuning=tuning)
    composite_bias = sig["composite_bias"]
    agreement = sig["agreement_score"]

    # ------------------------------------------------------------------
    # Step 2: build candidates and re-size the put with Phase 3's tool
    # ------------------------------------------------------------------
    candidates = _build_candidates(
        bundle, prediction, risk, spot, composite_bias, settings, errors, tuning
    )

    # ------------------------------------------------------------------
    # Step 3: gate + score + rank
    # ------------------------------------------------------------------
    # Risk posture used for scoring comes from the composite risk number.
    risk_score = float(risk.risk_score or 0.0)
    risk_level = risk.risk_level or "moderate"
    greeks = (risk.risk_metrics or {}).get("calculate_greeks", {}) or {}
    spread_pct = float(greeks.get("spread_pct", 0.0) or 0.0)
    greeks_source = str(greeks.get("greeks_source", "black_scholes_estimated"))

    ranked = rank_opportunities(
        candidates,
        context={
            "composite_bias": composite_bias,
            "agreement_score": agreement,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "spread_pct": spread_pct,
            "greeks_source": greeks_source,
        },
        min_confidence=settings.min_confidence,
        min_risk_reward=settings.min_risk_reward,
        tuning=tuning,
    )

    decision = ranked["trade_decision"]
    confidence = ranked["confidence_score"]
    top = ranked["opportunities"][0] if ranked["opportunities"] else None

    # ------------------------------------------------------------------
    # Build typed DecisionResult
    # ------------------------------------------------------------------
    chosen_type = ""
    if top is not None and decision.startswith("long_"):
        chosen_type = decision.removeprefix("long_")
    elif decision == "long_equity":
        chosen_type = "equity"

    rationale = " ".join(
        [x for x in (ranked["rationale"], _rationale_from_risk(risk)) if x]
    )

    opportunity_models = [
        Opportunity(**{k: v for k, v in o.items() if k in Opportunity.model_fields})
        for o in ranked["opportunities"]
    ]

    chosen = _selected_opportunity(ranked["opportunities"], decision)

    return DecisionResult(
        status="ok" if not errors else "partial",
        summary=summary(decision, confidence, composite_bias, agreement, chosen),
        errors=errors,
        trade_decision=decision,
        confidence_score=round(confidence, 4),
        composite_bias=composite_bias,
        agreement_score=round(agreement, 4),
        divergences=sig["divergences"],
        symbol=symbol,
        instrument="option" if chosen_type in ("call", "put") else ("equity" if chosen_type == "equity" else "none"),
        option_type=chosen_type if chosen_type in ("call", "put") else "",
        entry_price=float(chosen["entry_price"]) if chosen else 0.0,
        stop_loss=float(chosen["stop_loss"]) if chosen else 0.0,
        take_profit=float(chosen["take_profit"]) if chosen else 0.0,
        risk_reward_ratio=float(chosen["risk_reward_ratio"]) if chosen else 0.0,
        position_shares=float(chosen["position_shares"]) if chosen else 0.0,
        option_contracts=float(chosen["option_contracts"]) if chosen else 0.0,
        premium_risk=float(chosen["premium_risk"]) if chosen else 0.0,
        rationale=rationale,
        opportunities=opportunity_models,
        decision_metrics={
            "synthesize_signals": sig,
            "rank_opportunities": ranked,
            "prediction_confidence": prediction.confidence,
            "risk_score": risk_score,
            "risk_level": risk_level,
        },
    )


def _build_candidates(
    bundle: Phase1Bundle,
    prediction: PredictionResult,
    risk: RiskResult,
    spot: float,
    composite_bias: str,
    settings: Settings,
    errors: list[str],
    tuning: "TuningConfig | None" = None,
) -> list[dict[str, Any]]:
    """Build call / put / equity candidates sized deterministically.

    The call and equity candidates are sized by Phase 3 (``RiskResult``); the
    put candidate is sized by re-running ``calculate_position_size`` with the
    Phase 3 put premium / |put delta| (the Risk Agent only sizes the call).
    """
    rec = risk.position_recommendation
    eq = rec.equity
    opt = rec.option
    greeks = (risk.risk_metrics or {}).get("calculate_greeks", {}) or {}
    put_g = ((greeks.get("greeks") or {}).get("put") or {})
    put_premium = float(greeks.get("put_premium", 0.0) or 0.0)
    put_delta = abs(float(put_g.get("delta", 0.0) or 0.0))
    cand_confidence = float(
        prediction.confidence if prediction.confidence > 0 else 0.5
    )

    call_distance = spot - risk.stop_loss_level if risk.stop_loss_level > 0 else spot * 0.02
    stop_distance = call_distance if call_distance > 0 else spot * 0.02
    iv_quality = 1.0 if risk.greeks_source == "alpaca_option_chain" else 0.5

    candidates: list[dict[str, Any]] = []

    # long call (bullish instrument) — sized by Phase 3
    if composite_bias == "bullish" and opt.contracts > 0:
        candidates.append(
            _candidate("call", spot, risk, matches=True, confidence=cand_confidence,
                       contracts=opt.contracts, premium_risk=opt.premium_risk,
                       stop_distance=stop_distance)
        )

    # long put (bearish instrument) — sized here with Phase 3's tool.
    # The tool only needs a valid stop (any distance) to compute the option
    # premium risk; the candidate's *displayed* put stop is mirrored above.
    if composite_bias == "bearish" and put_premium > 0 and put_delta > 0:
        put_size = calculate_position_size(
            capital=settings.account_capital,
            risk_per_trade_pct=settings.risk_per_trade_pct,
            entry=spot,
            stop=spot - stop_distance,  # valid stop avoids tool error
            premium=put_premium,
            delta=put_delta,
            confidence=cand_confidence,
            iv_quality=iv_quality,
            spread_pct=float(greeks.get("spread_pct", 0.0) or 0.0),
            drawdown_risk=_drawdown_risk(bundle),
            max_position_pct=settings.max_position_pct,
            tuning=tuning,
        )
        if put_size.get("errors"):
            errors.extend(put_size["errors"])
        else:
            candidates.append(
                _candidate("put", spot, risk, matches=True, confidence=cand_confidence,
                           contracts=float(put_size["option_contracts"]),
                           premium_risk=float(put_size["option_premium_risk"]),
                           stop_distance=stop_distance)
            )

    # long equity — sized by Phase 3
    if composite_bias == "bullish" and eq.shares > 0:
        candidates.append(
            _candidate("", spot, risk, matches=True, confidence=cand_confidence,
                       shares=eq.shares, premium_risk=0.0, stop_distance=stop_distance)
        )

    return candidates


def _candidate(
    option_type: str,
    spot: float,
    risk: RiskResult,
    matches: bool,
    confidence: float,
    contracts: float = 0.0,
    premium_risk: float = 0.0,
    shares: float = 0.0,
    stop_distance: float = 0.0,
) -> dict[str, Any]:
    """Create a candidate dict from Phase 3 levels (mirrored for puts)."""
    r_r = risk.risk_reward_ratio
    if option_type == "put":
        # mirror the levels: put stop above entry, put target below entry.
        stop = spot + stop_distance
        target = spot - max((risk.take_profit_level - spot), stop_distance) * _PUT_TARGET_MULT
        target = max(target, 0.01)
        risk_amt = stop - spot
        reward_amt = spot - target
        r_r = round(reward_amt / risk_amt, 4) if risk_amt > 0 else 0.0
    else:
        stop = risk.stop_loss_level if risk.stop_loss_level > 0 else spot - stop_distance
        target = risk.take_profit_level if risk.take_profit_level > spot else spot + 2 * stop_distance

    return {
        "id": option_type or "equity",
        "symbol": "",
        "instrument": "option" if option_type else "equity",
        "option_type": option_type,
        "direction_matches": matches,
        "entry": spot,
        "stop": stop,
        "target": target,
        "r_r": r_r,
        "confidence": confidence,
        "premium_risk": premium_risk,
        "contracts": contracts,
        "shares": shares,
    }


def _drawdown_risk(bundle: Phase1Bundle) -> str:
    risk = (bundle.historical.risk or {}).get("calculate_drawdown", {}) or {}
    return str(risk.get("risk_level", "low"))


def _rationale_from_risk(risk: RiskResult) -> str:
    if risk.status == "insufficient_data":
        return "Risk assessment unavailable."
    return (
        f"R:R {risk.risk_reward_ratio:.2f}; "
        f"risk {risk.risk_level} ({risk.risk_score:.0f}/100); "
        f"greeks={risk.greeks_source}"
    )


def _selected_opportunity(
    opportunities: list[dict[str, Any]], decision: str
) -> dict[str, Any] | None:
    """Return the opportunity matching the chosen decision (top-ranked)."""
    if not opportunities:
        return None
    # decision maps to the top-ranked candidate that matches direction + gates.
    for opp in opportunities:
        label = ("option" if opp["instrument"] == "option" else "equity")
        if decision == "long_equity" and label == "equity":
            return opp
        if decision in ("long_call", "long_put") and opp["option_type"] == decision.removeprefix("long_"):
            return opp
    # fall back to the highest-ranked opportunity for display
    return opportunities[0]


def summary(
    decision: str,
    confidence: float,
    composite_bias: str,
    agreement: float,
    chosen: dict[str, Any] | None,
) -> str:
    if decision == "hold":
        return f"Hold — composite {composite_bias} (agreement {agreement:.2f}) does not clear the trade gates."
    if decision == "avoid":
        return f"Avoid — risk level is too high (agreement {agreement:.2f})."
    if chosen:
        pos = (
            f"{chosen['option_contracts']:.1f} contracts (premium risk ${chosen['premium_risk']:,.0f})"
            if chosen["instrument"] == "option"
            else f"{chosen['position_shares']:.1f} shares"
        )
        return (
            f"{decision.replace('_', ' ').title()} — entry ${chosen['entry_price']:.2f}, "
            f"stop ${chosen['stop_loss']:.2f}, target ${chosen['take_profit']:.2f}, "
            f"R:R {chosen['risk_reward_ratio']:.2f}; {pos}; "
            f"confidence {confidence:.2f} (agreement {agreement:.2f})."
        )
    return f"{decision.replace('_', ' ').title()} (confidence {confidence:.2f})."
