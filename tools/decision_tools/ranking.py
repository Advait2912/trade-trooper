"""Phase 4 - Opportunity ranking tool.

``rank_opportunities`` evaluates one or more trade candidates (long call /
long put / long equity) against deterministic *gates* and produces a 0-100
score per candidate so the best instrument/direction wins.  The design is
multi-candidate and thus future-proof for a multi-symbol Decision Agent, but
it already handles the single-ticker case used by Phase 4.

Direction-aware and premium-defined: the agent is bullish > long call/equity,
bearish > long put, otherwise hold.  Scores are deterministic and auditable.
"""

from __future__ import annotations

from typing import Any

from tuning import DEFAULT_TUNING, TuningConfig


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def calculate_opportunity_score(
    candidate: dict[str, Any],
    composite_bias: str = "neutral",
    agreement_score: float = 0.5,
    risk_score: float = 50.0,
    spread_pct: float = 0.0,
    greeks_source: str = "black_scholes_estimated",
    tuning: TuningConfig | None = None,
) -> dict[str, Any]:
    """Score a single candidate 0-100 and report component breakdown.

    ``direction_matches`` on the candidate indicates whether its instrument
    suits the composite bias (call/equity for bullish, put for bearish); a
    mismatch suppresses the signal component to penalise contradicting the
    composite view.
    """
    t = tuning or DEFAULT_TUNING
    cm = t.component_max
    direction_matches = bool(candidate.get("direction_matches", True))
    confidence = _clamp(float(candidate.get("confidence", 0.5) or 0.5), 0.0, 1.0)
    r_r = max(0.0, float(candidate.get("r_r", 0.0) or 0.0))
    risk_score = _clamp(float(risk_score), 0.0, 100.0)
    spread_pct = _clamp(float(spread_pct), 0.0, 1.0)
    execution_mult = 1.0 if greeks_source == "alpaca_option_chain" else t.execution_fallback_mult

    # 1) signal / agreement + confidence, gated by direction agreement.
    signal = 0.5 * _clamp(agreement_score, 0.0, 1.0) + 0.5 * confidence
    signal_component = cm["signal"] * signal
    if not direction_matches:
        signal_component *= 0.15

    # 2) reward/risk: full band at rr_full_at.
    rr_component = cm["reward_risk"] * _clamp(r_r / t.rr_full_at, 0.0, 1.0)

    # 3) risk quality: lower composite risk is better.
    risk_component = cm["risk_quality"] * (1.0 - risk_score / 100.0)

    # 4) execution: spread + greeks source quality.
    exec_component = (
        cm["execution"] * (1.0 - spread_pct / t.spread_full_at) * execution_mult
    )

    score = signal_component + rr_component + risk_component + exec_component
    if candidate.get("instrument") == "equity":
        score += t.equity_score_boost
    score = _clamp(score, 0.0, 100.0)

    return {
        "score": round(score, 2),
        "components": {
            "signal": round(signal_component, 2),
            "reward_risk": round(rr_component, 2),
            "risk_quality": round(risk_component, 2),
            "execution": round(exec_component, 2),
        },
    }


def rank_opportunities(
    candidates: list[dict[str, Any]],
    context: dict[str, Any],
    min_confidence: float = 0.35,
    min_risk_reward: float = 1.0,
    tuning: TuningConfig | None = None,
) -> dict[str, Any]:
    """Apply gates, score and rank candidates -> trade decision.

    Parameters
    ----------
    candidates:
        List of candidate dicts (see ``calculate_opportunity_score``).
    context:
        ``composite_bias``, ``agreement_score``, ``risk_score``,
        ``risk_level``, ``spread_pct``, ``greeks_source``.
    min_confidence / min_risk_reward:
        Gate thresholds (from Settings).

    Returns a dict with ``trade_decision``, ``confidence_score``,
    ``opportunities`` (sorted, scored), ``gates`` and ``rationale``.
    """
    composite_bias = str(context.get("composite_bias", "neutral"))
    agreement = _clamp(float(context.get("agreement_score", 0.5) or 0.5), 0.0, 1.0)
    risk_score = _clamp(float(context.get("risk_score", 50.0) or 50.0), 0.0, 100.0)
    risk_level = str(context.get("risk_level", "moderate"))
    spread_pct = float(context.get("spread_pct", 0.0) or 0.0)
    greeks_source = str(context.get("greeks_source", "black_scholes_estimated"))

    scored: list[dict[str, Any]] = []
    gates: dict[str, Any] = {}
    rationale: list[str] = []

    for cand in candidates:
        conf = _clamp(float(cand.get("confidence", 0.5) or 0.5), 0.0, 1.0)
        r_r = max(0.0, float(cand.get("r_r", 0.0) or 0.0))

        gate_conf = conf >= min_confidence
        gate_rr = r_r >= min_risk_reward
        gate_risk = risk_level not in ("very_high",)
        cand_gates = {
            "min_confidence": {"passed": gate_conf, "value": conf, "detail": f"conf >= {min_confidence:.2f}"},
            "min_risk_reward": {"passed": gate_rr, "value": r_r, "detail": f"R:R >= {min_risk_reward:.2f}"},
            "risk_level_ok": {"passed": gate_risk, "value": risk_score, "detail": f"risk={risk_level}"},
        }
        gates[cand.get("id", f"{cand.get('instrument', '?')}_{cand.get('option_type', '')}")] = cand_gates

        score_info = calculate_opportunity_score(
            cand,
            composite_bias=composite_bias,
            agreement_score=agreement,
            risk_score=risk_score,
            spread_pct=spread_pct,
            greeks_source=greeks_source,
            tuning=tuning,
        )
        cand = dict(cand)
        cand["score"] = score_info["score"]
        cand["score_components"] = score_info["components"]
        cand["gates"] = cand_gates
        cand["gates_passed"] = all(g["passed"] for g in cand_gates.values())
        cand["direction_matches"] = bool(cand.get("direction_matches", True))
        scored.append(cand)

    scored.sort(key=lambda c: (-c["score"], c.get("rank", 0)))

    # pick the best *eligible* candidate
    eligible = [c for c in scored if c["direction_matches"] and c["gates_passed"]]
    if eligible:
        best = eligible[0]
        trade_decision = _decision_from_candidate(best)
        confidence_score = _clamp(
            0.5 * agreement + 0.3 * best.get("confidence", 0.5) + 0.2 * (1.0 - risk_score / 100.0),
            0.0, 1.0,
        )
        rationale.append(
            f"Selected {trade_decision} ({round(best['score'], 1)}/100) "
            f"from {len(scored)} candidate(s); bias={composite_bias}"
        )
    elif risk_level == "very_high":
        trade_decision = "avoid"
        confidence_score = 0.0
        rationale.append(f"Risk level is very_high ({risk_score:.0f}/100) — avoiding all trades.")
    else:
        trade_decision = "hold"
        best_eligible_score = max(
            (c["score"] for c in scored if c["direction_matches"]), default=0.0
        )
        confidence_score = _clamp(0.5 * agreement, 0.0, 1.0)
        if not scored:
            rationale.append("No candidates supplied.")
        elif not eligible and best_eligible_score > 0:
            unmet = _unmet_gate_names(scored)
            rationale.append(f"No candidate passed the gates (unmet: {unmet or 'n/a'}).")
        else:
            rationale.append("Composite bias is neutral or no eligible opportunity.")

    # build Opportunity-shaped output
    opportunities = []
    for i, c in enumerate(scored, start=1):
        opp = {
            "symbol": c.get("symbol", ""),
            "direction": "long",
            "instrument": c.get("instrument", ""),
            "option_type": c.get("option_type", ""),
            "score": c["score"],
            "rank": i,
            "entry_price": c.get("entry", 0.0),
            "stop_loss": c.get("stop", 0.0),
            "take_profit": c.get("target", 0.0),
            "risk_reward_ratio": c.get("r_r", 0.0),
            "position_shares": c.get("shares", 0.0),
            "option_contracts": c.get("contracts", 0.0),
            "premium_risk": c.get("premium_risk", 0.0),
            "rationale": (
                f"score={c['score']:.1f}; "
                f"{'passes' if c['gates_passed'] else 'fails'} gates; "
                f"{'matches' if c['direction_matches'] else 'mismatches'} bias"
            ),
        }
        opportunities.append(opp)

    return {
        "trade_decision": trade_decision,
        "confidence_score": round(confidence_score, 4),
        "opportunities": opportunities,
        "gates": gates,
        "rationale": " ".join(rationale),
    }


def _decision_from_candidate(cand: dict[str, Any]) -> str:
    instrument = cand.get("instrument", "")
    option_type = cand.get("option_type", "")
    if instrument == "option":
        return f"long_{option_type}" if option_type in ("call", "put") else "long_option"
    if instrument == "equity":
        return "long_equity"
    return "hold"


def _unmet_gate_names(scored: list[dict[str, Any]]) -> list[str]:
    """Collect the (deterministic) gate names that blocked eligible candidates."""
    names: list[str] = []
    for c in scored:
        if c.get("direction_matches"):
            for g_name, g in c.get("gates", {}).items():
                if not g.get("passed") and g_name not in names:
                    names.append(g_name)
    return names
