"""Phase 3 - Composite risk score and reward:risk tools.

``calculate_risk_score`` folds every deterministic risk signal into a single
0–100 score (higher = riskier) so the risk level is a bucket of one auditable
number rather than a chain of branches.  ``risk_reward_ratio`` is the classic
reward / risk for the proposed entry/stop/target.
"""

from __future__ import annotations

from typing import Any

_VOL_REGIME_SCORE: dict[str, float] = {
    "low": -10.0,
    "normal": 0.0,
    "high": 15.0,
    "very_high": 25.0,
}

_DRAWDOWN_SCORE: dict[str, float] = {
    "low": 0.0,
    "moderate": 10.0,
    "high": 20.0,
    "extreme": 30.0,
}

_GAP_SCORE: dict[str, float] = {
    "rare": 0.0,
    "occasional": 5.0,
    "frequent": 12.0,
    "very_frequent": 20.0,
}


def risk_reward_ratio(entry: float, stop: float, target: float) -> float:
    """Return reward / risk for a long position, or 0.0 when undefined."""
    risk = entry - stop
    reward = target - entry
    if risk <= 0:
        return 0.0
    return round(reward / risk, 4)


def calculate_risk_score(
    vol_regime: str = "normal",
    iv_percentile: float = 50.0,
    drawdown_risk: str = "low",
    gap_frequency: str = "rare",
    spread_pct: float = 0.0,
    max_loss_pct: float = 0.0,
    confidence: float = 0.5,
) -> dict[str, Any]:
    """Aggregate risk signals into a 0–100 score and a risk level.

    Base 50 (neutral), then:
        vol_regime       : low -10 / normal 0 / high +15 / very_high +25
        iv_percentile    : (percentile - 50) * 0.4
        drawdown_risk    : low 0 / moderate +10 / high +20 / extreme +30
        gap_frequency    : rare 0 / occasional +5 / frequent +12 / very_frequent +20
        spread_pct       : + spread * 300
        max_loss_pct     : + max_loss * 200
        confidence       : -(confidence - 0.5) * 40   (higher confidence lowers risk)

    Returns dict with ``risk_score``, ``risk_level`` and the components.
    """
    vol = _VOL_REGIME_SCORE.get(vol_regime, 0.0)
    dd = _DRAWDOWN_SCORE.get(drawdown_risk, 0.0)
    gap = _GAP_SCORE.get(gap_frequency, 0.0)
    iv = (iv_percentile - 50.0) * 0.4
    spread = spread_pct * 300.0
    ml = max_loss_pct * 200.0
    conf = -(confidence - 0.5) * 40.0

    score = 50.0 + vol + iv + dd + gap + spread + ml + conf
    score = max(0.0, min(100.0, score))

    if score < 25:
        level = "low"
    elif score < 50:
        level = "moderate"
    elif score < 75:
        level = "high"
    else:
        level = "very_high"

    return {
        "risk_score": round(score, 1),
        "risk_level": level,
        "components": {
            "base": 50.0,
            "vol_regime": round(vol, 2),
            "iv_percentile": round(iv, 2),
            "drawdown": round(dd, 2),
            "gap_frequency": round(gap, 2),
            "spread": round(spread, 2),
            "max_loss": round(ml, 2),
            "confidence": round(conf, 2),
        },
    }
