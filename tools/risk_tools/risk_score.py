"""Phase 3 - Composite risk score and reward:risk tools.

``calculate_risk_score`` folds every deterministic risk signal into a single
0–100 score (higher = riskier) so the risk level is a bucket of one auditable
number rather than a chain of branches.  ``risk_reward_ratio`` is the classic
reward / risk for the proposed entry/stop/target.

The composite is a *normalized weighted risk budget*: each factor is mapped to
a 0..1 sub-score and combined with fixed weights against a neutral base.  This
keeps the score spread across the whole 0..100 range (a calm stock scores low,
a genuinely adverse setup scores very_high) instead of saturating at 100 the
moment a couple of factors are elevated.  ``max_loss_pct`` is position-relative
(the stop distance as a fraction of entry), not capital-relative, and is
normalised accordingly.
"""

from __future__ import annotations

from typing import Any

# Sub-score (0..1) maps for the qualitative signals.  Lower is safer.
_VOL_REGIME: dict[str, float] = {"low": 0.05, "normal": 0.35, "high": 0.65, "very_high": 0.95}
_DRAWDOWN: dict[str, float] = {"low": 0.10, "moderate": 0.40, "high": 0.70, "extreme": 0.95}
_GAP: dict[str, float] = {"rare": 0.10, "occasional": 0.30, "frequent": 0.60, "very_frequent": 0.85}

# Relative importance of each factor (sums to 1.0 with the base).
_FACTOR_WEIGHTS: dict[str, float] = {
    "vol": 0.22,
    "drawdown": 0.16,
    "gap": 0.12,
    "spread": 0.10,
    "max_loss": 0.20,
    "confidence": 0.12,
}
_BASE_WEIGHT = 0.08
_MAX_LOSS_SPAN = 12.0  # % of entry mapped to 0..1 (2% -> 0, 14% -> 1)


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


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

    Each deterministic signal is mapped to a 0..1 sub-score and combined by
    fixed weight against a neutral ``base`` (score = 100 * (base + Σ w_i·f_i)).
    Sub-scores:

        vol_regime     : low 0.05 / normal 0.35 / high 0.65 / very_high 0.95
        iv_percentile  : folds into the vol sub-score (percentile/100)
        drawdown_risk  : low 0.10 / moderate 0.40 / high 0.70 / extreme 0.95
        gap_frequency  : rare 0.10 / occasional 0.30 / frequent 0.60 / very_frequent 0.85
        spread_pct     : spread / 0.15 (10% + spread is fully risky)
        max_loss_pct   : position stop distance % mapped over 2%..14%
        confidence     : 1 - confidence (low confidence -> higher risk)

    Returns dict with ``risk_score``, ``risk_level`` and the ``components``.
    """
    vol_sub = _clamp01(0.6 * _VOL_REGIME.get(vol_regime, 0.35) + 0.4 * (iv_percentile / 100.0))
    dd_sub = _DRAWDOWN.get(drawdown_risk, 0.10)
    gap_sub = _GAP.get(gap_frequency, 0.10)
    spread_sub = _clamp01(spread_pct / 0.15)
    max_loss_sub = _clamp01((max_loss_pct * 100.0 - 2.0) / _MAX_LOSS_SPAN)
    conf_sub = _clamp01(1.0 - max(0.0, min(1.0, confidence)))

    weighted = (
        _FACTOR_WEIGHTS["vol"] * vol_sub
        + _FACTOR_WEIGHTS["drawdown"] * dd_sub
        + _FACTOR_WEIGHTS["gap"] * gap_sub
        + _FACTOR_WEIGHTS["spread"] * spread_sub
        + _FACTOR_WEIGHTS["max_loss"] * max_loss_sub
        + _FACTOR_WEIGHTS["confidence"] * conf_sub
    )
    total_weight = sum(_FACTOR_WEIGHTS.values()) + _BASE_WEIGHT
    score = 100.0 * (_BASE_WEIGHT + weighted) / total_weight
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
            "base": round(_BASE_WEIGHT * 100.0 / total_weight, 2),
            "vol": round(vol_sub, 4),
            "drawdown": round(dd_sub, 4),
            "gap": round(gap_sub, 4),
            "spread": round(spread_sub, 4),
            "max_loss": round(max_loss_sub, 4),
            "confidence": round(conf_sub, 4),
        },
    }
