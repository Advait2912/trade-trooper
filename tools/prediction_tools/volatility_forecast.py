"""Phase 2 - Estimated forward volatility / IV proxy.

Reuses pre-computed volatility results from ``HistoricalAgentResult.volatility``
(produced by Phase 1 Historical Agent) and ``HistoricalAgentResult.historical_trends``.

IMPORTANT TERMINOLOGY
---------------------
``iv_forecast`` here is an **estimated forward volatility proxy**, NOT
market-observed implied volatility.  Real implied volatility requires an
options chain (bid/ask for specific strikes and expiries).  Phase 3 will
fetch the options chain and can replace this estimate with observed IV.

Estimation method:
    iv_estimate ≈ HV_20 × regime_multiplier

Regime multipliers (justified: contraction regimes compress realized vol;
expansion regimes are associated with elevated uncertainty and options premium):
    low_contraction :  0.90
    contraction     :  1.00
    expansion       :  1.20
    high_expansion  :  1.45

Vol regime bucket (from HV_20):
    < 15 %  → "low"
    < 30 %  → "normal"
    < 50 %  → "high"
    ≥ 50 %  → "very_high"
"""

from __future__ import annotations

from typing import Any

from tuning import DEFAULT_TUNING, TuningConfig

_VOL_BUCKETS = [
    (15.0, "low"),
    (30.0, "normal"),
    (50.0, "high"),
]


def _vol_regime_bucket(hv_pct: float) -> str:
    for threshold, label in _VOL_BUCKETS:
        if hv_pct < threshold:
            return label
    return "very_high"


def forecast_volatility(
    volatility: dict[str, Any],
    historical_trends: dict[str, Any],
    tuning: TuningConfig | None = None,
) -> dict[str, Any]:
    """Derive estimated forward volatility from Phase 1 volatility bundle.

    Parameters
    ----------
    volatility:
        ``HistoricalAgentResult.volatility`` — already contains:
        * ``calculate_historical_volatility`` → {historical_vol, vol_trend, vol_percentile}
        * ``detect_volatility_regimes``       → {regime, vol_acceleration, expected_duration_days}
        * ``analyze_mean_reversion``          → {mean_reversion_score, ...}
    historical_trends:
        ``HistoricalAgentResult.historical_trends`` — contains ``vol_percentile``
        and ``historical_vol`` as convenience keys.

    Returns
    -------
    dict with:
        iv_forecast         : float  — estimated forward vol (annualized %)
        iv_source           : "estimated"
        vol_regime          : str    — low | normal | high | very_high
        vol_percentile      : float  — 0–100
        hv_20               : float  — 20-day realized vol (annualized %)
        hv_60               : float  — 60-day realized vol (annualized %) if available
        mean_reversion_score: float
        vol_acceleration    : float
    """
    hv_bundle = volatility.get("calculate_historical_volatility") or {}
    regime_bundle = volatility.get("detect_volatility_regimes") or {}
    reversion_bundle = volatility.get("analyze_mean_reversion") or {}

    hv_20: float = float(hv_bundle.get("historical_vol") or 0.0)
    # hv_60 is not separately computed in Phase 1; use historical_trends fallback.
    hv_60: float = float(historical_trends.get("historical_vol") or hv_20)
    vol_percentile: float = float(hv_bundle.get("vol_percentile") or 0.0)

    regime_str: str = str(regime_bundle.get("regime") or "contraction")
    vol_acceleration: float = float(regime_bundle.get("vol_acceleration") or 0.0)
    mean_reversion_score: float = float(
        reversion_bundle.get("mean_reversion_score") or 0.0
    )

    multiplier = (tuning or DEFAULT_TUNING).regime_multipliers.get(regime_str, 1.00)
    iv_estimate = hv_20 * multiplier

    # Ensure we return a positive, finite value even with sparse data.
    if iv_estimate <= 0.0 or not _is_finite(iv_estimate):
        iv_estimate = max(hv_20, 10.0)  # 10 % floor — better than 0

    vol_regime = _vol_regime_bucket(hv_20)

    return {
        "iv_forecast": round(iv_estimate, 4),
        "iv_source": "estimated",
        "vol_regime": vol_regime,
        "vol_percentile": round(vol_percentile, 1),
        "hv_20": round(hv_20, 4),
        "hv_60": round(hv_60, 4),
        "mean_reversion_score": round(mean_reversion_score, 4),
        "vol_acceleration": round(vol_acceleration, 4),
    }


def _is_finite(x: float) -> bool:
    try:
        import math
        return math.isfinite(x)
    except (TypeError, ValueError):
        return False
