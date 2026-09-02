"""Phase 3 - Maximum loss tool.

Computes the worst-case dollar loss for an equity position:

1. Stop-based loss: ``(entry - stop)`` per share, inflated by historical gap
   slippage (a stop will not fill at the stop level through a gap).
2. Tail loss: historical Value-at-Risk / CVaR applied to the notional.

The option leg is not sized here — a long option's max loss is simply the
premium paid (defined risk), which ``position_size`` already reports.
"""

from __future__ import annotations

from typing import Any

_GAP_FREQUENCY_MULTIPLIER: dict[str, float] = {
    "rare": 1.0,
    "occasional": 1.1,
    "frequent": 1.25,
    "very_frequent": 1.5,
}


def calculate_max_loss(
    entry: float,
    stop: float,
    position_size: float,
    avg_gap_size: float = 0.0,
    gap_frequency: str = "rare",
    var_pct: float = 0.0,
    cvar_pct: float = 0.0,
) -> dict[str, Any]:
    """Calculate maximum loss for a sized equity position.

    ``avg_gap_size`` and ``var_pct``/``cvar_pct`` are percentages (e.g. 2.5 = 2.5 %),
    matching the Phase 1 ``analyze_gaps`` / ``calculate_value_at_risk`` outputs.

    Returns
    -------
    dict with per-share losses, max loss in dollars/percent, and tail VaR/CVaR.
    """
    if entry <= 0 or position_size <= 0:
        return {
            "base_loss_per_share": 0.0,
            "gap_slippage_per_share": 0.0,
            "effective_loss_per_share": 0.0,
            "max_loss_dollars": 0.0,
            "max_loss_pct": 0.0,
            "tail_var_dollars": 0.0,
            "tail_cvar_dollars": 0.0,
        }

    base_loss = max(0.0, entry - stop)
    gap_mult = _GAP_FREQUENCY_MULTIPLIER.get(gap_frequency, 1.0)
    gap_slippage = entry * (avg_gap_size / 100.0) * gap_mult
    effective_loss = base_loss + gap_slippage

    notional = position_size * entry
    max_loss_dollars = position_size * effective_loss
    max_loss_pct = effective_loss / entry

    tail_var_dollars = notional * (var_pct / 100.0)
    tail_cvar_dollars = notional * (cvar_pct / 100.0)

    return {
        "base_loss_per_share": round(base_loss, 4),
        "gap_slippage_per_share": round(gap_slippage, 4),
        "effective_loss_per_share": round(effective_loss, 4),
        "max_loss_dollars": round(max_loss_dollars, 2),
        "max_loss_pct": round(max_loss_pct, 6),
        "tail_var_dollars": round(tail_var_dollars, 2),
        "tail_cvar_dollars": round(tail_cvar_dollars, 2),
    }
