"""Phase 3 - Risk tools.

    calculate_greeks(chain, spot, iv_forecast, horizon_days, risk_free_rate) -> dict
    calculate_position_size(capital, risk_per_trade_pct, entry, stop, ...) -> dict
    calculate_max_loss(entry, stop, position_size, ...) -> dict
    calculate_risk_score(vol_regime, iv_percentile, ...) -> dict
    risk_reward_ratio(entry, stop, target) -> float
"""

from tools.risk_tools.greeks import calculate_greeks, parse_occ_symbol
from tools.risk_tools.max_loss import calculate_max_loss
from tools.risk_tools.position_size import calculate_position_size
from tools.risk_tools.risk_score import calculate_risk_score, risk_reward_ratio

__all__ = [
    "calculate_greeks",
    "parse_occ_symbol",
    "calculate_position_size",
    "calculate_max_loss",
    "calculate_risk_score",
    "risk_reward_ratio",
]
