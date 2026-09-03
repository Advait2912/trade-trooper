"""Central, externalized tuning knobs for the deterministic pipeline.

Every numeric weight / threshold in Phases 2-4 lives here as a single
``TuningConfig`` so the tuning harness (``scripts/tune.py``) can sweep them
without editing the tool implementations.  The defaults are exactly the
constants that were previously inlined in the tool modules, so a default
``TuningConfig`` reproduces the current behaviour bit-for-bit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class TuningConfig:
    """All tunable knobs, grouped by phase.

    Fields that are dictionaries (weights, sub-score maps) are mutable per
    instance via ``field(default_factory=...)``.
    """

    # ------------------------------------------------------------------
    # Phase 2 — momentum signal (tools/prediction_tools/technical.py)
    # ------------------------------------------------------------------
    momentum_weights: dict[str, float] = field(default_factory=lambda: {
        "macd": 0.25,
        "adx": 0.20,
        "rsi": 0.20,
        "bollinger": 0.15,
        "obv": 0.10,
        "stochastic": 0.10,
    })
    adx_strength_scale: dict[str, float] = field(default_factory=lambda: {
        "no_trend": 0.0,
        "weak": 0.3,
        "moderate": 0.7,
        "strong": 1.0,
        "very_strong": 1.0,
    })
    news_weight: float = 0.20  # max momentum shift from news (±)

    # ------------------------------------------------------------------
    # Phase 2 — price move / confidence (prediction_tools/price_move.py)
    # ------------------------------------------------------------------
    direction_dampen: float = 0.50
    conf_base: float = 0.50
    conf_min: float = 0.10
    conf_max: float = 0.90
    conf_adx_bonus: float = 0.15
    conf_normal_vol_bonus: float = 0.10
    conf_momentum_bonus: float = 0.10
    conf_high_vol_penalty: float = 0.10
    conf_reversion_penalty: float = 0.15

    # ------------------------------------------------------------------
    # Phase 2 — vol forecast (prediction_tools/volatility_forecast.py)
    # ------------------------------------------------------------------
    regime_multipliers: dict[str, float] = field(default_factory=lambda: {
        "low_contraction": 0.90,
        "contraction": 1.00,
        "expansion": 1.20,
        "high_expansion": 1.45,
    })

    # ------------------------------------------------------------------
    # Phase 3 — risk score (tools/risk_tools/risk_score.py)
    # ------------------------------------------------------------------
    factor_weights: dict[str, float] = field(default_factory=lambda: {
        "vol": 0.22,
        "drawdown": 0.16,
        "gap": 0.12,
        "spread": 0.10,
        "max_loss": 0.20,
        "confidence": 0.12,
    })
    base_weight: float = 0.08
    max_loss_span: float = 12.0  # % of entry mapped over 2%..14%
    vol_regime_subs: dict[str, float] = field(default_factory=lambda: {
        "low": 0.05,
        "normal": 0.35,
        "high": 0.65,
        "very_high": 0.95,
    })
    drawdown_subs: dict[str, float] = field(default_factory=lambda: {
        "low": 0.10,
        "moderate": 0.40,
        "high": 0.70,
        "extreme": 0.95,
    })
    gap_subs: dict[str, float] = field(default_factory=lambda: {
        "rare": 0.10,
        "occasional": 0.30,
        "frequent": 0.60,
        "very_frequent": 0.85,
    })

    # ------------------------------------------------------------------
    # Phase 3 — position sizing (tools/risk_tools/position_size.py)
    # ------------------------------------------------------------------
    drawdown_factor: dict[str, float] = field(default_factory=lambda: {
        "low": 1.0,
        "moderate": 0.8,
        "high": 0.6,
        "extreme": 0.4,
    })
    conf_floor: float = 0.25        # confidence sizing floor
    iv_quality_floor: float = 0.50  # IV-quality sizing floor
    spread_factor_slope: float = 6.0
    spread_factor_floor: float = 0.30

    # ------------------------------------------------------------------
    # Phase 3 — max loss (tools/risk_tools/max_loss.py)
    # ------------------------------------------------------------------
    gap_frequency_multiplier: dict[str, float] = field(default_factory=lambda: {
        "rare": 1.0,
        "occasional": 1.1,
        "frequent": 1.25,
        "very_frequent": 1.5,
    })

    # ------------------------------------------------------------------
    # Phase 4 — signal synthesis (tools/decision_tools/signals.py)
    # ------------------------------------------------------------------
    signal_weights: dict[str, float] = field(default_factory=lambda: {
        "news_sentiment": 0.20,
        "technical_summary": 0.25,
        "historical_trend": 0.10,
        "prediction_signal": 0.30,
        "market_trend": 0.15,
    })

    # ------------------------------------------------------------------
    # Phase 4 — opportunity ranking (tools/decision_tools/ranking.py)
    # ------------------------------------------------------------------
    component_max: dict[str, float] = field(default_factory=lambda: {
        "signal": 40.0,
        "reward_risk": 25.0,
        "risk_quality": 20.0,
        "execution": 15.0,
    })
    rr_full_at: float = 3.0           # R:R that earns the full reward/risk band
    spread_full_at: float = 0.15      # spread fraction that zeroes execution
    execution_fallback_mult: float = 0.6  # greeks_source fallback multiplier

    # ------------------------------------------------------------------
    # Phase 4 — instrument preference (long options carry theta + spread)
    # ------------------------------------------------------------------
    equity_only: bool = False         # True -> only build the long-equity candidate
    equity_score_boost: float = 0.0   # added to equity candidates' opportunity score

    def to_dict(self) -> dict:
        """Return a flat dict of all knobs (for serialization / display)."""
        return asdict(self)

    @classmethod
    def from_overrides(cls, overrides: dict) -> "TuningConfig":
        """Build a config from a dict of (possibly partial) overrides.

        Unknown keys raise ``ValueError`` so typos in a sweep grid surface
        immediately instead of silently being ignored.
        """
        cfg = cls()
        for key, value in overrides.items():
            if not hasattr(cfg, key):
                raise ValueError(f"Unknown tuning key: {key!r}")
            setattr(cfg, key, value)
        return cfg


# Convenience default used by tool functions when no tuning is supplied.
DEFAULT_TUNING = TuningConfig()


# Named preset overrides, usable via ``scripts/tune.py --preset NAME``.  Values
# may be TuningConfig fields or Settings fields (gates/sizing); the harness
# routes them accordingly.
PRESETS: dict[str, dict] = {
    "default": {},
    "equity_only": {"equity_only": True},
    "conservative": {"min_confidence": 0.50, "min_risk_reward": 1.25},
    "aggressive": {"min_confidence": 0.30, "min_risk_reward": 0.75},
    "signal_prediction_led": {
        "signal_weights": {
            "news_sentiment": 0.10,
            "technical_summary": 0.20,
            "historical_trend": 0.10,
            "prediction_signal": 0.50,
            "market_trend": 0.10,
        },
    },
    "signal_technical_led": {
        "signal_weights": {
            "news_sentiment": 0.10,
            "technical_summary": 0.40,
            "historical_trend": 0.10,
            "prediction_signal": 0.25,
            "market_trend": 0.15,
        },
    },
    "tuned": {
        "equity_only": True,
        "min_confidence": 0.3589769763094654,
        "min_risk_reward": 1.2448794246808945,
        "trade_horizon_days": 3,
        "momentum_weights": {
            "macd": 0.135791,
            "adx": 0.242160,
            "rsi": 0.202523,
            "bollinger": 0.121021,
            "obv": 0.233844,
            "stochastic": 0.064660,
        },
        "signal_weights": {
            "news_sentiment": 0.253104,
            "technical_summary": 0.182606,
            "historical_trend": 0.192505,
            "prediction_signal": 0.258161,
            "market_trend": 0.113623,
        },
    },
}
