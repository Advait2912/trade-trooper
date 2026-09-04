"""Tuning page — per-industry weights viewer."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from web.ui import charts
from web.ui.data import load_weights_db
from web.ui.theme import kpi_tile, section


def render() -> None:
    section("🎯", "Industry Weights Database",
            "One Optuna-tuned config per industry (news-aware backtests, 15 trials/industry).")

    db = load_weights_db()
    if not db:
        st.warning("data/weights_db_tuned_backup.json not found.")
        return

    industries = [k for k in db if k not in ("default", "tickers")]
    selected = st.selectbox("Industry", ["default"] + sorted(industries) + ["tickers"])
    cfg = db.get(selected, {})

    if selected == "tickers":
        ticker = st.selectbox("Ticker override", sorted(cfg.keys()) if cfg else ["—"])
        cfg = cfg.get(ticker, {})
        st.write(f"Stock-specific config for **{ticker}**")
    else:
        st.write(f"Config for **{selected}**")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        kpi_tile("Industries tuned", str(len(industries)))
    with col_b:
        kpi_tile("Trials/industry", "15")
    with col_c:
        kpi_tile("News-aware", "✓ FinBERT cache")

    if cfg:
        knobs = {k: v for k, v in cfg.items() if not isinstance(v, dict)}

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Knobs**")
            st.json(knobs)
        with col2:
            st.markdown("**Signal weights (radar)**")
            st.plotly_chart(charts.signal_vote_radar(cfg), width="stretch")

        st.plotly_chart(charts.momentum_weights_bar(cfg), width="stretch")

        if selected != "default" and "default" in db:
            default_cfg = db["default"]
            st.divider()
            st.markdown(f"**Side-by-side: `{selected}` vs `default`**")
            compare = pd.DataFrame(
                {
                    selected: {k: v for k, v in cfg.items() if not isinstance(v, dict)},
                    "default": {k: v for k, v in default_cfg.items() if not isinstance(v, dict)},
                }
            ).dropna(how="all")
            st.dataframe(compare, width="stretch")
    else:
        st.info("No config for this entry.")

    st.divider()
    st.subheader("Available presets (tuning.py)")
    from tuning import PRESETS

    st.code(json.dumps({k: v for k, v in PRESETS.items() if k in ("conservative", "balanced", "aggressive")}, indent=2))
