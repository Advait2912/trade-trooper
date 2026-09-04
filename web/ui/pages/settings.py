"""Settings page — API credentials, runner controls, risk profile, market clock."""

from __future__ import annotations

import subprocess

import streamlit as st

from web.ui.data import read_env, write_env
from web.ui.runner_control import DEFAULT_UNIVERSE, restart, start, status, stop
from web.ui.theme import section, status_pill

RISK_PROFILES = {
    "Conservative": {"min_confidence": 0.50, "min_risk_reward": 1.25, "equity_only": True, "trade_horizon_days": 7},
    "Balanced": {"min_confidence": 0.40, "min_risk_reward": 1.00, "equity_only": False, "trade_horizon_days": 5},
    "Aggressive": {"min_confidence": 0.30, "min_risk_reward": 0.75, "equity_only": False, "trade_horizon_days": 3},
}


def _runner_controls() -> None:
    section("🚀", "Runner Controls",
            "Start / stop / restart the paper-trading loop. 'Apply & Restart' swaps the universe.")
    rs = status()
    if rs["running"]:
        status_pill(f"🟢 RUNNER LIVE — PID {rs['pid']} · {len(rs['universe'])} tickers", "running")
    else:
        status_pill("🟡 RUNNER STOPPED", "stopped")

    if rs["last_log_lines"]:
        with st.expander("Last runner log lines"):
            st.code("\n".join(rs["last_log_lines"]))

    from weights_db import INDUSTRY_STOCKS

    all_tickers = sorted({t for ts in INDUSTRY_STOCKS.values() for t in ts})
    current = rs["universe"] or DEFAULT_UNIVERSE
    selected = st.multiselect(
        "Universe (tickers the agent trades)",
        options=all_tickers,
        default=[t for t in current if t in all_tickers],
        help="Change the set of stocks the runner analyses and trades. Applies on restart.",
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("▶ Start", type="primary", disabled=rs["running"]):
            ok, msg = start(selected or DEFAULT_UNIVERSE)
            st.success(msg) if ok else st.error(msg)
    with col2:
        if st.button("⏹ Stop", disabled=not rs["running"]):
            ok, msg = stop()
            st.success(msg) if ok else st.error(msg)
    with col3:
        if st.button("🔄 Apply & Restart"):
            ok, msg = restart(selected or DEFAULT_UNIVERSE)
            st.success(msg) if ok else st.error(msg)

    st.caption("Stop is allowed even when the market is closed — the loop simply idles.")


def _risk_profile() -> None:
    section("⚖️", "Risk Profile",
            "Adjusts entry gates and instrument selection. Takes effect on the next cycle.")
    env = read_env()
    current = env.get("RISK_PROFILE", "Balanced")
    if current not in RISK_PROFILES:
        current = "Balanced"

    profile = st.selectbox("Active risk profile", list(RISK_PROFILES),
                           index=list(RISK_PROFILES).index(current))
    st.json(RISK_PROFILES[profile])

    if st.button("Apply profile", type="primary"):
        patch = {
            "RISK_PROFILE": profile,
            "MIN_CONFIDENCE": str(RISK_PROFILES[profile]["min_confidence"]),
            "MIN_RISK_REWARD": str(RISK_PROFILES[profile]["min_risk_reward"]),
            "TRADE_HORIZON_DAYS": str(RISK_PROFILES[profile]["trade_horizon_days"]),
        }
        write_env(patch)
        st.success(f"Profile '{profile}' written to .env — next cycle picks it up.")


def _market_status() -> None:
    section("🕒", "Market Status")
    try:
        result = subprocess.run(["alpaca", "clock"], capture_output=True, text=True, timeout=15)
        st.code(result.stdout[:1200])
    except Exception as exc:  # noqa: BLE001
        st.write(f"Alpaca CLI unavailable: {exc}")


def render() -> None:
    st.header("Settings")
    st.caption("Paper-trading only. Live (AK) keys are rejected everywhere. "
               "API credentials live in the 🔑 API tab.")

    _risk_profile()

    st.divider()
    _market_status()

    st.divider()
    env = read_env()
    safe_env = {k: ("***" if "SECRET" in k or "KEY" in k else v) for k, v in env.items()}
    st.subheader("Trading configuration (.env)")
    st.json(safe_env)
