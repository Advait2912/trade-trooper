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

    ind_cols = st.columns(4)
    chosen = []
    for i, ind in enumerate(sorted(INDUSTRY_STOCKS)):
        if ind_cols[i % 4].checkbox(ind, key=f"run_ind_{ind}"):
            chosen.append(ind)
    defaults = sorted({t for ind in chosen for t in INDUSTRY_STOCKS[ind]}) if chosen else \
        [t for t in current if t in all_tickers]

    selected = st.multiselect(
        "Universe (tickers the agent trades)",
        options=all_tickers,
        default=defaults,
        help="Pick by industry above, then fine-tune here. Industry/stock weights from "
             "data/weights_db.json are applied automatically per ticker.",
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

    st.divider()
    _risk_params()


def _risk_params() -> None:
    section("🔧", "Granular Risk Parameters",
            "Fine-grained sizing / gates / limits written to .env.")
    env = read_env()

    def num(key: str, default: float, fmt="%.3f") -> float:
        try:
            return float(env.get(key, ""))
        except ValueError:
            return default

    c1, c2, c3 = st.columns(3)
    with c1:
        risk_pct = st.number_input("Risk per trade (%)", 0.0, 10.0,
                                   num("RISK_PER_TRADE_PCT", 0.01) * 100, 0.1,
                                   help="Fraction of capital risked per trade")
        max_pos = st.number_input("Max position (%)", 0.0, 100.0,
                                  num("MAX_POSITION_PCT", 0.05) * 100, 1.0)
        min_conf = st.number_input("Min confidence", 0.0, 0.95,
                                   num("MIN_CONFIDENCE", 0.35), 0.01)
    with c2:
        max_open = st.number_input("Max open positions (portfolio)", 1, 20,
                                   int(num("MAX_OPEN_POSITIONS", 3, "%.0f")))
        max_open_t = st.number_input("Max open positions / ticker", 1, 10,
                                     int(num("MAX_OPEN_POSITIONS_PER_TICKER", 1, "%.0f")))
        min_rr = st.number_input("Min reward/risk", 0.0, 3.0,
                                 num("MIN_RISK_REWARD", 1.0), 0.05)
    with c3:
        daily_loss = st.number_input("Daily loss limit (%)", 0.0, 20.0,
                                     num("DAILY_LOSS_LIMIT_PCT", 0.02) * 100, 0.5)
        dd_cap = st.number_input("Max portfolio drawdown (%)", 0.0, 50.0,
                                 num("MAX_PORTFOLIO_DRAWDOWN_PCT", 0.10) * 100, 1.0)
        horizon = st.number_input("Trade horizon (days)", 1, 30,
                                  int(num("TRADE_HORIZON_DAYS", 5, "%.0f")))

    c1b, c2b = st.columns(2)
    with c1b:
        interval = st.number_input("Cycle interval (min)", 1, 240,
                                   int(num("TRADING_INTERVAL_MIN", 30, "%.0f")))
    with c2b:
        order_type = st.selectbox("Order type", ["limit", "market"],
                                  index=0 if env.get("ORDER_TYPE", "limit") == "limit" else 1)

    if st.button("💾 Save risk parameters", type="primary"):
        write_env({
            "RISK_PER_TRADE_PCT": f"{risk_pct / 100:.4f}",
            "MAX_POSITION_PCT": f"{max_pos / 100:.4f}",
            "MIN_CONFIDENCE": f"{min_conf:.2f}",
            "MIN_RISK_REWARD": f"{min_rr:.2f}",
            "MAX_OPEN_POSITIONS": str(int(max_open)),
            "MAX_OPEN_POSITIONS_PER_TICKER": str(int(max_open_t)),
            "DAILY_LOSS_LIMIT_PCT": f"{daily_loss / 100:.4f}",
            "MAX_PORTFOLIO_DRAWDOWN_PCT": f"{dd_cap / 100:.4f}",
            "TRADE_HORIZON_DAYS": str(int(horizon)),
            "TRADING_INTERVAL_MIN": str(int(interval)),
            "ORDER_TYPE": order_type,
        })
        st.success("Risk parameters written to .env — applied on the next cycle.")


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
