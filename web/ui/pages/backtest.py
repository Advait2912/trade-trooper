"""Backtest page — run on-demand backtests and visualize results."""

from __future__ import annotations

import subprocess
import sys

import streamlit as st

from web.ui import charts
from web.ui.data import cache_db, load_cycles, load_trades, repo_root
from web.ui.theme import kpi_tile, section


def _run_backtest(tickers: list[str], months: int) -> str:
    cmd = [sys.executable, "-m", "scripts.tune", "evaluate",
           "--universe", ",".join(tickers), "--months", str(months)]
    if cache_db().exists():
        cmd += ["--news-cache", str(cache_db())]
    try:
        result = subprocess.run(cmd, cwd=str(repo_root()), capture_output=True, text=True, timeout=900)
        return result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return "Backtest timed out after 15 minutes."


def render() -> None:
    section("📈", "Backtest & Tuning Evaluation",
            "Replays the deterministic Phase 2–4 chain over historical bars (news-aware when the cache exists).")

    from weights_db import INDUSTRY_STOCKS

    options = sorted({t for ts in INDUSTRY_STOCKS.values() for t in ts})
    tickers = st.multiselect("Tickers", options=options, default=["NVDA", "AAPL", "JPM", "XOM"])
    months = st.slider("Months of history", 3, 24, 12, step=3)

    if st.button("▶ Run backtest", type="primary"):
        if not tickers:
            st.error("Select at least one ticker.")
        else:
            with st.spinner("Running backtest — this can take a few minutes..."):
                output = _run_backtest(tickers, months)
            st.text(output[-4000:])

    st.divider()
    trades = load_trades()
    cycles = load_cycles()

    if not trades.empty:
        col1, col2, col3, col4 = st.columns(4)
        wins = trades[trades["pnl"] > 0]
        losses = trades[trades["pnl"] <= 0]
        gross_win = wins["pnl"].sum()
        gross_loss = abs(losses["pnl"].sum())
        pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
        with col1:
            kpi_tile("Win rate", f"{len(wins) / len(trades) * 100:.1f}%")
        with col2:
            kpi_tile("Profit factor", f"{pf:.2f}" if pf != float("inf") else "∞")
        with col3:
            kpi_tile("Expectancy", f"${trades['pnl'].mean():,.2f}")
        with col4:
            kpi_tile("Max drawdown", f"${trades['pnl'].cumsum().min():,.0f}")

        st.plotly_chart(charts.cumulative_pnl(trades), width="stretch")
        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(charts.pnl_bar(trades), width="stretch")
        with col_b:
            st.plotly_chart(charts.winloss_heatmap(trades), width="stretch")

        st.dataframe(trades.sort_values("closed_ts", ascending=False).head(50), width="stretch")
    else:
        st.info("No trades recorded yet — run the backtest above or let the paper loop trade.")

    if not cycles.empty:
        st.plotly_chart(charts.decision_donut(cycles), width="stretch")
