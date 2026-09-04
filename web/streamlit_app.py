"""Trade-Trooper Streamlit dashboard — thin entry.

Run:  streamlit run web/streamlit_app.py --server.port 8501

Tabs:
  🔑 API       — enter Alpaca paper credentials
  🚀 Runner    — start/stop/restart the paper-trading loop
  📊 Live      — equity, drawdown, decisions, LLM narrative
  📈 Backtest  — on-demand backtest + charts
  🎯 Tuning    — per-industry weights viewer
  ⚙️ Settings  — risk profile, market clock, .env display
"""

from __future__ import annotations

import sys
from pathlib import Path

# Streamlit runs this file as a script, so the repo root is not on sys.path
# by default — add it so `web.ui.*` imports resolve regardless of CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from web.ui.theme import inject_theme_css, section, status_pill

st.set_page_config(page_title="Trade-Trooper", page_icon="📈", layout="wide")
inject_theme_css()

st.title("📈 Trade-Trooper — Autonomous Alpaca Paper-Trading Agent")

env = {}
from web.ui.data import read_env  # noqa: E402

env = read_env()
if env.get("ALPACA_API_KEY", "").startswith("AK"):
    status_pill("🚨 LIVE KEY DETECTED — paper-only app. Fix in API tab.", "error")
elif env.get("ALPACA_API_KEY"):
    status_pill("🛡 PAPER ONLY — connected", "running")
else:
    status_pill("⚠ No API key yet — enter one in the API tab", "warning")

tab_api, tab_runner, tab_live, tab_backtest, tab_tuning, tab_settings = st.tabs(
    ["🔑 API", "🚀 Runner", "📊 Live", "📈 Backtest", "🎯 Tuning", "⚙️ Settings"]
)

with tab_api:
    from web.ui import api_keys

    section("🔑", "API Credentials", "Enter your Alpaca paper key and secret to begin.")
    api_keys.credentials_form()

with tab_runner:
    from web.ui.pages import settings as settings_page

    settings_page._runner_controls()

with tab_live:
    from web.ui.pages import live as live_page

    live_page.render()
    live_page.auto_refresh()

with tab_backtest:
    from web.ui.pages import backtest as backtest_page

    backtest_page.render()

with tab_tuning:
    from web.ui.pages import tuning as tuning_page

    tuning_page.render()

with tab_settings:
    from web.ui.pages import settings as settings_page

    settings_page.render()
