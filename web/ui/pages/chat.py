"""Chat page — talk to the local Ollama model about decisions/trades.

Conversation lives in ``st.session_state``. A decision trace (market context,
prediction, risk, decision, realized outcome) is attached as context when you
pick a trade from the journals.
"""

from __future__ import annotations

import json

import streamlit as st

from web.ui.data import load_cycles, load_trades
from web.ui.runner_control import status as runner_status
from web.ui.theme import section
from web.ui.trace import chat_about_trade, decision_trace_text


def _ensure_state() -> None:
    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []


def _pick_trade_context() -> None:
    trades = load_trades()
    if trades.empty:
        st.caption("No trades in journals yet — run a backtest or let the paper loop trade.")
        return
    options = trades.apply(
        lambda r: f"{r['ticker']}  {str(r['opened_ts'])[:10]}  pnl=${r['pnl']:.0f}", axis=1
    ).tolist()
    choice = st.selectbox("Attach a trade's decision trace", ["— none —"] + options)
    if choice == "— none —":
        return
    row = trades.iloc[options.index(choice)]
    cycles = load_cycles()
    snapshot = {}
    if not cycles.empty:
        sub = cycles[cycles["ticker"] == row["ticker"]].copy()
        if not sub.empty:
            sub["ts"] = pd_to_datetime(sub["ts"])
            sub["dist"] = (sub["ts"] - pd_to_datetime(row["opened_ts"])).abs()
            best = sub.loc[sub["dist"].idxmin()]
            try:
                snapshot = json.loads(best["snapshot"])
            except (ValueError, TypeError):
                snapshot = {}
    include_logs = st.checkbox("Include recent execution logs in context", value=True)
    logs = None
    if include_logs:
        logs = runner_status().get("last_log_lines") or []
    trace = decision_trace_text(snapshot, row.to_dict(), logs=logs)
    if st.button("📎 Attach trace to next message"):
        st.session_state["chat_messages"].append(
            {"role": "user",
             "content": f"Here is a decision trace with execution context:\n{trace}\n\nPlease analyze it and explain the decision and logs."}
        )
        st.rerun()


def _render_chat() -> None:
    for msg in st.session_state["chat_messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


def render() -> None:
    section("💬", "Agent Reasoning Chat",
            "Uses your local Ollama model to reason over decision traces and answer questions.")
    _ensure_state()
    _pick_trade_context()

    _render_chat()

    prompt = st.chat_input("Ask about a decision, the strategy, or a trade…")
    if prompt:
        st.session_state["chat_messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Reasoning…"):
                reply = chat_about_trade(
                    "No decision trace attached — answer the question directly.",
                    question=prompt,
                    history=st.session_state["chat_messages"],
                )
            st.markdown(reply)
        st.session_state["chat_messages"].append({"role": "assistant", "content": reply})

    if st.session_state["chat_messages"]:
        if st.button("🧹 Clear conversation"):
            st.session_state["chat_messages"] = []
            st.rerun()


def pd_to_datetime(x):
    import pandas as pd

    return pd.to_datetime(x)
