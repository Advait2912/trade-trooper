"""Tuning page — launch tuning jobs, edit weights, manage checkpoints.

- Launch ``optimize-industries`` / ``optimize-stocks`` as async jobs.
- Review a finished job's weights and *Apply* them to the live DB (auto
  checkpoint first) — nothing writes to the live DB without your say-so.
- Edit any industry/ticker entry inline and save.
- Snapshot / restore / delete weights-DB checkpoints under ``data/checkpoints``.
"""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from web.ui import jobs
from web.ui.data import venv_python, weights_db_path
from web.ui.job_widgets import jobs_auto_refresh, render_job
from web.ui.theme import section
from weights_db import (
    INDUSTRY_STOCKS,
    TICKER_NAMESPACE,
    delete_checkpoint,
    list_checkpoints,
    load_weights_db,
    restore_checkpoint,
    save_weights_db,
    snapshot_weights_db,
)

_KNOB_FIELDS = ("min_confidence", "min_risk_reward", "trade_horizon_days")


def _tune_cmd(mode: str, selection: list[str], months: int, n_trials: int,
              min_trades: int, start: str, end: str, weights_out: str,
              progress_file: str) -> list[str]:
    cmd = [venv_python(), "-m", "scripts.tune"]
    if mode == "industry":
        cmd += ["optimize-industries", "--weights-db", weights_out,
                "--industries", ",".join(selection)]
    else:
        cmd += ["optimize-stocks", "--weights-db", weights_out,
                "--tickers", ",".join(selection)]
    cmd += ["--months", str(months), "--n-trials", str(n_trials), "--min-trades", str(min_trades),
            "--progress-file", progress_file]
    if start and end:
        cmd += ["--start-date", start, "--end-date", end]
    return cmd


def _launch_form() -> None:
    section("🎛️", "Launch Tuning Job",
            "Runs as a background subprocess — backtests and other tunes keep working meanwhile.")
    col_mode, col_months, col_trials, col_min = st.columns(4)
    with col_mode:
        mode = st.radio("Scope", ["industry", "ticker"], horizontal=True, key="tune_scope")
    with col_months:
        months = st.slider("Months", 3, 24, 12, step=3, key="tune_months")
    with col_trials:
        n_trials = st.number_input("Optuna trials", 5, 300, 50, step=5, key="tune_trials")
    with col_min:
        min_trades = st.number_input("Min trades", 1, 100, 5 if mode == "ticker" else 30, step=1,
                                     key="tune_min_trades")

    if mode == "industry":
        selection = st.multiselect("Industries", sorted(INDUSTRY_STOCKS),
                                   default=["Technology", "Financials"], key="tune_industries")
    else:
        all_tickers = sorted({t for ts in INDUSTRY_STOCKS.values() for t in ts})
        selection = st.multiselect("Tickers", all_tickers, default=["NVDA"], key="tune_tickers")

    use_range = st.checkbox("Use historical date range (train/test)", value=False, key="tune_use_range")
    start, end = "", ""
    if use_range:
        c1, c2 = st.columns(2)
        with c1:
            start = st.date_input("Start date", key="tune_start_date").isoformat()
        with c2:
            end = st.date_input("End date", key="tune_end_date").isoformat()

    if st.button("▶ Launch", type="primary", disabled=not selection):
        job_id = jobs.create("tune", label=f"{mode}: {','.join(selection[:3])}")
        weights_out = str(jobs.job_artifact(job_id, "weights.json"))
        progress_file = str(jobs.progress_file(job_id))
        cmd = _tune_cmd(mode, selection, months, n_trials, min_trades, start, end,
                        weights_out, progress_file)
        jobs.launch(job_id, cmd)
        st.success(f"Job {job_id} launched. It writes to a job-scoped weights file — "
                   "review it below and click Apply to promote it to the live DB.")
        st.rerun()


def _apply_ready(job: dict) -> None:
    if job.get("status") != "done":
        return
    weights_out = jobs.job_artifact(job["id"], "weights.json")
    if not weights_out.exists():
        return
    if st.session_state.get(f"applied_{job['id']}"):
        st.success(f"Job {job['id']} already applied.")
        return
    if st.button(f"⬆ Apply weights from {job['id']}", key=f"apply_{job['id']}"):
        try:
            job_db = json.loads(weights_out.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            st.error(f"Cannot read job weights: {exc}")
            return
        if not job_db:
            st.warning("Job produced no weights.")
            return
        live = load_weights_db(weights_db_path())
        snapshot_weights_db(weights_db_path(), label=f"before_apply_{job['id']}")
        changed = []
        for key, value in job_db.items():
            if key == TICKER_NAMESPACE:
                live.setdefault(TICKER_NAMESPACE, {}).update(value)
                changed.extend(f"tickers.{t}" for t in value)
            else:
                live[key] = value
                changed.append(key)
        save_weights_db(weights_db_path(), live)
        st.session_state[f"applied_{job['id']}"] = True
        st.success(f"Applied {len(changed)} entries: {', '.join(changed)} (checkpoint saved).")
        st.rerun()


def _tune_jobs() -> None:
    section("🧪", "Tuning Jobs", "Progress is parsed from the job log.")
    for job in jobs.list_jobs("tune")[:6]:
        render_job(job)
        _apply_ready(job)
        st.divider()


def _edit_weights_dict(weights: dict, widget_key: str) -> dict:
    if not weights:
        return {}
    df = pd.DataFrame({"key": list(weights), "value": list(weights.values())})
    edited = st.data_editor(df, key=widget_key, num_rows="fixed", width="stretch")
    out: dict = {}
    for _, row in edited.iterrows():
        try:
            out[str(row["key"])] = float(row["value"])
        except (TypeError, ValueError):
            continue
    return out


def _weights_editor() -> None:
    section("✏️", "Edit Weights", "Edits are saved atomically to data/weights_db.json.")
    db = load_weights_db(weights_db_path())
    if not db:
        st.warning("weights_db.json not found — run a tuning job first.")
        return

    industries = [k for k in db if k not in ("default", TICKER_NAMESPACE)]
    options = ["default"] + sorted(industries)
    if db.get(TICKER_NAMESPACE):
        options += ["tickers ▸ " + t for t in sorted(db[TICKER_NAMESPACE])]

    selected = st.selectbox("Entry", options)
    if selected.startswith("tickers ▸ "):
        key, ticker = TICKER_NAMESPACE, selected.split("▸ ", 1)[1]
        cfg = dict(db.get(TICKER_NAMESPACE, {}).get(ticker, {}))
    else:
        key, ticker = selected, None
        cfg = dict(db.get(selected, {}))

    if not cfg:
        st.info("Empty entry — the config below builds a new one.")
        cfg = {}

    with st.form(f"edit_{selected}"):
        edits: dict = {}
        col1, col2, col3 = st.columns(3)
        with col1:
            edits["min_confidence"] = st.number_input(
                "min_confidence", 0.0, 0.9, float(cfg.get("min_confidence", 0.35)), 0.01)
        with col2:
            edits["min_risk_reward"] = st.number_input(
                "min_risk_reward", 0.0, 3.0, float(cfg.get("min_risk_reward", 1.0)), 0.05)
        with col3:
            edits["trade_horizon_days"] = st.number_input(
                "trade_horizon_days", 1, 30, int(cfg.get("trade_horizon_days", 5)), 1)
        edits["equity_only"] = st.checkbox("equity_only (equities only, no options)",
                                           value=bool(cfg.get("equity_only", False)))

        st.markdown("**momentum_weights**")
        momentum = _edit_weights_dict(cfg.get("momentum_weights", {}), f"mw_{selected}")
        st.markdown("**signal_weights**")
        signal = _edit_weights_dict(cfg.get("signal_weights", {}), f"sw_{selected}")

        submitted = st.form_submit_button("💾 Save entry", type="primary")
        if submitted:
            new_cfg = {k: v for k, v in edits.items()}
            if momentum:
                new_cfg["momentum_weights"] = momentum
            if signal:
                new_cfg["signal_weights"] = signal
            db = load_weights_db(weights_db_path())
            if ticker is not None:
                db.setdefault(TICKER_NAMESPACE, {})[ticker] = new_cfg
            else:
                db[key] = new_cfg
            save_weights_db(weights_db_path(), db)
            st.success(f"Saved entry {selected}.")
            st.rerun()


def _checkpoints() -> None:
    section("💾", "Checkpoints",
            "Snapshots of weights_db.json. Restore reverts; applying a job auto-creates one.")
    c1, c2 = st.columns([1, 3])
    with c1:
        if st.button("📸 Snapshot now"):
            p = snapshot_weights_db(weights_db_path(), label="manual")
            st.success(f"Saved {p.name}")
    checkpoints = list_checkpoints()
    with c2:
        if checkpoints:
            sel = st.selectbox("Checkpoint", [c.name for c in checkpoints], key="cp_sel")
            ca, cb = st.columns(2)
            with ca:
                if st.button("↩ Restore", type="primary"):
                    restore_checkpoint(next(c for c in checkpoints if c.name == sel),
                                       weights_db_path())
                    st.success(f"Restored {sel}.")
                    st.rerun()
            with cb:
                if st.button("🗑 Delete"):
                    delete_checkpoint(next(c for c in checkpoints if c.name == sel))
                    st.rerun()
    if not checkpoints:
        st.caption("No checkpoints yet.")


def render() -> None:
    _launch_form()
    st.divider()
    _tune_jobs()
    st.divider()
    _weights_editor()
    st.divider()
    _checkpoints()
    jobs_auto_refresh()
