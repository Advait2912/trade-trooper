"""Shared Streamlit widgets for rendering async jobs (progress, stop, logs)."""

from __future__ import annotations

import streamlit as st

from web.ui import jobs
from web.ui.theme import status_pill


def render_job(job: dict) -> None:
    """Render a job card: status pill, trial counter, progress bar, logs."""
    status = job.get("status", "missing")
    label = job.get("label") or job.get("kind", "job")
    status_map = {
        "running": ("🟢 RUNNING", "running"),
        "done": ("✅ DONE", "success"),
        "error": ("❌ ERROR", "error"),
        "missing": ("⚠ MISSING", "warning"),
    }
    pill, kind = status_map.get(status, ("⚪ UNKNOWN", "info"))
    status_pill(f"{pill} · {label}", kind)

    progress = job.get("progress")
    done = int(progress["done"]) if progress and progress.get("done") else 0
    total = int(progress["total"]) if progress and progress.get("total") else 0

    if status == "running":
        if total:
            frac = min(1.0, done / total)
            st.progress(frac)
            eta = _eta(job, done, total)
            st.markdown(f"**Trial {done} of {total}** · {100 * done // total}%{eta}")
            msg = (progress or {}).get("message", "")
            if msg:
                st.markdown(f"`{msg[:140]}`")
        else:
            st.progress(0)
            st.caption("Waiting for the first trial to finish…")

    if job.get("exit_code") not in (None, 0):
        st.caption(f"Exit code: {job['exit_code']}")
    if job.get("message"):
        st.caption(job["message"])

    log_tail = job.get("log_tail") or []
    if log_tail:
        with st.expander("Log tail"):
            st.code("\n".join(log_tail[-6:]))

    if status == "running":
        if st.button("⏹ Stop", key=f"stop_{job['id']}"):
            ok, msg = jobs.stop(job["id"])
            st.caption(msg)
            st.rerun()


def _eta(job: dict, done: int, total: int) -> str:
    """Estimate remaining time from the last progress timestamp."""
    from datetime import datetime, timezone

    ts = (job.get("progress") or {}).get("updated_at")
    if not ts or done <= 0:
        return ""
    try:
        updated = datetime.fromisoformat(ts)
    except (TypeError, ValueError):
        return ""
    now = datetime.now(timezone.utc)
    elapsed_s = max(1.0, (now - updated).total_seconds())
    if elapsed_s > 300:
        return ""
    remaining_s = elapsed_s * (total - done) / done
    return f" · ~{int(remaining_s // 60)}m {int(remaining_s % 60)}s left"


def render_job_list(kind: str | None = None, limit: int = 8) -> None:
    """Render recent jobs of a kind with progress."""
    items = jobs.list_jobs(kind)[:limit]
    if not items:
        st.caption("No jobs yet â€” launch one above.")
        return
    for job in items:
        render_job(job)
        st.divider()


def jobs_auto_refresh(refresh_every_s: int = 8) -> None:
    """Rerun the page periodically while any job is active."""
    import time

    active = any(j.get("status") == "running" for j in jobs.list_jobs())
    if not active:
        return
    now = time.time()
    last = st.session_state.get("jobs_last_refresh", now)
    if now - last >= refresh_every_s:
        st.session_state["jobs_last_refresh"] = now
        st.rerun()
