"""UI job manager — detached subprocess backtests / tuning runs.

Each job lives in ``data/jobs/<id>/``:

    job.json    — metadata (id, kind, label, cmd, pid, created_at)
    job.log     — combined stdout/stderr
    state.json  — written by ``scripts/job_runner.py`` (running/done/error)
    artifacts   — job-scoped outputs (e.g. ``weights.json``, ``trades.db``)

Backtest and tuning jobs run concurrently (the paper runner stays a
singleton).  Status is polled from the pid + state file, and tune progress is
parsed from the log's ``[n/total] score=...`` lines.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from web.ui.data import repo_root

_PROGRESS_RE = re.compile(r"\[(\d+)/(\d+)\](.*)$", re.MULTILINE)


def jobs_dir() -> Path:
    d = repo_root() / "data" / "jobs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def job_dir(job_id: str) -> Path:
    return jobs_dir() / job_id


def job_artifact(job_id: str, name: str) -> Path:
    return job_dir(job_id) / name


def progress_file(job_id: str) -> Path:
    return job_artifact(job_id, "progress.json")


def _new_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]


def _alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        # os.kill(pid, 0) is invalid on Windows (raises WinError 87); probe
        # the process handle instead.
        import ctypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        kernel32.CloseHandle(handle)
        return True
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def start(kind: str, cmd: list[str], label: str = "", cwd: str | None = None) -> str:
    """Convenience: create a job dir, then launch the command."""
    job_id = create(kind, label)
    launch(job_id, cmd, cwd=cwd)
    return job_id


def create(kind: str, label: str = "") -> str:
    """Allocate a job id + directory (before the command is known)."""
    job_id = _new_id()
    jd = job_dir(job_id)
    jd.mkdir(parents=True, exist_ok=True)
    meta = {
        "id": job_id,
        "kind": kind,
        "label": label,
        "cmd": [],
        "pid": 0,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (jd / "job.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return job_id


def launch(job_id: str, cmd: list[str], cwd: str | None = None) -> int:
    """Spawn the command under an existing job id; returns the pid."""
    jd = job_dir(job_id)
    jd.mkdir(parents=True, exist_ok=True)
    state_path = jd / "state.json"
    log_path = jd / "job.log"

    wrapper = [
        sys.executable, str(repo_root() / "scripts" / "job_runner.py"),
        "--state", str(state_path), "--",
    ] + cmd

    log_file = open(log_path, "a", encoding="utf-8")
    try:
        proc = subprocess.Popen(
            wrapper,
            cwd=cwd or str(repo_root()),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    except Exception:
        log_file.close()
        raise

    meta = _read_json(jd / "job.json")
    meta.update({"cmd": cmd, "pid": proc.pid})
    (jd / "job.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return proc.pid


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _progress_from_log(text: str) -> dict | None:
    """Parse the last ``[n/total] msg`` line from a tune log."""
    matches = list(_PROGRESS_RE.finditer(text))
    if not matches:
        return None
    m = matches[-1]
    return {"done": int(m.group(1)), "total": int(m.group(2)),
            "message": m.group(3).strip()}


def status(job_id: str) -> dict:
    """Polled status of a job: running / done / error / missing."""
    jd = job_dir(job_id)
    meta = _read_json(jd / "job.json")
    state = _read_json(jd / "state.json")
    if not meta and not state:
        return {"id": job_id, "status": "missing"}

    log_text = ""
    log_path = jd / "job.log"
    if log_path.exists():
        try:
            log_text = log_path.read_text(encoding="utf-8")[-20000:]
        except OSError:
            log_text = ""

    pid = int(meta.get("pid") or 0)
    running = state.get("status") == "running" and (pid and _alive(pid))
    if running or state.get("status") in ("done", "error"):
        status_str = state.get("status", "running")
    else:
        status_str = "done" if state else "running"
        if not state and pid and not _alive(pid):
            status_str = "done"

    tail = log_text.splitlines()[-8:]
    progress = _progress_from_log(log_text)
    pf_path = jd / "progress.json"
    if pf_path.exists():
        try:
            pf = _read_json(pf_path)
            done = int(pf.get("done", 0))
            total = int(pf.get("total", 0))
            if done and total:
                progress = {"done": done, "total": total,
                            "message": str(pf.get("message", "")),
                            "updated_at": pf.get("updated_at", "")}
        except (ValueError, TypeError):
            pass
    return {
        "id": job_id,
        "kind": meta.get("kind", ""),
        "label": meta.get("label", ""),
        "cmd": meta.get("cmd", []),
        "pid": pid,
        "status": status_str,
        "exit_code": state.get("exit_code"),
        "message": state.get("message"),
        "progress": progress,
        "log_tail": tail,
        "created_at": meta.get("created_at", ""),
    }


def stop(job_id: str) -> tuple[bool, str]:
    meta = _read_json(job_dir(job_id) / "job.json")
    pid = int(meta.get("pid") or 0)
    if not pid or not _alive(pid):
        return True, "Job is not running."
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        pass
    for _ in range(10):
        if not _alive(pid):
            break
        time.sleep(0.5)
    else:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
    return True, f"Job {job_id} stopped."


def list_jobs(kind: str | None = None) -> list[dict]:
    out = []
    for jd in sorted(jobs_dir().glob("*/"), reverse=True):
        meta = _read_json(jd / "job.json")
        if not meta:
            continue
        if kind and meta.get("kind") != kind:
            continue
        out.append(status(meta["id"]))
    return out


def is_running(job_id: str) -> bool:
    return status(job_id).get("status") == "running"


def get_job_log(job_id: str, max_lines: int = 60) -> list[str]:
    """Return the last N lines of a job's log."""
    log_path = job_dir(job_id) / "job.log"
    if not log_path.exists():
        return []
    try:
        return log_path.read_text(encoding="utf-8").splitlines()[-max_lines:]
    except OSError:
        return []
