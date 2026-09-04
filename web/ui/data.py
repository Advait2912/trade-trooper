"""Relative-path data loaders for the dashboard.

Every path is derived from ``repo_root()`` (computed from this file's
location) — no absolute paths, no ``.absolute()``.
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import pandas as pd


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def data_dir() -> Path:
    return repo_root() / "data"


def journals_dir() -> Path:
    return data_dir() / "journals"


def logs_dir() -> Path:
    return data_dir() / "logs"


def cache_db() -> Path:
    return data_dir() / "news_cache.db"


def weights_db_path() -> Path:
    return data_dir() / "weights_db_tuned_backup.json"


def env_path() -> Path:
    return repo_root() / ".env"


def runner_pid_file() -> Path:
    return data_dir() / "runner.pid"


def runner_log_file() -> Path:
    return data_dir() / "runner.log"


def venv_python() -> str:
    """Python interpreter for subprocesses (the venv running streamlit)."""
    import sys

    return sys.executable


# ---------------------------------------------------------------------------
# Journal / log readers
# ---------------------------------------------------------------------------
def _connect(db_path: Path) -> sqlite3.Connection | None:
    if not db_path.exists():
        return None
    return sqlite3.connect(str(db_path))


def load_journals() -> dict[str, sqlite3.Connection]:
    conns: dict[str, sqlite3.Connection] = {}
    if not journals_dir().exists():
        return conns
    for f in sorted(journals_dir().glob("journal_*.db")):
        ticker = f.stem.removeprefix("journal_").upper()
        conns[ticker] = _connect(f)
    return conns


def load_latest_decisions(n: int = 20) -> pd.DataFrame:
    rows: list[dict] = []
    if logs_dir().exists():
        for f in sorted(logs_dir().glob("decisions_*.jsonl")):
            for line in f.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    if not rows:
        return pd.DataFrame(columns=["ts", "ticker", "decision", "composite_bias", "confidence"])
    return pd.DataFrame(rows).tail(n)


def load_equity_history() -> pd.DataFrame:
    frames = []
    for ticker, conn in load_journals().items():
        try:
            df = pd.read_sql_query("SELECT ts, equity, cash FROM equity ORDER BY ts", conn)
            df["ticker"] = ticker
            frames.append(df)
        except sqlite3.Error:
            continue
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_trades() -> pd.DataFrame:
    frames = []
    for ticker, conn in load_journals().items():
        try:
            df = pd.read_sql_query(
                "SELECT opened_ts, closed_ts, ticker, instrument, option_type, quantity, "
                "entry_price, exit_price, pnl, pnl_pct, exit_reason FROM trades ORDER BY id",
                conn,
            )
            frames.append(df)
        except sqlite3.Error:
            continue
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_cycles() -> pd.DataFrame:
    frames = []
    for ticker, conn in load_journals().items():
        try:
            df = pd.read_sql_query(
                "SELECT ts, ticker, decision, composite_bias, confidence, snapshot "
                "FROM cycles ORDER BY id",
                conn,
            )
            frames.append(df)
        except sqlite3.Error:
            continue
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def latest_report_snapshot() -> dict | None:
    latest: tuple[str, str] | None = None
    snapshot = None
    for ticker, conn in load_journals().items():
        try:
            row = conn.execute("SELECT ts, snapshot FROM cycles ORDER BY id DESC LIMIT 1").fetchone()
            if row and (latest is None or row[0] > latest[0]):
                latest = (row[0], ticker)
                snapshot = json.loads(row[1])
        except (sqlite3.Error, json.JSONDecodeError):
            continue
    return snapshot


def load_weights_db() -> dict:
    if not weights_db_path().exists():
        return {}
    return json.loads(weights_db_path().read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# .env helpers
# ---------------------------------------------------------------------------
def read_env() -> dict[str, str]:
    env: dict[str, str] = {}
    p = env_path()
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                env[k.strip()] = v.strip()
    return env


def write_env(patch: dict[str, str]) -> None:
    """Atomically patch .env (tmp file + os.replace)."""
    env_file = env_path()
    env = read_env()
    env.update(patch)
    lines = [f"{k}={v}" for k, v in env.items()]
    tmp = env_file.with_suffix(".env.tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(tmp, env_file)
    try:
        os.chmod(env_file, 0o600)
    except OSError:
        pass
