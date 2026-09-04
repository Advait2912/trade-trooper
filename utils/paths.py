"""Central location for runtime artifacts (DBs, weights JSON, journals, logs).

All generated data files — news-sentiment caches, the industry/stock weights
DB, trade and backtest journals — live under the repo ``data/`` directory by
default so the project root stays clean.  Use ``data_path()`` instead of
hardcoding a bare filename.
"""

from __future__ import annotations

from pathlib import Path

#: Absolute path to the runtime-artifact directory (``<repo>/data``).
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def data_path(name: str) -> Path:
    """Return the absolute path to a runtime artifact under ``DATA_DIR``.

    Creates the directory if it does not yet exist.  Callers may also pass a
    user-supplied path explicitly, in which case ``name`` is used as-is.
    """
    if Path(name).is_absolute() or Path(name).parent != Path(""):
        return Path(name)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR / name
