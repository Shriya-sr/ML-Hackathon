"""Lightweight experiment tracking: JSON/CSV records with config, seed, feature
version, timestamps and the git commit that produced them."""
from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..config import PROJECT_ROOT


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def git_dirty() -> bool:
    try:
        out = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=PROJECT_ROOT,
                                      stderr=subprocess.DEVNULL, text=True)
        return bool(out.strip())
    except Exception:
        return True


def run_metadata() -> dict:
    import sklearn
    return {"timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_commit": git_commit(), "git_dirty": git_dirty(), "python": platform.python_version(),
            "sklearn": sklearn.__version__, "platform": platform.platform()}


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def write_json(path: Path, obj) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_default), encoding="utf-8")
    return path
