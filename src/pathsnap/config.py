"""Central configuration and path handling.

All tunable settings live in ``configs/*.yaml``. Paths can be overridden with
environment variables so that Laptop 1 can point the API at the evaluator
dataset without touching code:

* ``PATHSNAP_DATA_DIR``      root containing ``train/``, ``student_test/``, ``evaluator_test/``
* ``PATHSNAP_ARTIFACT_DIR``  folder holding the exported model bundle
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"


def data_dir() -> Path:
    return Path(os.environ.get("PATHSNAP_DATA_DIR", PROJECT_ROOT / "data" / "raw"))


def artifact_dir() -> Path:
    return Path(os.environ.get("PATHSNAP_ARTIFACT_DIR", PROJECT_ROOT / "artifacts"))


def processed_dir() -> Path:
    p = PROJECT_ROOT / "data" / "processed"
    p.mkdir(parents=True, exist_ok=True)
    return p


@lru_cache(maxsize=None)
def load_config(name: str) -> dict[str, Any]:
    """Load ``configs/<name>.yaml`` as a dict."""
    path = CONFIG_DIR / f"{name}.yaml"
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


SEED = 42
