"""Backend settings: configs/api.yaml overlaid with PATHSNAP_* environment variables."""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from pathsnap.config import load_config  # noqa: E402


@dataclass
class Settings:
    host: str = "0.0.0.0"
    port: int = 8000
    evaluation_partition: str = "evaluator_test"
    fallback_partition: str = "student_test"
    cors_origins: list[str] = field(default_factory=lambda: ["*"])
    prediction_timeout_s: float = 120.0
    result_cache_size: int = 20
    low_confidence_threshold: float = 0.5
    max_network_arcs: int = 12000


def get_settings() -> Settings:
    cfg = load_config("api")
    s = Settings(**{k: v for k, v in cfg.items() if k in Settings.__dataclass_fields__})
    env = os.environ
    s.host = env.get("PATHSNAP_HOST", s.host)
    s.port = int(env.get("PATHSNAP_PORT", s.port))
    s.evaluation_partition = env.get("PATHSNAP_EVAL_PARTITION", s.evaluation_partition)
    if env.get("PATHSNAP_CORS_ORIGINS"):
        s.cors_origins = [o.strip() for o in env["PATHSNAP_CORS_ORIGINS"].split(",") if o.strip()]
    s.prediction_timeout_s = float(env.get("PATHSNAP_PREDICT_TIMEOUT", s.prediction_timeout_s))
    return s
