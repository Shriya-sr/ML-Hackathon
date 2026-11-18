"""Application state shared by the routers (created once at startup)."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from pathsnap.inference.pipeline import ArtifactError, MapMatcher

from .services.prediction_service import PredictionService
from .services.trip_service import TripRegistry
from .settings import Settings, get_settings

log = logging.getLogger("pathsnap.api")


@dataclass
class AppState:
    settings: Settings
    registry: TripRegistry
    service: PredictionService
    model_error: str | None
    started: float


_state: AppState | None = None


def init_state(settings: Settings | None = None) -> AppState:
    global _state
    settings = settings or get_settings()
    registry = TripRegistry(settings.evaluation_partition, settings.fallback_partition)
    matcher, err = None, None
    try:
        matcher = MapMatcher()
        log.info("model loaded: %s", matcher.info()["model"])
    except (ArtifactError, Exception) as exc:  # API still starts and reports the problem on /health
        err = f"{type(exc).__name__}: {exc}"
        log.error("model could not be loaded: %s", err)
    _state = AppState(settings, registry, PredictionService(matcher, registry, settings), err, time.time())
    log.info("mode=%s partition=%s trips=%d", registry.mode, registry.partition, len(registry.trip_ids))
    return _state


def get_state() -> AppState:
    if _state is None:
        return init_state()
    return _state
