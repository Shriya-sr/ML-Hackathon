from __future__ import annotations

import platform
import time
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException

from pathsnap.features.feature_schema import FEATURE_COLUMNS

from ..dependencies import AppState, get_state
from ..schemas import Health, ModelInfo, SystemInfo

router = APIRouter()
API_VERSION = "1.0.0"


@router.get("/health", response_model=Health, tags=["system"])
def health(state: AppState = Depends(get_state)):
    ok = state.service.matcher is not None
    return Health(status="ok" if ok else "degraded", model_loaded=ok, model_error=state.model_error,
                  mode=state.registry.mode, n_trips=len(state.registry.trip_ids), uptime_s=time.time() - state.started)


@router.get("/api/model/info", response_model=ModelInfo, tags=["system"])
def model_info(state: AppState = Depends(get_state)):
    m = state.service.matcher
    if m is None:
        raise HTTPException(503, detail=f"model not loaded: {state.model_error}")
    info = m.info()
    return ModelInfo(**info, feature_columns=FEATURE_COLUMNS)


@router.get("/api/system/info", response_model=SystemInfo, tags=["system"])
def system_info(state: AppState = Depends(get_state)):
    r = state.registry
    note = ("Official evaluator trips loaded." if r.mode == "EVALUATION" else
            "DEVELOPMENT mode: evaluator_test folder is empty, serving the 10 student-testing trips instead. "
            "Drop the reserved trips into data/raw/evaluator_test (or set PATHSNAP_DATA_DIR) and restart.")
    return SystemInfo(mode=r.mode, partition=r.partition, n_trips=len(r.trip_ids), api_version=API_VERSION,
                      python=platform.python_version(),
                      server_time_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"), note=note)
