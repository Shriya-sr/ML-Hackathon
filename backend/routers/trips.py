from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool

from pathsnap.data.loader import DataFormatError

from ..dependencies import AppState, get_state
from ..schemas import Explanation, PredictionResponse, TripMetadata, TripSummary
from ..services.trip_service import TripNotFound

router = APIRouter(prefix="/api", tags=["trips"])
log = logging.getLogger("pathsnap.api")


def _meta(state: AppState, trip_id: str) -> dict:
    try:
        return state.registry.metadata(trip_id)
    except TripNotFound:
        raise HTTPException(404, detail=f"unknown trip id '{trip_id}'")
    except DataFormatError as exc:
        raise HTTPException(422, detail=f"trip files are malformed: {exc}")


@router.get("/trips", response_model=list[TripSummary])
def list_trips(state: AppState = Depends(get_state)):
    return [TripSummary(**{k: v for k, v in _meta(state, t).items() if k in TripSummary.model_fields})
            for t in state.registry.trip_ids]


@router.get("/trips/{trip_id}/metadata", response_model=TripMetadata)
def trip_metadata(trip_id: str, state: AppState = Depends(get_state)):
    return TripMetadata(**_meta(state, trip_id))


@router.post("/predict/{trip_id}", response_model=PredictionResponse)
async def predict(trip_id: str, state: AppState = Depends(get_state)):
    _meta(state, trip_id)  # validates id -> 404 before any work
    if state.service.matcher is None:
        raise HTTPException(503, detail=f"model not loaded: {state.model_error}")
    try:
        return await asyncio.wait_for(run_in_threadpool(state.service.predict, trip_id),
                                      timeout=state.settings.prediction_timeout_s)
    except asyncio.TimeoutError:
        log.error("prediction timeout for trip %s", trip_id)
        raise HTTPException(504, detail=f"prediction exceeded {state.settings.prediction_timeout_s:.0f}s")
    except DataFormatError as exc:
        raise HTTPException(422, detail=f"trip files are malformed: {exc}")
    except ValueError as exc:
        log.exception("prediction failed for %s", trip_id)
        raise HTTPException(422, detail=f"prediction failed: {exc}")


@router.get("/results/{trip_id}", response_model=PredictionResponse)
def results(trip_id: str, state: AppState = Depends(get_state)):
    _meta(state, trip_id)
    res = state.service.cached(trip_id)
    if res is None:
        raise HTTPException(404, detail="no result yet - POST /api/predict/{trip_id} first")
    return res


@router.get("/results/{trip_id}/explain/{fix_index}", response_model=Explanation)
def explain(trip_id: str, fix_index: int, state: AppState = Depends(get_state)):
    _meta(state, trip_id)
    try:
        out = state.service.explain(trip_id, fix_index)
    except IndexError:
        raise HTTPException(404, detail=f"fix index {fix_index} out of range")
    if out is None:
        raise HTTPException(404, detail="no result yet - POST /api/predict/{trip_id} first")
    return out
