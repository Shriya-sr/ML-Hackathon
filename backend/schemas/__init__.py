"""Pydantic response models. Note: no schema contains a reference-route field."""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class Health(BaseModel):
    status: str
    model_loaded: bool
    model_error: Optional[str] = None
    mode: str
    n_trips: int
    uptime_s: float


class TripSummary(BaseModel):
    trip_id: str
    n_fixes: int
    duration_s: float
    center: list[float] = Field(description="[lat, lon]")
    evaluation_available: bool = Field(description="server holds a reference route for scoring (never sent)")


class TripMetadata(TripSummary):
    bbox: list[float] = Field(description="[south, west, north, east]")
    n_nodes: int
    n_arcs: int
    n_gaps_over_30s: int
    max_gap_s: float


class ModelInfo(BaseModel):
    model: Optional[str]
    model_family: Optional[str]
    feature_version: Optional[str]
    n_features: int
    feature_columns: list[str]
    trained_at: Optional[str]
    git_commit: Optional[str]
    cv_macro_f1: Optional[float]
    student_test_top1: Optional[float]
    hmm_params: dict[str, Any]


class SystemInfo(BaseModel):
    mode: str
    partition: str
    n_trips: int
    api_version: str
    python: str
    server_time_utc: str
    note: str


class PredictionResponse(BaseModel):
    trip_id: str
    mode: str
    model: dict[str, Any]
    n_fixes: int
    gps: dict[str, list[float]]
    fixes: dict[str, list[Any]]
    arcs: dict[str, list[list[float]]]
    network: list[list[list[float]]]
    route: dict[str, list[Any]]
    robustness: dict[str, Any]
    metrics: dict[str, Any]
    timings: dict[str, float]
    diagnostics: dict[str, Any]


class Explanation(BaseModel):
    trip_id: str
    fix_index: int
    timestamp: float
    hmm_arc: int
    classifier_arc: int
    candidates: list[dict[str, Any]]
