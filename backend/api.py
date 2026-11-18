from __future__ import annotations

import os
from pathlib import Path
from functools import lru_cache
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from sklearn.metrics import accuracy_score, f1_score
from .data import discover_trips, partition_name
from .features import candidate_features
from .decoder import viterbi, continuity
from .maps import save_trip_map

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("PATHSNAP_DATA_DIR", ROOT.parent / "Student folder" / "Student folder"))
MODEL_PATH = Path(os.environ.get("PATHSNAP_MODEL", ROOT / "artifacts" / "pathsnap.joblib"))
app = FastAPI(title="PathSnap", version="0.1.0")
app.mount("/static", StaticFiles(directory=ROOT / "frontend"), name="static")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_PATH.exists(), "data_dir_exists": DATA_DIR.exists()}


@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Model artifact missing. Run backend.train first.")
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def list_dev_trips():
    try:
        return [t for t in discover_trips(DATA_DIR) if partition_name(t.partition) == "student_test"]
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/", response_class=HTMLResponse)
def index():
    return (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")


@app.get("/api/trips")
def trips():
    return {"trips": [{"id": t.trip_id, "observations": len(t.gps)} for t in list_dev_trips()]}


@app.post("/api/predict/{trip_id}")
def predict(trip_id: str):
    trip = next((t for t in list_dev_trips() if t.trip_id == trip_id), None)
    if trip is None:
        raise HTTPException(status_code=404, detail="Trip not found in student_test partition")
    artifact = load_model(); model = artifact["model"]
    ids, feats, labels, misses = candidate_features(trip, artifact.get("candidate_k", 8))
    import time
    started = time.perf_counter()
    scores = [model.predict_proba(x)[:, 1] for x in feats]
    direct = np.asarray([ids[i][int(np.argmax(scores[i]))] for i in range(len(ids))], dtype=int)
    decoded = viterbi(ids, scores, trip.arcs)
    inference = time.perf_counter() - started
    map_path = ROOT / "results" / f"trip_{trip_id}_prediction.html"
    save_trip_map(trip, decoded, map_path, include_reference=False)
    return {"trip_id": trip_id, "observations": len(trip.gps), "route_arc_ids": decoded.tolist(),
        "gps": trip.gps[:, :2].tolist(), "route_geometry": trip.nodes[trip.arcs[decoded]].tolist(),
        "map_url": f"/maps/trip_{trip_id}_prediction.html", "inference_seconds": inference,
        # Labels are evaluated inside the API; they are never included in the response.
        "metrics": {"classifier_top1_accuracy": float(accuracy_score(labels, direct)),
            "hmm_top1_accuracy": float(accuracy_score(labels, decoded)),
            "classifier_macro_f1": float(f1_score(labels, direct, average="macro", zero_division=0)),
            "hmm_macro_f1": float(f1_score(labels, decoded, average="macro", zero_division=0)),
            "route_continuity": continuity(decoded, trip.arcs), "candidate_miss_rate": float(np.mean(misses))}}


@app.get("/maps/{name}")
def maps(name: str):
    if not name.startswith("trip_") or "/" in name or ".." in name:
        raise HTTPException(status_code=404, detail="Map not found")
    path = ROOT / "results" / name
    if not path.exists():
        raise HTTPException(status_code=404, detail="Map not generated yet")
    return HTMLResponse(path.read_text(encoding="utf-8"))
