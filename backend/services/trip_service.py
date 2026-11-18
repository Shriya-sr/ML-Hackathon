"""Trip registry for the evaluation partition.

Mode selection:
* EVALUATION  - ``<data>/evaluator_test`` contains trip folders (official reserved trips)
* DEVELOPMENT - evaluator folder empty/missing -> fall back to ``student_test`` and say so
Metadata is computed from .track/.nodes/.arcs only; .route is only checked for
existence (to tell the UI whether server-side scoring is possible).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from pathsnap.config import data_dir
from pathsnap.data.loader import list_trips, read_nodes, read_track

TRIP_ID_RE = re.compile(r"^[0-9A-Za-z_\-]{1,64}$")


class TripNotFound(KeyError):
    pass


class TripRegistry:
    def __init__(self, evaluation_partition: str, fallback_partition: str):
        root = data_dir()
        ev = root / evaluation_partition
        if list_trips(ev):
            self.mode, self.partition, self.dir = "EVALUATION", evaluation_partition, ev
        else:
            self.mode, self.partition, self.dir = "DEVELOPMENT", fallback_partition, root / fallback_partition
        self.trip_ids = list_trips(self.dir)
        self._meta: dict[str, dict] = {}

    def trip_dir(self, trip_id: str) -> Path:
        if not TRIP_ID_RE.match(trip_id or "") or trip_id not in self.trip_ids:
            raise TripNotFound(trip_id)
        return self.dir / trip_id

    def has_reference(self, trip_id: str) -> bool:
        d = self.trip_dir(trip_id)
        return (d / f"{trip_id}.route").exists()

    def metadata(self, trip_id: str) -> dict:
        if trip_id in self._meta:
            return self._meta[trip_id]
        d = self.trip_dir(trip_id)
        tr = read_track(d / f"{trip_id}.track")
        nodes = read_nodes(d / f"{trip_id}.nodes")
        n_arcs = sum(1 for _ in open(d / f"{trip_id}.arcs", "rb"))
        ok = np.isfinite(tr).all(1)
        tr = tr[ok]
        dt = np.diff(np.sort(tr[:, 2])) if len(tr) > 1 else np.array([0.0])
        meta = {
            "trip_id": trip_id, "n_fixes": int(len(tr)),
            "duration_s": float(tr[:, 2].max() - tr[:, 2].min()) if len(tr) else 0.0,
            "center": [float(np.median(tr[:, 1])), float(np.median(tr[:, 0]))] if len(tr) else [0.0, 0.0],
            "bbox": [float(tr[:, 1].min()), float(tr[:, 0].min()), float(tr[:, 1].max()), float(tr[:, 0].max())]
            if len(tr) else [0.0, 0.0, 0.0, 0.0],
            "n_nodes": int(len(nodes)), "n_arcs": int(n_arcs),
            "n_gaps_over_30s": int((dt > 30).sum()), "max_gap_s": float(dt.max()) if len(dt) else 0.0,
            "evaluation_available": self.has_reference(trip_id),
        }
        self._meta[trip_id] = meta
        return meta
