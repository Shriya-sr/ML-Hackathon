from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass
class Trip:
    trip_id: str
    partition: str
    folder: Path
    gps: np.ndarray
    nodes: np.ndarray
    arcs: np.ndarray
    route: np.ndarray | None
    raw_gps_rows: int = 0
    invalid_gps_rows: int = 0


def _read(path: Path, dtype=float) -> np.ndarray:
    try:
        data = np.loadtxt(path, dtype=dtype, ndmin=2)
    except ValueError as exc:
        raise ValueError(f"Could not parse {path}: {exc}") from exc
    return data


def load_trip(track_path: Path, partition: str | None = None) -> Trip:
    base = track_path.with_suffix("")
    required = [base.with_suffix(ext) for ext in (".nodes", ".arcs")]
    missing = [p.name for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Trip {track_path.parent.name} is missing {missing}")
    gps = _read(track_path)
    raw_gps_rows = len(gps)
    nodes = _read(base.with_suffix(".nodes"))
    arcs = _read(base.with_suffix(".arcs"), dtype=np.int64)
    route_path = base.with_suffix(".route")
    route = _read(route_path, dtype=np.int64).reshape(-1) if route_path.exists() else None
    if gps.shape[1] != 3 or nodes.shape[1] != 2 or arcs.shape[1] != 2:
        raise ValueError(f"Unexpected columns in trip {track_path.parent.name}")
    if np.any(arcs < 0) or np.any(arcs >= len(nodes)):
        raise ValueError(f"Trip {track_path.parent.name} has arc endpoints outside its node table")
    finite = np.isfinite(gps).all(axis=1)
    gps = gps[finite]
    if len(gps) == 0:
        raise ValueError(f"Trip {track_path.parent.name} has no valid GPS observations")
    return Trip(track_path.parent.name, partition or track_path.parent.parent.name,
                track_path.parent, gps, nodes, arcs, route, raw_gps_rows,
                int(raw_gps_rows - len(gps)))


def discover_trips(root: str | Path, partitions: set[str] | None = None) -> list[Trip]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    tracks = sorted(root.rglob("*.track"))
    if partitions:
        tracks = [p for p in tracks if p.parent.parent.name.lower() in partitions]
    if not tracks:
        raise FileNotFoundError(f"No .track files found under {root}")
    trips = [load_trip(p) for p in tracks]
    ids = [t.trip_id for t in trips]
    if len(ids) != len(set((t.partition, t.trip_id) for t in trips)):
        raise ValueError("Duplicate trip IDs within a partition")
    return trips


def partition_name(value: str) -> str:
    name = value.lower().replace("-", "_")
    if name in {"train", "training"}:
        return "train"
    if name in {"student_test", "student_testing", "studenttest"}:
        return "student_test"
    return name
