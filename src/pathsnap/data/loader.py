"""Robust readers for the four PathSnap raw file types.

File formats (verified against every trip in the supplied archive):

* ``.track``  lon, lat, timestamp[s]           (float, float, float)
* ``.nodes``  lon, lat                          row index == node_id
* ``.arcs``   from_node_id, to_node_id          row index == arc_id
* ``.route``  arc_id                            chronological reference route

The loaders never assume headers, report malformed content with a
``DataFormatError`` naming the file, and never guess missing values.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

EXTENSIONS = ("track", "nodes", "arcs", "route")


class DataFormatError(ValueError):
    """Raised when a raw file is missing, empty or malformed."""


def _read_table(path: Path, ncols: int, dtype) -> np.ndarray:
    if not path.exists():
        raise DataFormatError(f"missing file: {path.name}")
    if path.stat().st_size == 0:
        raise DataFormatError(f"empty file: {path.name}")
    try:
        df = pd.read_csv(path, sep=r"\s+", header=None, engine="c", comment=None)
    except pd.errors.EmptyDataError as exc:
        raise DataFormatError(f"empty file: {path.name}") from exc
    except Exception as exc:  # tokenisation errors -> ragged rows
        raise DataFormatError(f"cannot parse {path.name}: {exc}") from exc
    if df.shape[1] != ncols:
        raise DataFormatError(f"{path.name}: expected {ncols} columns, found {df.shape[1]}")
    values = df.apply(pd.to_numeric, errors="coerce").to_numpy()
    if dtype is np.int64:
        bad = ~np.isfinite(values) | (np.abs(values - np.round(values)) > 1e-9)
        if bad.any():
            row = int(np.argwhere(bad)[0][0])
            raise DataFormatError(f"{path.name}: non-integer value at row {row}")
        return values.astype(np.int64)
    return values.astype(np.float64)


def read_track(path: Path) -> np.ndarray:
    """Return an (N, 3) float array [lon, lat, t]. Non-numeric cells become NaN
    so that the validator can report them instead of crashing."""
    return _read_table(Path(path), 3, np.float64)


def read_nodes(path: Path) -> np.ndarray:
    arr = _read_table(Path(path), 2, np.float64)
    if not np.isfinite(arr).all():
        raise DataFormatError(f"{Path(path).name}: non-numeric node coordinates")
    return arr


def read_arcs(path: Path, n_nodes: int | None = None) -> np.ndarray:
    arr = _read_table(Path(path), 2, np.int64)
    if n_nodes is not None and (arr.min() < 0 or arr.max() >= n_nodes):
        raise DataFormatError(f"{Path(path).name}: arc references node outside [0, {n_nodes})")
    return arr


def read_route(path: Path, n_arcs: int | None = None) -> np.ndarray:
    arr = _read_table(Path(path), 1, np.int64).ravel()
    if n_arcs is not None and (arr.min() < 0 or arr.max() >= n_arcs):
        raise DataFormatError(f"{Path(path).name}: route references arc outside [0, {n_arcs})")
    return arr


@dataclass
class Trip:
    trip_id: str
    track: np.ndarray            # (N,3) lon, lat, t  -- raw, unmodified
    nodes: np.ndarray            # (V,2) lon, lat
    arcs: np.ndarray             # (E,2) from, to
    route: np.ndarray | None = None   # (M,) arc ids  -- reference, may be withheld
    partition: str = ""
    source_dir: Path | None = None
    meta: dict = field(default_factory=dict)

    @property
    def n_obs(self) -> int:
        return int(self.track.shape[0])


def trip_files(trip_dir: Path) -> dict[str, Path]:
    trip_dir = Path(trip_dir)
    tid = trip_dir.name
    return {ext: trip_dir / f"{tid}.{ext}" for ext in EXTENSIONS}


def load_trip(trip_dir: Path, with_route: bool = True, partition: str = "") -> Trip:
    """Load one trip folder. ``with_route=False`` never opens the .route file,
    which is how the prediction path guarantees it cannot see labels."""
    trip_dir = Path(trip_dir)
    if not trip_dir.is_dir():
        raise DataFormatError(f"trip folder not found: {trip_dir}")
    files = trip_files(trip_dir)
    track = read_track(files["track"])
    nodes = read_nodes(files["nodes"])
    arcs = read_arcs(files["arcs"], n_nodes=len(nodes))
    route = None
    if with_route and files["route"].exists():
        route = read_route(files["route"], n_arcs=len(arcs))
    return Trip(trip_dir.name, track, nodes, arcs, route, partition, trip_dir)


def list_trips(partition_dir: Path) -> list[str]:
    partition_dir = Path(partition_dir)
    if not partition_dir.is_dir():
        return []
    return sorted(p.name for p in partition_dir.iterdir() if p.is_dir())
