"""Data-quality checks for one trip. Produces a flat dict used by EDA and the API."""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .loader import Trip


def valid_lonlat(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    return np.isfinite(lon) & np.isfinite(lat) & (np.abs(lon) <= 180) & (np.abs(lat) <= 90)


def validate_trip(trip: Trip, gap_threshold_s: float = 30.0) -> dict:
    tr = trip.track
    lon, lat, t = tr[:, 0], tr[:, 1], tr[:, 2]
    ok = valid_lonlat(lon, lat)
    t_ok = np.isfinite(t)
    dt = np.diff(t[t_ok]) if t_ok.sum() > 1 else np.array([])
    dup_rows = int(len(tr) - len(np.unique(tr, axis=0))) if len(tr) else 0
    dup_ts = int(len(t[t_ok]) - len(np.unique(t[t_ok])))

    V, E = len(trip.nodes), len(trip.arcs)
    a = trip.arcs
    deg = np.bincount(a.ravel(), minlength=V)
    isolated = int((deg == 0).sum())
    g = coo_matrix((np.ones(E), (a[:, 0], a[:, 1])), shape=(V, V))
    n_weak, weak_lbl = connected_components(g, directed=True, connection="weak")
    n_strong, strong_lbl = connected_components(g, directed=True, connection="strong")
    # size of the largest weak component, counted on non-isolated nodes
    active = deg > 0
    largest_weak = int(np.bincount(weak_lbl[active]).max()) if active.any() else 0
    largest_strong = int(np.bincount(strong_lbl[active]).max()) if active.any() else 0
    self_loops = int((a[:, 0] == a[:, 1]).sum())
    arc_set = set(map(tuple, a.tolist()))
    two_way = float(np.mean([(y, x) in arc_set for x, y in a.tolist()])) if E else 0.0
    dup_arcs = int(E - len(arc_set))

    out = {
        "trip_id": trip.trip_id,
        "partition": trip.partition,
        "n_obs": int(len(tr)),
        "n_invalid_coord": int((~ok).sum()),
        "n_invalid_time": int((~t_ok).sum()),
        "n_duplicate_rows": dup_rows,
        "n_duplicate_timestamps": dup_ts,
        "timestamps_monotonic": bool((dt > 0).all()) if len(dt) else True,
        "n_nonpositive_dt": int((dt <= 0).sum()) if len(dt) else 0,
        "dt_min": float(dt.min()) if len(dt) else np.nan,
        "dt_median": float(np.median(dt)) if len(dt) else np.nan,
        "dt_max": float(dt.max()) if len(dt) else np.nan,
        "frac_dt_1s": float(np.mean(dt == 1)) if len(dt) else np.nan,
        "n_gaps_over_threshold": int((dt > gap_threshold_s).sum()) if len(dt) else 0,
        "duration_s": float(t[t_ok].max() - t[t_ok].min()) if t_ok.any() else 0.0,
        "lon_min": float(np.nanmin(lon)), "lon_max": float(np.nanmax(lon)),
        "lat_min": float(np.nanmin(lat)), "lat_max": float(np.nanmax(lat)),
        "n_nodes": V, "n_arcs": E,
        "n_isolated_nodes": isolated,
        "n_weak_components": int(n_weak - isolated),   # isolated nodes are their own component
        "largest_weak_component_frac": largest_weak / max(int(active.sum()), 1),
        "n_strong_components": int(n_strong - isolated),
        "largest_strong_component_frac": largest_strong / max(int(active.sum()), 1),
        "n_self_loops": self_loops,
        "n_duplicate_arcs": dup_arcs,
        "two_way_arc_frac": two_way,
    }
    if trip.route is not None:
        r = trip.route
        valid_ids = (r >= 0) & (r < E)
        rc = r[valid_ids]
        if len(rc) > 1:
            connected = a[rc[:-1], 1] == a[rc[1:], 0]
        else:
            connected = np.array([True])
        out.update({
            "route_len": int(len(r)),
            "route_unique_arcs": int(len(np.unique(r))),
            "route_invalid_ids": int((~valid_ids).sum()),
            "route_breaks": int((~connected).sum()),
            "route_continuity": float(connected.mean()),
            "route_repeated_consecutive": int((r[1:] == r[:-1]).sum()) if len(r) > 1 else 0,
        })
    return out
