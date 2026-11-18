from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from .data import Trip

FEATURE_NAMES = ["distance_m", "heading_difference", "segment_length_m", "gps_speed_mps", "time_gap_s"]
EARTH_M = 111_320.0


def project(lonlat: np.ndarray, lat0: float) -> np.ndarray:
    xy = np.asarray(lonlat, dtype=float).copy()
    xy[:, 0] *= EARTH_M * np.cos(np.deg2rad(lat0))
    xy[:, 1] *= EARTH_M
    return xy


def headings(points: np.ndarray) -> np.ndarray:
    delta = np.diff(points, axis=0, prepend=points[:1])
    return np.arctan2(delta[:, 1], delta[:, 0])


def route_labels(trip: Trip) -> tuple[np.ndarray, np.ndarray]:
    """Align observations monotonically to the supplied ordered arc sequence."""
    if trip.route is None or len(trip.route) == 0:
        raise ValueError(f"Trip {trip.trip_id} has no reference route")
    if np.any(trip.route < 0) or np.any(trip.route >= len(trip.arcs)):
        raise ValueError(f"Trip {trip.trip_id} route contains invalid arc IDs")
    lat0 = float(np.median(trip.gps[:, 1]))
    xy = project(trip.gps[:, :2], lat0)
    nodes = project(trip.nodes, lat0)
    edges = nodes[trip.arcs[trip.route]]
    a, b = edges[:, 0], edges[:, 1]
    v = b - a
    vv = np.maximum((v * v).sum(axis=1), 1e-8)
    # Stream the dynamic program so long trips do not allocate an observations
    # by route-length distance matrix. Backpointers need only hold route indexes.
    pointer_dtype = np.uint16 if len(edges) < np.iinfo(np.uint16).max else np.uint32
    back = np.zeros((len(xy), len(edges)), dtype=pointer_dtype)
    previous = np.full(len(edges), np.inf, dtype=np.float64)
    selected_distance = np.empty(len(xy), dtype=np.float32)
    for i, point in enumerate(xy):
        u = np.clip(((point - a) * v).sum(axis=1) / vv, 0.0, 1.0)
        q = a + u[:, None] * v
        row = np.linalg.norm(point - q, axis=1)
        if i == 0:
            current = row.astype(np.float64)
        else:
            prefix = np.minimum.accumulate(previous)
            # First index attaining each cumulative minimum, for backtracking.
            best_idx = np.maximum.accumulate(
                np.where(previous == prefix, np.arange(len(edges)), -1)
            ).astype(pointer_dtype)
            current = row + prefix
            back[i] = best_idx
        previous = current
    state = np.empty(len(xy), dtype=np.int32)
    state[-1] = int(np.argmin(previous))
    for i in range(len(xy) - 1, 0, -1):
        state[i - 1] = back[i, state[i]]
    for i, j in enumerate(state):
        u = np.clip(((xy[i] - a[j]) * v[j]).sum() / vv[j], 0.0, 1.0)
        selected_distance[i] = np.linalg.norm(xy[i] - (a[j] + u * v[j]))
    return trip.route[state], selected_distance


def candidate_features(trip: Trip, k: int = 8) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray]:
    """Return candidate arc IDs, candidate feature rows, aligned labels, misses."""
    xy = project(trip.gps[:, :2], float(np.median(trip.gps[:, 1])))
    node_xy = project(trip.nodes, float(np.median(trip.gps[:, 1])))
    edges = node_xy[trip.arcs]
    mid = edges.mean(axis=1)
    tree = cKDTree(mid)
    n_candidates = min(k, len(trip.arcs))
    pool_size = min(max(64, n_candidates * 8), len(trip.arcs))
    _, candidates = tree.query(xy, k=pool_size)
    if pool_size == 1:
        candidates = candidates[:, None]
    labels, label_distance = route_labels(trip)
    gps_head = headings(xy)
    gps_dist = np.linalg.norm(np.diff(xy, axis=0, prepend=xy[:1]), axis=1)
    dt = np.diff(trip.gps[:, 2], prepend=trip.gps[:1, 2])
    speed = gps_dist / np.maximum(dt, 1.0)
    feats, ids = [], []
    for i, pool in enumerate(candidates):
        pa, pb = edges[pool, 0], edges[pool, 1]
        pv = pb - pa
        pvv = np.maximum((pv * pv).sum(axis=1), 1e-8)
        pu = np.clip(((xy[i] - pa) * pv).sum(axis=1) / pvv, 0, 1)
        pp = pa + pu[:, None] * pv
        pool_distance = np.linalg.norm(xy[i] - pp, axis=1)
        cand = pool[np.argsort(pool_distance)[:n_candidates]]
        a, b = edges[cand, 0], edges[cand, 1]
        v = b - a
        vv = np.maximum((v * v).sum(axis=1), 1e-8)
        u = np.clip(((xy[i] - a) * v).sum(axis=1) / vv, 0, 1)
        proj = a + u[:, None] * v
        d = np.linalg.norm(xy[i] - proj, axis=1)
        ang = np.arctan2(v[:, 1], v[:, 0])
        ad = np.abs((ang - gps_head[i] + np.pi) % (2 * np.pi) - np.pi)
        ids.append(np.asarray(cand, dtype=np.int32))
        feats.append(np.column_stack((d, ad, np.sqrt(vv), np.full(len(cand), speed[i]), np.full(len(cand), dt[i]))).astype(np.float32))
    misses = np.asarray([labels[i] not in ids[i] for i in range(len(labels))], dtype=bool)
    return ids, feats, labels, misses
