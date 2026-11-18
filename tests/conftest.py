"""Shared fixtures: synthetic trips written in the official raw format."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

DEG = 1 / 111320.0  # ~1 m in degrees of latitude near the equator


def write_trip(folder: Path, tid: str, track, nodes, arcs, route=None):
    d = folder / tid
    d.mkdir(parents=True, exist_ok=True)
    np.savetxt(d / f"{tid}.track", np.asarray(track), fmt="%.7f", delimiter="\t")
    np.savetxt(d / f"{tid}.nodes", np.asarray(nodes), fmt="%.7f", delimiter="\t")
    np.savetxt(d / f"{tid}.arcs", np.asarray(arcs, dtype=int), fmt="%d", delimiter="\t")
    if route is not None:
        np.savetxt(d / f"{tid}.route", np.asarray(route, dtype=int), fmt="%d")
    return d


def grid_network(n=6, spacing_m=100.0, lon0=10.0, lat0=50.0):
    """n x n grid of two-way streets. Returns nodes (lon,lat), arcs, node index fn."""
    k = np.cos(np.radians(lat0))
    nodes = []
    for r in range(n):
        for c in range(n):
            nodes.append((lon0 + c * spacing_m * DEG / k, lat0 + r * spacing_m * DEG))
    idx = lambda r, c: r * n + c  # noqa: E731
    arcs = []
    for r in range(n):
        for c in range(n):
            if c + 1 < n:
                arcs += [(idx(r, c), idx(r, c + 1)), (idx(r, c + 1), idx(r, c))]
            if r + 1 < n:
                arcs += [(idx(r, c), idx(r + 1, c)), (idx(r + 1, c), idx(r, c))]
    return np.array(nodes), np.array(arcs), idx


@pytest.fixture
def grid_trip(tmp_path):
    """Vehicle drives east along row 2 of a 6x6 grid at ~10 m/s with 3 m noise."""
    nodes, arcs, idx = grid_network()
    arc_id = {tuple(a): i for i, a in enumerate(arcs.tolist())}
    route = [arc_id[(idx(2, c), idx(2, c + 1))] for c in range(5)]
    rng = np.random.default_rng(0)
    lon_s, lon_e = nodes[idx(2, 0)][0], nodes[idx(2, 5)][0]
    lat = nodes[idx(2, 0)][1]
    n = 50
    lons = np.linspace(lon_s + 5 * DEG, lon_e - 5 * DEG, n)
    track = np.column_stack([lons + rng.normal(0, 3 * DEG, n), lat + rng.normal(0, 3 * DEG, n), np.arange(1, n + 1)])
    d = write_trip(tmp_path / "student_test", "00000001", track, nodes, arcs, route)
    return {"dir": d, "nodes": nodes, "arcs": arcs, "route": route, "track": track, "idx": idx, "root": tmp_path}
