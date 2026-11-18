"""End-to-end edge cases through the REAL exported model (MapMatcher + RouteEvaluator)
on synthetic trips written in the official file format."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pathsnap.inference.pipeline import MapMatcher, RouteEvaluator
from tests.conftest import DEG, grid_network, write_trip

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(not (ROOT / "artifacts" / "models" / "final_classifier.joblib").exists(),
                                reason="model artifacts not exported yet")


@pytest.fixture(scope="module")
def matcher():
    return MapMatcher()


def _east_track(nodes, idx, n=40, noise=3.0, row=2, seed=0, dt=1.0):
    rng = np.random.default_rng(seed)
    lat = nodes[idx(row, 0)][1]
    lons = np.linspace(nodes[idx(row, 0)][0] + 5 * DEG, nodes[idx(row, 5)][0] - 5 * DEG, n)
    t = np.arange(1, n + 1) * dt
    return np.column_stack([lons + rng.normal(0, noise * DEG, n), lat + rng.normal(0, noise * DEG, n), t])


def test_normal_trip_end_to_end(matcher, grid_trip):
    pred = matcher.predict(grid_trip["dir"])
    m = RouteEvaluator().evaluate(grid_trip["dir"], pred)
    assert m["available"] and m["hmm"]["top1_accuracy"] >= 0.8
    assert m["hmm"]["route_continuity"] == 1.0
    assert set(pred.route.tolist()) <= set(range(len(grid_trip["arcs"])))


def test_single_observation_trip(matcher, tmp_path):
    nodes, arcs, idx = grid_network()
    tr = _east_track(nodes, idx, n=1)
    d = write_trip(tmp_path, "single", tr, nodes, arcs)
    pred = matcher.predict(d)
    assert len(pred.hmm_arc) == 1 and pred.hmm_arc[0] >= 0


def test_all_fixes_far_from_network(matcher, tmp_path):
    nodes, arcs, idx = grid_network()
    tr = _east_track(nodes, idx)
    tr[:, 1] += 5000 * DEG                      # 5 km north of every road
    d = write_trip(tmp_path, "far", tr, nodes, arcs)
    pred = matcher.predict(d)
    assert (pred.hmm_arc == -1).all() and len(pred.route) == 0     # no fabricated route


def test_partial_dropout_far_fixes(matcher, tmp_path):
    nodes, arcs, idx = grid_network()
    tr = _east_track(nodes, idx, n=40)
    tr[15:20, 1] += 3000 * DEG                  # 5 wild fixes
    d = write_trip(tmp_path, "jump", tr, nodes, arcs)
    pred = matcher.predict(d)
    assert (pred.hmm_arc[15:20] == -1).all()
    assert (pred.hmm_arc[:15] >= 0).all() and (pred.hmm_arc[20:] >= 0).all()


def test_large_gap_and_irregular_timestamps(matcher, tmp_path):
    nodes, arcs, idx = grid_network()
    tr = _east_track(nodes, idx, n=40)
    tr[20:, 2] += 900                           # 15-minute gap
    tr[5, 2] = tr[4, 2] + 0.3                   # irregular sub-second step
    d = write_trip(tmp_path, "gap", tr, nodes, arcs)
    pred = matcher.predict(d)
    assert pred.diagnostics["n_gap_breaks"] >= 1
    assert (pred.hmm_arc >= 0).all()


def test_out_of_order_duplicate_and_invalid_rows(matcher, tmp_path):
    nodes, arcs, idx = grid_network()
    tr = _east_track(nodes, idx, n=30)
    tr = np.vstack([tr, tr[3:4]])               # duplicate timestamp
    tr[[7, 8]] = tr[[8, 7]]                     # out of order
    tr[10, 0] = 999.0                           # invalid longitude
    d = write_trip(tmp_path, "messy", tr, nodes, arcs)
    pred = matcher.predict(d)
    assert len(pred.hmm_arc) == 29              # 31 rows - 1 duplicate - 1 invalid
    assert pred.ctx.track.was_resorted and np.all(np.diff(pred.ctx.track.t) > 0)


def test_disconnected_network_components(matcher, tmp_path):
    """Two separate street pieces with no connection: decoder must break the chain, not invent a link."""
    nodes, arcs, idx = grid_network(n=6)
    keep = [a for a in arcs.tolist() if not ({a[0], a[1]} == {idx(2, 2), idx(2, 3)})]
    # remove every arc touching column 2->3 boundary on all rows to split the grid in two
    keep = [a for a in keep if not ((a[0] % 6 == 2 and a[1] % 6 == 3) or (a[0] % 6 == 3 and a[1] % 6 == 2))]
    d = write_trip(tmp_path, "split", _east_track(nodes, idx, n=60, noise=1.0), nodes, np.array(keep))
    pred = matcher.predict(d)
    assert pred.diagnostics["n_transition_breaks"] >= 1 or pred.unfilled_gaps >= 1
    # every consecutive pair inside the reconstructed route is either connected or a recorded gap
    r = pred.route
    net = pred.ctx.net
    bad = int((net.v[r[:-1]] != net.u[r[1:]]).sum())
    assert bad <= pred.unfilled_gaps


def test_dense_network(matcher, tmp_path):
    nodes, arcs, idx = grid_network(n=12, spacing_m=20)      # 20 m blocks: many candidates per fix
    lat = nodes[idx(5, 0)][1]
    n = 40
    lons = np.linspace(nodes[idx(5, 0)][0], nodes[idx(5, 11)][0], n)
    tr = np.column_stack([lons, np.full(n, lat), np.arange(1, n + 1)])
    d = write_trip(tmp_path, "dense", tr, nodes, arcs)
    pred = matcher.predict(d)
    assert (pred.hmm_arc >= 0).all()
    assert pred.ctx.cands.groupby("obs").size().max() <= 10


def test_route_file_absent_means_no_metrics(matcher, tmp_path):
    nodes, arcs, idx = grid_network()
    d = write_trip(tmp_path, "noref", _east_track(nodes, idx), nodes, arcs, route=None)
    pred = matcher.predict(d)
    m = RouteEvaluator().evaluate(d, pred)
    assert m == {"available": False, "reason": "no reference route for this trip on the server"}
