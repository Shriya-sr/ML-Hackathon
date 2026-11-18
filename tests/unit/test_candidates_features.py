import numpy as np
import pandas as pd
import pytest

from pathsnap.data.dataset import process_trip
from pathsnap.data.labeling import align_to_route
from pathsnap.data.loader import load_trip
from pathsnap.features.candidate_generation import attach_labels, candidate_coverage, generate_candidates
from pathsnap.features.feature_schema import FEATURE_COLUMNS, SchemaError, to_matrix
from pathsnap.geography.projection import LocalProjector
from pathsnap.geography.road_network import RoadNetwork

from tests.conftest import grid_network


def _net(n=6):
    nodes, arcs, idx = grid_network(n=n)
    proj = LocalProjector.for_points(nodes[:, 0], nodes[:, 1])
    return RoadNetwork(nodes, arcs, proj), idx, arcs


def test_candidates_within_radius_and_sorted():
    net, idx, _ = _net()
    x0, y0 = net.nx[idx(2, 1)], net.ny[idx(2, 1)]
    c = generate_candidates(np.array([x0 + 50]), np.array([y0 + 3]), net, radius_m=20, max_candidates=10,
                            fallback_radius_m=None)
    assert (c["dist_m"] <= 20 + 1e-6).all()
    assert c["dist_m"].is_monotonic_increasing
    assert len(c) == 2               # the two directions of the one street


def test_candidate_cap_and_no_duplicates():
    net, idx, _ = _net()
    x0, y0 = net.nx[idx(2, 2)], net.ny[idx(2, 2)]      # on an intersection: 8 arcs touch it
    c = generate_candidates(np.array([x0]), np.array([y0]), net, radius_m=5, max_candidates=3, fallback_radius_m=None)
    assert len(c) == 3
    assert not c.duplicated(["obs", "arc"]).any()


def test_no_candidates_and_fallback():
    net, idx, _ = _net()
    far_x, far_y = net.nx.max() + 120, net.ny.max() + 120
    none = generate_candidates(np.array([far_x]), np.array([far_y]), net, 50, 10, fallback_radius_m=None)
    assert len(none) == 0
    fb = generate_candidates(np.array([far_x]), np.array([far_y]), net, 50, 10, fallback_radius_m=200)
    assert len(fb) == 1
    too_far = generate_candidates(np.array([far_x + 1000]), np.array([far_y]), net, 50, 10, fallback_radius_m=200)
    assert len(too_far) == 0


def test_empty_track_and_empty_network():
    net, _, _ = _net()
    c = generate_candidates(np.array([]), np.array([]), net)
    assert len(c) == 0 and list(c.columns)[:2] == ["obs", "arc"]


def test_true_candidate_missing_is_counted():
    cands = pd.DataFrame({"obs": [0, 0, 1], "arc": [5, 6, 7]})
    labels = np.array([5, 9])          # fix 1's true arc 9 was not generated
    cov = candidate_coverage(cands, labels)
    assert cov["n_missed"] == 1 and cov["miss_rate"] == 0.5
    lab = attach_labels(cands, labels)
    assert lab["label"].tolist() == [1, 0, 0]


def test_labels_do_not_change_candidates(grid_trip):
    """Candidate generation must be identical with and without the reference route."""
    a = process_trip(load_trip(grid_trip["dir"], with_route=True))
    b = process_trip(load_trip(grid_trip["dir"], with_route=False), with_labels=False)
    pd.testing.assert_frame_equal(a.cands[["obs", "arc", "dist_m"]], b.cands[["obs", "arc", "dist_m"]])
    assert "label" not in b.cands.columns and "true_arc" not in b.cands.columns


def test_alignment_recovers_route(grid_trip):
    ctx = process_trip(load_trip(grid_trip["dir"]))
    assert (ctx.obs_labels >= 0).all()
    pos = ctx.route_pos
    assert (np.diff(pos) >= 0).all()                     # monotone
    assert set(ctx.obs_labels) <= set(grid_trip["route"])
    assert ctx.coverage["miss_rate"] == 0.0


def test_alignment_handles_loop():
    """Out-and-back on the same street: ordering must pick forward then reverse arcs."""
    net, idx, arcs = _net(n=3)
    a_id = {tuple(a): i for i, a in enumerate(arcs.tolist())}
    fwd = [a_id[(idx(0, 0), idx(0, 1))], a_id[(idx(0, 1), idx(0, 2))]]
    back = [a_id[(idx(0, 2), idx(0, 1))], a_id[(idx(0, 1), idx(0, 0))]]
    route = np.array(fwd + back)
    x = np.concatenate([np.linspace(10, 190, 10), np.linspace(190, 10, 10)]) + net.nx[idx(0, 0)]
    y = np.full(20, net.ny[idx(0, 0)] + 2.0)
    labels, pos, _ = align_to_route(x, y, net, route)
    # fixes 9 and 10 sit exactly on the turnaround where both directions are geometrically identical
    assert set(labels[:9]) <= set(fwd) and set(labels[11:]) <= set(back)


def test_feature_schema_and_matrix(grid_trip):
    ctx = process_trip(load_trip(grid_trip["dir"]))
    X = to_matrix(ctx.cands)
    assert X.shape[1] == len(FEATURE_COLUMNS)
    assert np.isfinite(X).all()
    # heading feature: driving east on an east-west street -> forward arc agrees
    best = ctx.cands[ctx.cands.label == 1]
    assert (best["cos_heading"] > 0.5).mean() > 0.8


def test_feature_matrix_errors():
    df = pd.DataFrame({c: [0.0] for c in FEATURE_COLUMNS})
    with pytest.raises(SchemaError, match="missing"):
        to_matrix(df.drop(columns=["dist_m"]))
    df.loc[0, "dist_m"] = np.inf
    with pytest.raises(SchemaError, match="non-finite"):
        to_matrix(df)


def test_feature_order_is_schema_order():
    df = pd.DataFrame({c: [float(i)] for i, c in enumerate(FEATURE_COLUMNS)})
    shuffled = df[FEATURE_COLUMNS[::-1]].assign(extra_unexpected=1.0)
    assert np.array_equal(to_matrix(shuffled)[0], np.arange(len(FEATURE_COLUMNS), dtype=float))
