import numpy as np
import pandas as pd

from pathsnap.evaluation.metrics import classification_metrics, select_top1, selection_macro_f1, top1_accuracy
from pathsnap.evaluation.route_metrics import collapse, continuity, lcs_length, levenshtein, route_metrics
from pathsnap.geography.projection import LocalProjector
from pathsnap.geography.road_network import RoadNetwork

from tests.conftest import grid_network


def ref_lev(a, b):
    d = np.zeros((len(a) + 1, len(b) + 1), int)
    d[:, 0] = range(len(a) + 1); d[0, :] = range(len(b) + 1)
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (a[i - 1] != b[j - 1]))
    return d[-1, -1]


def ref_lcs(a, b):
    d = np.zeros((len(a) + 1, len(b) + 1), int)
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            d[i, j] = d[i - 1, j - 1] + 1 if a[i - 1] == b[j - 1] else max(d[i - 1, j], d[i, j - 1])
    return d[-1, -1]


def test_levenshtein_and_lcs_against_reference():
    rng = np.random.default_rng(0)
    for _ in range(50):
        a = rng.integers(0, 5, rng.integers(0, 12)); b = rng.integers(0, 5, rng.integers(0, 12))
        assert levenshtein(a, b) == ref_lev(a, b)
        assert lcs_length(a, b) == ref_lcs(a, b)


def test_collapse():
    assert collapse([3, 3, -1, 3, 4, 4, -1]).tolist() == [3, 4]


def test_route_metrics_perfect_and_broken():
    nodes, arcs, idx = grid_network(n=3)
    net = RoadNetwork(nodes, arcs, LocalProjector.for_points(nodes[:, 0], nodes[:, 1]))
    a_id = {tuple(a): i for i, a in enumerate(arcs.tolist())}
    ref = [a_id[(idx(0, 0), idx(0, 1))], a_id[(idx(0, 1), idx(0, 2))], a_id[(idx(0, 2), idx(1, 2))]]
    m = route_metrics([ref[0], ref[0], ref[1], ref[2]], net, ref)
    assert m["route_continuity"] == 1.0 and m["route_f1"] == 1.0 and m["edit_similarity"] == 1.0
    m2 = route_metrics([ref[0], ref[2]], net, ref)                       # skipped the middle arc
    assert m2["invalid_transitions"] == 1 and m2["missing_segments"] == 1 and m2["lcs_recall"] < 1
    assert continuity(np.array([ref[0]]), net) == (1.0, 0)


def test_classification_and_top1():
    y = np.array([1, 0, 0, 1, 0]); p = np.array([.9, .2, .6, .4, .1])
    m = classification_metrics(y, p)
    assert m["tp"] == 1 and m["fp"] == 1 and m["fn"] == 1 and 0 < m["macro_f1"] < 1
    c = pd.DataFrame({"trip_id": "t", "obs": [0, 0, 1, 1], "arc": [5, 6, 7, 8], "true_arc": [5, 5, 8, 8],
                      "label": [1, 0, 0, 1]})
    pred = select_top1(c, np.array([.9, .1, .7, .3]))
    obs = pd.DataFrame({"trip_id": "t", "obs": [0, 1, 2], "label": [5, 8, 9]})    # fix 2 has no candidates
    r = top1_accuracy(obs, pred)
    assert r["n_correct"] == 1 and r["n_labelled_obs"] == 3 and r["n_no_prediction"] == 1
    assert 0 < selection_macro_f1(c, pred) < 1
