import itertools

import numpy as np
import pandas as pd

from pathsnap.decoding.emissions import log_emission
from pathsnap.decoding.hmm import HMMDecoder, classifier_only, reconstruct_route
from pathsnap.decoding.viterbi import viterbi_chain, viterbi_segmented
from pathsnap.geography.projection import LocalProjector
from pathsnap.geography.road_network import RoadNetwork

from tests.conftest import grid_network


def brute_force(log_em, trans):
    best, arg = -np.inf, None
    for path in itertools.product(*[range(len(e)) for e in log_em]):
        s = log_em[0][path[0]] + sum(trans(t)[path[t - 1], path[t]] + log_em[t][path[t]] for t in range(1, len(path)))
        if s > best:
            best, arg = s, path
    return np.array(arg)


def test_viterbi_matches_brute_force():
    rng = np.random.default_rng(1)
    for _ in range(20):
        sizes = rng.integers(1, 4, size=5)
        em = [rng.normal(size=k) for k in sizes]
        mats = {t: rng.normal(size=(sizes[t - 1], sizes[t])) for t in range(1, 5)}
        path = viterbi_chain(em, lambda t: mats[t])
        assert np.array_equal(path, brute_force(em, lambda t: mats[t]))


def test_empty_and_single():
    assert len(viterbi_chain([], lambda t: None)) == 0
    p, c, r = viterbi_segmented([], lambda t: None)
    assert len(p) == 0
    p, c, r = viterbi_segmented([np.array([-1.0, -0.1])], lambda t: None)
    assert p.tolist() == [1] and c.tolist() == [0]


def test_one_candidate_each_and_variable_counts():
    em = [np.array([0.0]), np.array([-1.0, -0.5, -3.0]), np.array([0.0])]
    T = {1: np.array([[0.0, -10.0, 0.0]]), 2: np.array([[0.0], [0.0], [0.0]])}
    p, _, _ = viterbi_segmented(em, lambda t: T[t])
    assert p.tolist() == [0, 0, 0]


def test_no_candidates_step_breaks_chain():
    em = [np.array([0.0]), np.array([]), np.array([0.0])]
    p, c, r = viterbi_segmented(em, lambda t: np.zeros((1, 1)))
    assert p.tolist() == [0, -1, 0]
    assert c[0] != c[2] and r[1] == "no_candidates"


def test_disconnected_transition_breaks_chain():
    em = [np.array([0.0]), np.array([0.0]), np.array([0.0])]
    T = {1: np.array([[0.0]]), 2: np.array([[-np.inf]])}
    p, c, r = viterbi_segmented(em, lambda t: T[t])
    assert p.tolist() == [0, 0, 0] and c.tolist() == [0, 0, 1] and r[2] == "no_valid_transition"


def test_forced_gap_break():
    em = [np.array([0.0])] * 4
    p, c, r = viterbi_segmented(em, lambda t: np.zeros((1, 1)), force_break=[False, False, True, False])
    assert c.tolist() == [0, 0, 1, 1] and r[2] == "gps_gap"


def test_zero_probabilities_are_floored():
    le = log_emission(np.array([0.0, 1.0, np.nan]), np.array([1.0, 1.0, 1.0]))
    assert np.isfinite(le).all() and le[1] == 0.0


def _setup_grid():
    nodes, arcs, idx = grid_network(n=4, spacing_m=100)
    net = RoadNetwork(nodes, arcs, LocalProjector.for_points(nodes[:, 0], nodes[:, 1]))
    a_id = {tuple(a): i for i, a in enumerate(arcs.tolist())}
    return net, idx, a_id


def test_hmm_prefers_connected_route_over_noisy_argmax():
    """Fix 2 is closer (and has higher P) on the parallel street; the HMM must keep the vehicle on its street."""
    net, idx, a_id = _setup_grid()
    main = [a_id[(idx(1, 0), idx(1, 1))], a_id[(idx(1, 1), idx(1, 2))]]
    parallel = a_id[(idx(2, 0), idx(2, 1))]
    x0, y0 = net.nx[idx(1, 0)], net.ny[idx(1, 0)]
    px = x0 + np.array([20.0, 50.0, 80.0, 120.0])
    py = y0 + np.array([2.0, 3.0, 2.0, 1.0])
    t = np.arange(4.0)
    rows = [(0, main[0], 2, .2, .9), (0, parallel, 98, .2, .1),
            (1, main[0], 3, .5, .4), (1, parallel, 97, .5, .6),      # classifier is wrong here
            (2, main[0], 2, .8, .9), (2, parallel, 98, .8, .1),
            (3, main[1], 1, .2, .9), (3, parallel, 99, 1.0, .1)]
    c = pd.DataFrame(rows, columns=["obs", "arc", "dist_m", "proj_frac", "p"])
    clf, _ = classifier_only(c, c["p"].to_numpy(), 4)
    assert clf[1] == parallel
    res = HMMDecoder({"beta_m": 5.0}).decode(net, c, c["p"].to_numpy(), px, py, t)
    assert res.arc.tolist() == [main[0], main[0], main[0], main[1]]
    route, inferred, gaps = reconstruct_route(net, res.arc, res.chain)
    assert route.tolist() == main and gaps == 0


def test_large_gap_breaks_chain_in_decoder():
    net, idx, a_id = _setup_grid()
    a = a_id[(idx(1, 0), idx(1, 1))]
    c = pd.DataFrame({"obs": [0, 1, 2], "arc": [a, a, a], "dist_m": [1.0, 1, 1], "proj_frac": [.1, .2, .3]})
    res = HMMDecoder({"max_gap_s": 60}).decode(net, c, np.array([.9, .9, .9]), np.array([10., 20, 30]),
                                                np.zeros(3), np.array([0., 1, 500]))
    assert res.diagnostics["n_gap_breaks"] == 1 and res.chain.tolist() == [0, 0, 1]


def test_impossible_speed_rejected():
    net, idx, a_id = _setup_grid()
    a = a_id[(idx(0, 0), idx(0, 1))]
    b = a_id[(idx(3, 2), idx(3, 3))]          # ~500 m away by road
    c = pd.DataFrame({"obs": [0, 1], "arc": [a, b], "dist_m": [1.0, 1.0], "proj_frac": [.5, .5]})
    res = HMMDecoder({"vmax_mps": 55}).decode(net, c, np.array([.9, .9]), np.array([0., 400]), np.array([0., 300]),
                                               np.array([0., 1.]))
    assert res.diagnostics["n_transition_breaks"] == 1        # 500 m in 1 s is impossible -> new chain, no fake path
