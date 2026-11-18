import numpy as np
import pytest

from pathsnap.data.loader import DataFormatError, load_trip, read_arcs, read_route, read_track
from pathsnap.data.preprocessing import clean_track
from pathsnap.data.validator import validate_trip


def test_load_valid_trip(grid_trip):
    t = load_trip(grid_trip["dir"])
    assert t.track.shape == (50, 3)
    assert t.arcs.dtype == np.int64
    assert list(t.route) == grid_trip["route"]


def test_inference_load_never_reads_route(grid_trip):
    t = load_trip(grid_trip["dir"], with_route=False)
    assert t.route is None


def test_missing_file(grid_trip):
    (grid_trip["dir"] / "00000001.nodes").unlink()
    with pytest.raises(DataFormatError, match="missing file"):
        load_trip(grid_trip["dir"])


def test_missing_folder(tmp_path):
    with pytest.raises(DataFormatError, match="not found"):
        load_trip(tmp_path / "nope")


def test_empty_file(tmp_path):
    p = tmp_path / "x.track"
    p.write_text("")
    with pytest.raises(DataFormatError, match="empty"):
        read_track(p)


def test_wrong_column_count(tmp_path):
    p = tmp_path / "x.track"
    p.write_text("1 2\n3 4\n")
    with pytest.raises(DataFormatError, match="expected 3 columns"):
        read_track(p)


def test_ragged_rows(tmp_path):
    p = tmp_path / "x.track"
    p.write_text("1 2 3\n4 5\n6 7 8 9\n")
    with pytest.raises(DataFormatError):
        read_track(p)


def test_non_numeric_track_becomes_nan_and_is_dropped(tmp_path):
    p = tmp_path / "x.track"
    p.write_text("10.0 50.0 1\nabc 50.0 2\n10.1 50.1 3\n")
    tr = read_track(p)
    assert np.isnan(tr[1, 0])
    ct = clean_track(tr)
    assert len(ct) == 2 and ct.n_dropped_invalid == 1
    assert list(ct.orig_index) == [0, 2]


def test_non_integer_arcs(tmp_path):
    p = tmp_path / "x.arcs"
    p.write_text("0 1\n1 2.5\n")
    with pytest.raises(DataFormatError, match="non-integer"):
        read_arcs(p)


def test_arc_references_unknown_node(tmp_path):
    p = tmp_path / "x.arcs"
    p.write_text("0 1\n1 7\n")
    with pytest.raises(DataFormatError, match="outside"):
        read_arcs(p, n_nodes=3)


def test_invalid_route_arc_id(tmp_path):
    p = tmp_path / "x.route"
    p.write_text("0\n1\n99\n")
    with pytest.raises(DataFormatError, match="outside"):
        read_route(p, n_arcs=5)


def test_invalid_coordinates_dropped():
    tr = np.array([[10, 50, 1], [200, 50, 2], [10, 95, 3], [np.nan, 50, 4], [10.001, 50, 5]], float)
    ct = clean_track(tr)
    assert len(ct) == 2 and ct.n_dropped_invalid == 3


def test_out_of_order_timestamps_resorted():
    tr = np.array([[10, 50, 3], [10, 50.001, 1], [10, 50.002, 2]], float)
    ct = clean_track(tr)
    assert list(ct.t) == [1, 2, 3] and ct.was_resorted
    assert list(ct.orig_index) == [1, 2, 0]


def test_duplicate_timestamps_keep_first():
    tr = np.array([[10, 50, 1], [10.5, 50, 1], [10, 50.001, 2]], float)
    ct = clean_track(tr)
    assert len(ct) == 2 and ct.n_dropped_duplicate == 1 and ct.lon[0] == 10


def test_irregular_and_nonpositive_dt_flagged(grid_trip):
    t = load_trip(grid_trip["dir"])
    t.track[10, 2] = t.track[9, 2]      # zero dt
    t.track[20:, 2] += 500              # big gap
    v = validate_trip(t, gap_threshold_s=30)
    assert v["n_nonpositive_dt"] == 1
    assert v["n_gaps_over_threshold"] == 1
    assert not v["timestamps_monotonic"]


def test_single_observation():
    ct = clean_track(np.array([[10.0, 50.0, 1.0]]))
    assert len(ct) == 1


def test_validator_network_and_route(grid_trip):
    v = validate_trip(load_trip(grid_trip["dir"]))
    assert v["n_isolated_nodes"] == 0
    assert v["n_weak_components"] == 1
    assert v["route_continuity"] == 1.0
    assert v["two_way_arc_frac"] == 1.0
