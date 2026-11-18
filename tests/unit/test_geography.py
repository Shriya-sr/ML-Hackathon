import numpy as np

from pathsnap.geography.distance import angle_diff, bearing_xy, haversine_m, point_segment
from pathsnap.geography.projection import LocalProjector, utm_epsg
from pathsnap.geography.road_network import RoadNetwork

from tests.conftest import grid_network


def test_point_segment_basic():
    d, t = point_segment(np.array([5.0]), np.array([3.0]), 0.0, 0.0, 10.0, 0.0)
    assert np.isclose(d[0], 3.0) and np.isclose(t[0], 0.5)


def test_point_segment_beyond_end_clips():
    d, t = point_segment(np.array([14.0]), np.array([3.0]), 0.0, 0.0, 10.0, 0.0)
    assert np.isclose(d[0], 5.0) and t[0] == 1.0


def test_point_segment_zero_length():
    d, t = point_segment(np.array([3.0]), np.array([4.0]), 0.0, 0.0, 0.0, 0.0)
    assert np.isclose(d[0], 5.0) and t[0] == 0.0


def test_bearing_and_angle_diff():
    assert np.isclose(bearing_xy(0, 1), 0) and np.isclose(bearing_xy(1, 0), 90)
    assert np.isclose(angle_diff(350, 10), 20) and np.isclose(angle_diff(0, 180), 180)


def test_utm_zones():
    assert utm_epsg(8.6, 49.9) == 32632        # Germany, north
    assert utm_epsg(-58.4, -34.6) == 32721     # Buenos Aires, south
    assert utm_epsg(-93.4, 44.8) == 32615


def test_projection_matches_haversine():
    lon = np.array([8.58, 8.60]); lat = np.array([49.85, 49.86])
    p = LocalProjector.for_points(lon, lat)
    x, y = p.to_xy(lon, lat)
    planar = np.hypot(x[1] - x[0], y[1] - y[0])
    geo = haversine_m(lon[0], lat[0], lon[1], lat[1])
    assert abs(planar - geo) / geo < 0.002
    lo, la = p.to_lonlat(x, y)
    assert np.allclose(lo, lon, atol=1e-9) and np.allclose(la, lat, atol=1e-9)


def test_road_network_reverse_and_paths():
    nodes, arcs, idx = grid_network(n=3, spacing_m=100)
    net = RoadNetwork(nodes, arcs, LocalProjector.for_points(nodes[:, 0], nodes[:, 1]))
    assert (net.reverse_arc >= 0).all()
    assert np.allclose(net.length, 100, atol=0.5)
    a_id = {tuple(a): i for i, a in enumerate(arcs.tolist())}
    a = a_id[(idx(0, 0), idx(0, 1))]
    b = a_id[(idx(0, 2), idx(1, 2))]
    # from middle of a to middle of b: 50 + 100 + 50
    d = net.network_distance(a, 0.5, b, 0.5, limit=1000)
    assert abs(d - 200) < 1.0
    assert net.connecting_arcs(a, b) == [a_id[(idx(0, 1), idx(0, 2))]]
    assert net.network_distance(a, 0.5, b, 0.5, limit=100) == np.inf


def test_one_way_unreachable():
    nodes = np.array([[10, 50], [10.001, 50], [10.002, 50]])
    arcs = np.array([[0, 1], [2, 1]])           # both point INTO node 1
    net = RoadNetwork(nodes, arcs, LocalProjector(10, 50))
    assert net.network_distance(0, 0.5, 1, 0.5, 1000) == np.inf
    assert net.connecting_arcs(0, 1) is None
