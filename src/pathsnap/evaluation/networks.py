"""Load road networks (projected exactly as during feature building) for route-level scoring."""
from __future__ import annotations

from ..data.dataset import partition_dir
from ..data.loader import load_trip
from ..data.preprocessing import clean_track
from ..geography.projection import LocalProjector
from ..geography.road_network import RoadNetwork


def load_network(partition: str, trip_id: str, with_route: bool = True):
    trip = load_trip(partition_dir(partition) / trip_id, with_route=with_route, partition=partition)
    ct = clean_track(trip.track)
    proj = LocalProjector.for_points(ct.lon, ct.lat)
    return trip, RoadNetwork(trip.nodes, trip.arcs, proj)
