"""Trip -> processed context -> candidate feature table.

``process_trip`` is the single code path used by training, evaluation and the
API. ``with_labels`` controls whether the reference route is used to attach
supervision; the inference path calls it with ``with_labels=False`` and a Trip
loaded without its .route file.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import data_dir, load_config
from ..features.candidate_generation import attach_labels, candidate_coverage, generate_candidates
from ..features.feature_extraction import extract_features
from ..geography.projection import LocalProjector
from ..geography.road_network import RoadNetwork
from .labeling import align_to_route
from .loader import Trip, list_trips, load_trip
from .preprocessing import CleanTrack, clean_track


@dataclass
class TripContext:
    trip: Trip
    track: CleanTrack
    projector: LocalProjector
    net: RoadNetwork
    px: np.ndarray
    py: np.ndarray
    cands: pd.DataFrame
    obs_labels: np.ndarray | None = None      # per cleaned observation (server-side only)
    route_pos: np.ndarray | None = None
    label_dist: np.ndarray | None = None
    coverage: dict | None = None
    timings: dict = field(default_factory=dict)


def process_trip(trip: Trip, with_labels: bool = True, feat_cfg: dict | None = None,
                 data_cfg: dict | None = None) -> TripContext:
    feat_cfg = feat_cfg or load_config("features")
    data_cfg = data_cfg or load_config("data")
    timings = {}
    t0 = time.perf_counter()
    pp = data_cfg["preprocessing"]
    ct = clean_track(trip.track, pp["drop_invalid_coordinates"], pp["sort_by_timestamp"],
                     pp["drop_duplicate_timestamps"])
    if len(ct) == 0:
        raise ValueError(f"trip {trip.trip_id}: no valid GPS observations after cleaning")
    proj = LocalProjector.for_points(ct.lon, ct.lat)
    net = RoadNetwork(trip.nodes, trip.arcs, proj)
    px, py = proj.to_xy(ct.lon, ct.lat)
    timings["preprocess_s"] = time.perf_counter() - t0

    t1 = time.perf_counter()
    cc = feat_cfg["candidates"]
    cands = generate_candidates(px, py, net, cc["radius_m"], cc["max_candidates"], cc.get("fallback_radius_m"))
    timings["candidates_s"] = time.perf_counter() - t1

    t2 = time.perf_counter()
    feats = extract_features(cands, px, py, ct.t, net, feat_cfg.get("kinematics", {}))
    feats.insert(0, "trip_id", trip.trip_id)
    timings["features_s"] = time.perf_counter() - t2

    ctx = TripContext(trip, ct, proj, net, px, py, feats, timings=timings)
    if with_labels and trip.route is not None:
        t3 = time.perf_counter()
        lc = data_cfg["labelling"]
        labels, pos, dist = align_to_route(px, py, net, trip.route, lc["distance_cap_m"], lc["max_label_distance_m"])
        ctx.obs_labels, ctx.route_pos, ctx.label_dist = labels, pos, dist
        ctx.cands = attach_labels(feats, labels)
        ctx.coverage = candidate_coverage(feats, labels)
        timings["labelling_s"] = time.perf_counter() - t3
    return ctx


def partition_dir(partition: str) -> Path:
    cfg = load_config("data")["partitions"]
    return data_dir() / cfg.get(partition, partition)


def iter_partition(partition: str, with_route: bool = True):
    root = partition_dir(partition)
    for tid in list_trips(root):
        yield load_trip(root / tid, with_route=with_route, partition=partition)
