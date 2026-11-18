"""Spatial index over straight road segments (one per arc).

Shapely's STRtree (an R-tree packed with the Sort-Tile-Recursive algorithm)
answers "which segments lie within R metres of each point" for all points of a
trip in one vectorised call, so we never scan the full network per GPS point.
"""
from __future__ import annotations

import numpy as np
import shapely
from shapely import STRtree


class SegmentIndex:
    def __init__(self, ax, ay, bx, by):
        coords = np.stack([np.stack([ax, ay], 1), np.stack([bx, by], 1)], axis=1)  # (E,2,2)
        self.geoms = shapely.linestrings(coords)
        self.tree = STRtree(self.geoms)

    def query_radius(self, px: np.ndarray, py: np.ndarray, radius: float):
        """Return (point_idx, segment_idx) pairs whose distance is <= radius."""
        pts = shapely.points(px, py)
        pairs = self.tree.query(pts, predicate="dwithin", distance=float(radius))
        return pairs[0].astype(np.int64), pairs[1].astype(np.int64)

    def nearest(self, px: np.ndarray, py: np.ndarray, max_distance: float | None = None):
        pts = shapely.points(px, py)
        res = self.tree.query_nearest(pts, max_distance=max_distance, all_matches=False)
        return res[0].astype(np.int64), res[1].astype(np.int64)
