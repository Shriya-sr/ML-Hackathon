"""Per-trip metric projection.

The trips are scattered world-wide (longitudes -100 to 137, latitudes -35 to 62),
but each individual trip spans at most a few tens of kilometres. We therefore
project every trip into the UTM zone of its own centroid (WGS84 / EPSG:326xx
north, 327xx south). Inside one UTM zone the scale error is < 0.1 %, i.e.
< 1 cm per 10 m, far below GPS noise. Degree-space Euclidean distances are never
used for physical quantities.
"""
from __future__ import annotations

import numpy as np
from pyproj import CRS, Transformer


def utm_epsg(lon: float, lat: float) -> int:
    zone = int(np.floor((lon + 180.0) / 6.0)) % 60 + 1
    return (32600 if lat >= 0 else 32700) + zone


class LocalProjector:
    """lon/lat <-> metres in the trip's UTM zone."""

    def __init__(self, lon0: float, lat0: float):
        self.epsg = utm_epsg(lon0, lat0)
        crs = CRS.from_epsg(self.epsg)
        self._fwd = Transformer.from_crs(CRS.from_epsg(4326), crs, always_xy=True)
        self._inv = Transformer.from_crs(crs, CRS.from_epsg(4326), always_xy=True)

    @classmethod
    def for_points(cls, lon: np.ndarray, lat: np.ndarray) -> "LocalProjector":
        return cls(float(np.nanmedian(lon)), float(np.nanmedian(lat)))

    def to_xy(self, lon, lat):
        x, y = self._fwd.transform(np.asarray(lon, float), np.asarray(lat, float))
        return np.asarray(x), np.asarray(y)

    def to_lonlat(self, x, y):
        lon, lat = self._inv.transform(np.asarray(x, float), np.asarray(y, float))
        return np.asarray(lon), np.asarray(lat)
