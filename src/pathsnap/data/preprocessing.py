"""Track cleaning. The cleaned track keeps a pointer (``orig_index``) back to the
raw row so every prediction can be reported against the original observation."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .validator import valid_lonlat


@dataclass
class CleanTrack:
    lon: np.ndarray
    lat: np.ndarray
    t: np.ndarray
    orig_index: np.ndarray       # row in the raw .track file
    n_raw: int
    n_dropped_invalid: int
    n_dropped_duplicate: int
    was_resorted: bool

    def __len__(self) -> int:
        return len(self.t)


def clean_track(track: np.ndarray, drop_invalid: bool = True, sort: bool = True,
                drop_duplicate_ts: bool = True) -> CleanTrack:
    n_raw = len(track)
    idx = np.arange(n_raw)
    lon, lat, t = track[:, 0], track[:, 1], track[:, 2]
    keep = np.ones(n_raw, bool)
    if drop_invalid:
        keep &= valid_lonlat(lon, lat) & np.isfinite(t)
    n_invalid = int((~keep).sum())
    idx = idx[keep]
    resorted = False
    if sort and len(idx) > 1:
        order = np.argsort(t[idx], kind="stable")
        resorted = bool((order != np.arange(len(order))).any())
        idx = idx[order]
    n_dup = 0
    if drop_duplicate_ts and len(idx) > 1:
        tt = t[idx]
        first = np.concatenate([[True], tt[1:] != tt[:-1]])
        n_dup = int((~first).sum())
        idx = idx[first]
    return CleanTrack(lon[idx].copy(), lat[idx].copy(), t[idx].copy(), idx, n_raw,
                      n_invalid, n_dup, resorted)
