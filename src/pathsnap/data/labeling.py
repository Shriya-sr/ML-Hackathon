"""Derive one reference arc per GPS observation from the ordered ``.route``.

The ``.route`` file lists arcs in driving order, but it has far fewer entries
than there are GPS fixes (about one arc per six fixes), so there is no
row-to-row correspondence. We recover per-observation labels with a *monotone
alignment*: observation i is assigned route position j_i such that
j_0 <= j_1 <= ... (the vehicle never goes backwards along its own route) and the
total capped point-to-segment distance is minimal. This is solved exactly by
dynamic programming:

    C[i, j] = d(i, j) + min_{j' <= j} C[i-1, j']

which costs O(N*M) time. Positions may be skipped (short arcs passed between two
fixes, or GPS gaps). Revisited arcs in looping routes are disambiguated by the
ordering constraint.

These labels are used ONLY to (a) mark which generated candidates are correct
for supervised training and (b) score predictions server-side. They never
influence candidate generation or inference.
"""
from __future__ import annotations

import numpy as np

from ..geography.distance import point_segment


def _motion_unit(px, py, w=2, min_disp=3.0):
    n = len(px)
    i = np.arange(n)
    lo, hi = np.clip(i - w, 0, n - 1), np.clip(i + w, 0, n - 1)
    dx, dy = px[hi] - px[lo], py[hi] - py[lo]
    d = np.hypot(dx, dy)
    ok = d >= min_disp
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(ok, dx / d, 0.0), np.where(ok, dy / d, 0.0)


def align_to_route(px: np.ndarray, py: np.ndarray, net, route: np.ndarray,
                   distance_cap_m: float = 100.0, max_label_distance_m: float = 60.0,
                   heading_weight_m: float = 15.0, chunk: int = 512):
    """Return (labels, route_pos, dist) for every observation.

    Cost of assigning fix i to route position j:
        min(d(i,j), cap) + heading_weight * (1 - cos(motion_i, arc_j)) / 2
    The heading term (0 when moving along the arc, ``heading_weight`` when moving
    against it, 0 when the vehicle is stationary) separates a road's two
    directions, which are geometrically identical, on out-and-back routes.

    labels[i]   arc id, or -1 when the aligned arc is farther than
                ``max_label_distance_m`` (reference unreliable for that fix)
    route_pos[i] index into ``route``
    dist[i]      distance (m) from the fix to its aligned arc
    """
    N, M = len(px), len(route)
    if N == 0 or M == 0:
        return (np.full(N, -1, np.int64), np.full(N, -1, np.int64), np.full(N, np.inf))
    ax, ay, bx, by = net.ax[route], net.ay[route], net.bx[route], net.by[route]
    L = np.maximum(np.hypot(bx - ax, by - ay), 1e-9)
    ux, uy = (bx - ax) / L, (by - ay) / L
    mx, my = _motion_unit(px, py)
    back = np.empty((N, M), dtype=np.int16 if M < 32000 else np.int32)
    ar = np.arange(M)
    C = None
    raw_d = np.empty(N)
    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        D, _ = point_segment(px[s:e, None], py[s:e, None], ax[None], ay[None], bx[None], by[None])
        cos = mx[s:e, None] * ux[None] + my[s:e, None] * uy[None]
        moving = (mx[s:e] != 0) | (my[s:e] != 0)
        Dc = np.minimum(D, distance_cap_m) + np.where(moving[:, None], heading_weight_m * (1 - cos) / 2, 0.0)
        for k in range(e - s):
            i = s + k
            if C is None:
                C = Dc[k].astype(np.float64)
                back[i] = ar
                continue
            cm = np.minimum.accumulate(C)
            # first index attaining each prefix minimum -> ties keep the vehicle on the earlier route arc
            is_new = C < np.concatenate([[np.inf], cm[:-1]])
            back[i] = np.maximum.accumulate(np.where(is_new, ar, 0))
            C = Dc[k] + cm
    pos = np.empty(N, np.int64)
    pos[-1] = int(np.argmin(C))
    for i in range(N - 1, 0, -1):
        pos[i - 1] = back[i, pos[i]]
    d, _ = point_segment(px, py, ax[pos], ay[pos], bx[pos], by[pos])
    raw_d[:] = d
    labels = route[pos].astype(np.int64)
    labels[raw_d > max_label_distance_m] = -1
    return labels, pos, raw_d
