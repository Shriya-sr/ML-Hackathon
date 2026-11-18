"""Transition model between candidates of consecutive fixes (Newson & Krumm style).

For candidate a (fix t-1, at fraction f_a along its arc) and b (fix t, f_b):

    d_route = driving distance a -> b over the directed network (bounded Dijkstra)
    d_gps   = straight-line distance between the two GPS fixes
    log T(a, b) = -|d_route - d_gps| / beta  -  uturn_penalty * [b is reverse(a)]

Impossible transitions get -inf:
  * b not reachable from a within the search radius (disconnected / wrong way on a one-way arc)
  * implied speed d_route / dt above vmax
"""
from __future__ import annotations

import numpy as np


def transition_matrix(net, arcs_prev, frac_prev, arcs_cur, frac_cur, d_gps: float, dt: float,
                      beta_m: float = 5.0, uturn_penalty: float = 2.0, vmax_mps: float = 55.0,
                      min_limit_m: float = 150.0, limit_factor: float = 2.5, max_limit_m: float = 4000.0,
                      **_) -> np.ndarray:
    limit = min(max(min_limit_m, limit_factor * d_gps + 100.0), max_limit_m)
    if dt > 0:  # never search farther than the vehicle could physically drive
        limit = min(limit, max(min_limit_m, vmax_mps * dt + 50.0))
    T = np.full((len(arcs_prev), len(arcs_cur)), -np.inf)
    rev = net.reverse_arc
    for i, (a, fa) in enumerate(zip(arcs_prev, frac_prev)):
        for j, (b, fb) in enumerate(zip(arcs_cur, frac_cur)):
            d = net.network_distance(int(a), float(fa), int(b), float(fb), limit)
            if not np.isfinite(d):
                continue
            if dt > 0 and d / dt > vmax_mps and d > 30.0:
                continue
            v = -abs(d - d_gps) / beta_m
            if rev[a] == b:
                v -= uturn_penalty
            T[i, j] = v
    return T
