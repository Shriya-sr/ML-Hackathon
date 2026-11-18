"""Emission model: how well does candidate c explain GPS fix t?

    log E_t(c) = w_clf * log max(P_clf(c), floor)  +  w_dist * ( -0.5 * (d_tc / sigma)^2 )

P_clf is the selected classifier's calibrated probability that c is the
reference arc; d_tc is the perpendicular GPS-to-segment distance (metres). The
floor keeps the log finite so a candidate the classifier dislikes can still be
chosen if the route demands it.
"""
from __future__ import annotations

import numpy as np


def log_emission(prob: np.ndarray, dist_m: np.ndarray, w_clf: float = 1.0, w_dist: float = 0.0,
                 sigma_m: float = 10.0, prob_floor: float = 1e-6) -> np.ndarray:
    p = np.clip(np.nan_to_num(np.asarray(prob, float), nan=prob_floor), prob_floor, 1.0)
    out = w_clf * np.log(p)
    if w_dist:
        out = out - w_dist * 0.5 * (np.asarray(dist_m, float) / sigma_m) ** 2
    return out
