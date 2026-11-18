"""Per-candidate feature computation (schema in feature_schema.py).

All features are functions of the GPS track and road network only; nothing is
derived from the reference route. Nothing here is *learned*, so there is no
preprocessing state that could leak across partitions; scaling for the
scale-sensitive models is fitted inside each model's sklearn Pipeline on
training folds only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..geography.distance import angle_diff, bearing_xy, point_segment


def kinematics(px, py, t, heading_window=2, min_heading_disp_m=3.0, long_window=5, min_long_disp_m=5.0):
    """Per-observation heading/speed estimates from neighbouring fixes."""
    N = len(px)
    i = np.arange(N)
    lo, hi = np.clip(i - heading_window, 0, N - 1), np.clip(i + heading_window, 0, N - 1)
    dx, dy = px[hi] - px[lo], py[hi] - py[lo]
    disp = np.hypot(dx, dy)
    dt = t[hi] - t[lo]
    heading = bearing_xy(dx, dy)
    heading_valid = disp >= min_heading_disp_m
    with np.errstate(divide="ignore", invalid="ignore"):
        speed = np.where(dt > 0, disp / dt, 0.0)
    lo2, hi2 = np.clip(i - long_window, 0, N - 1), np.clip(i + long_window, 0, N - 1)
    ldx, ldy = px[hi2] - px[lo2], py[hi2] - py[lo2]
    long_bearing = bearing_xy(ldx, ldy)
    long_valid = np.hypot(ldx, ldy) >= min_long_disp_m
    dt_prev = np.concatenate([[0.0], np.diff(t)]) if N else np.array([])
    dt_next = np.concatenate([np.diff(t), [0.0]]) if N else np.array([])
    return {"heading": heading, "heading_valid": heading_valid, "speed": speed,
            "long_bearing": long_bearing, "long_valid": long_valid,
            "dt_prev": dt_prev, "dt_next": dt_next}


def extract_features(cands: pd.DataFrame, px, py, t, net, cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or {}
    kin_cfg = {k: cfg[k] for k in ("heading_window", "min_heading_disp_m", "long_window", "min_long_disp_m") if k in cfg}
    w = int(cfg.get("context_window", 3))
    kin = kinematics(px, py, t, **kin_cfg)
    df = cands.copy()
    if len(df) == 0:
        for c in ["dist_minus_min", "dist_ratio", "n_candidates", "at_endpoint", "endpoint_dist_m",
                  "arc_length_m", "cos_heading", "heading_diff_deg", "heading_valid", "cos_long_bearing",
                  "win_mean_dist_m", "win_max_dist_m", "speed_mps", "dt_prev", "dt_next",
                  "from_in_degree", "to_out_degree", "is_two_way"]:
            df[c] = np.array([], dtype=float)
        return df
    o = df["obs"].to_numpy()
    a = df["arc"].to_numpy()
    g = df.groupby("obs")["dist_m"]
    dmin = g.transform("min").to_numpy()
    df["dist_minus_min"] = df["dist_m"].to_numpy() - dmin
    df["dist_ratio"] = (df["dist_m"].to_numpy() + 1.0) / (dmin + 1.0)
    df["n_candidates"] = g.transform("size").to_numpy().astype(float)
    frac = df["proj_frac"].to_numpy()
    L = net.length[a]
    df["at_endpoint"] = ((frac <= 1e-9) | (frac >= 1 - 1e-9)).astype(float)
    df["endpoint_dist_m"] = np.minimum(frac, 1 - frac) * L
    df["arc_length_m"] = L

    hv = kin["heading_valid"][o]
    diff = angle_diff(kin["heading"][o], net.bearing[a])
    df["cos_heading"] = np.where(hv, np.cos(np.radians(diff)), 0.0)
    df["heading_diff_deg"] = np.where(hv, diff, 90.0)
    df["heading_valid"] = hv.astype(float)
    lv = kin["long_valid"][o]
    ldiff = angle_diff(kin["long_bearing"][o], net.bearing[a])
    df["cos_long_bearing"] = np.where(lv, np.cos(np.radians(ldiff)), 0.0)

    N = len(px)
    offs = np.arange(-w, w + 1)
    nb = np.clip(o[:, None] + offs[None, :], 0, N - 1)
    wd, _ = point_segment(px[nb], py[nb], net.ax[a][:, None], net.ay[a][:, None],
                          net.bx[a][:, None], net.by[a][:, None])
    df["win_mean_dist_m"] = wd.mean(1)
    df["win_max_dist_m"] = wd.max(1)

    df["speed_mps"] = kin["speed"][o]
    df["dt_prev"] = kin["dt_prev"][o]
    df["dt_next"] = kin["dt_next"][o]
    df["from_in_degree"] = net.in_degree[net.u[a]].astype(float)
    df["to_out_degree"] = net.out_degree[net.v[a]].astype(float)
    df["is_two_way"] = (net.reverse_arc[a] >= 0).astype(float)
    df["dist_rank"] = df["dist_rank"].astype(float)
    return df
