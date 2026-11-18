"""Candidate road-segment generation.

For each cleaned GPS fix we return up to K arcs whose straight segment lies
within ``radius_m`` (STRtree query), ordered by perpendicular distance. If no arc
is in range, the single nearest arc within ``fallback_radius_m`` is used; beyond
that the fix gets no candidates (and later no prediction).

Candidate generation never sees the reference route: labels are attached
afterwards in ``attach_labels`` purely by checking membership.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..geography.distance import point_segment


def generate_candidates(px: np.ndarray, py: np.ndarray, net, radius_m: float = 50.0,
                        max_candidates: int = 10, fallback_radius_m: float | None = 150.0) -> pd.DataFrame:
    N = len(px)
    cols = ["obs", "arc", "dist_m", "proj_frac"]
    if N == 0 or net.E == 0:
        return pd.DataFrame({c: np.array([], dtype=float) for c in cols}).astype({"obs": np.int64, "arc": np.int64})
    pi, si = net.index.query_radius(px, py, radius_m)
    if fallback_radius_m and fallback_radius_m > radius_m:
        missing = np.setdiff1d(np.arange(N), pi)
        if len(missing):
            fi, fs = net.index.nearest(px[missing], py[missing], max_distance=fallback_radius_m)
            pi = np.concatenate([pi, missing[fi]])
            si = np.concatenate([si, fs])
    if len(pi) == 0:
        return pd.DataFrame({c: np.array([], dtype=float) for c in cols}).astype({"obs": np.int64, "arc": np.int64})
    d, t = point_segment(px[pi], py[pi], net.ax[si], net.ay[si], net.bx[si], net.by[si])
    df = pd.DataFrame({"obs": pi, "arc": si, "dist_m": d, "proj_frac": t})
    df = df.drop_duplicates(["obs", "arc"])
    df = df.sort_values(["obs", "dist_m", "arc"], kind="mergesort")
    df["dist_rank"] = df.groupby("obs").cumcount()
    df = df[df["dist_rank"] < max_candidates].reset_index(drop=True)
    return df


def attach_labels(cands: pd.DataFrame, obs_labels: np.ndarray) -> pd.DataFrame:
    """Mark candidates that equal the observation's reference arc (supervision only)."""
    cands = cands.copy()
    true_arc = obs_labels[cands["obs"].to_numpy()]
    cands["true_arc"] = true_arc
    cands["label"] = (cands["arc"].to_numpy() == true_arc).astype(np.int8)
    return cands


def candidate_coverage(cands: pd.DataFrame, obs_labels: np.ndarray) -> dict:
    """Candidate miss rate over observations that have a reliable reference arc."""
    labelled = np.where(obs_labels >= 0)[0]
    if len(labelled) == 0:
        return {"n_labelled": 0, "n_missed": 0, "miss_rate": float("nan"), "n_no_candidates": 0}
    hit = np.zeros(len(obs_labels), bool)
    sub = cands[cands["arc"].to_numpy() == obs_labels[cands["obs"].to_numpy()]]
    hit[sub["obs"].to_numpy()] = True
    has = np.zeros(len(obs_labels), bool)
    has[cands["obs"].to_numpy()] = True
    missed = int((~hit[labelled]).sum())
    return {"n_labelled": int(len(labelled)), "n_missed": missed,
            "miss_rate": missed / len(labelled),
            "n_no_candidates": int((~has[labelled]).sum())}
