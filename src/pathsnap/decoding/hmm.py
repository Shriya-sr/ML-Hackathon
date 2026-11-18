"""HMM map matcher.

Hidden state at fix t  = one of the candidate arcs generated for that fix
                         (each state keeps its arc_id, projection fraction,
                         distance and classifier probability, so the
                         candidate -> arc correspondence is never lost).
Observation at fix t   = the GPS fix itself.
Emission               = decoding/emissions.py
Transition             = decoding/transitions.py
Decoding               = segmented Viterbi (decoding/viterbi.py)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .emissions import log_emission
from .transitions import transition_matrix
from .viterbi import viterbi_segmented

DEFAULT_PARAMS = dict(w_clf=1.0, sigma_m=10.0, w_dist=0.0, beta_m=5.0, uturn_penalty=2.0, vmax_mps=55.0,
                      max_gap_s=120.0, max_states=8, min_limit_m=150.0, limit_factor=2.5, max_limit_m=4000.0,
                      prob_floor=1e-6)


@dataclass
class DecodeResult:
    arc: np.ndarray            # per fix: chosen arc id or -1
    prob: np.ndarray           # classifier probability of the chosen candidate (nan if none)
    chain: np.ndarray          # chain id per fix (-1 if none)
    breaks: dict               # step -> reason
    seconds: float
    diagnostics: dict = field(default_factory=dict)


class HMMDecoder:
    def __init__(self, params: dict | None = None):
        self.p = {**DEFAULT_PARAMS, **(params or {})}

    def decode(self, net, cands: pd.DataFrame, prob: np.ndarray, px: np.ndarray, py: np.ndarray,
               t: np.ndarray) -> DecodeResult:
        t0 = time.perf_counter()
        p = self.p
        N = len(px)
        df = cands[["obs", "arc", "dist_m", "proj_frac"]].copy()
        df["prob"] = prob
        df["le"] = log_emission(df["prob"].to_numpy(), df["dist_m"].to_numpy(), p["w_clf"], p["w_dist"],
                                p["sigma_m"], p["prob_floor"])
        df = df.sort_values(["obs", "le"], ascending=[True, False], kind="mergesort")
        df = df[df.groupby("obs").cumcount() < int(p["max_states"])]
        groups = {o: g for o, g in df.groupby("obs", sort=True)}
        empty = df.iloc[0:0]
        states = [groups.get(i, empty) for i in range(N)]
        arcs = [s["arc"].to_numpy() for s in states]
        fracs = [s["proj_frac"].to_numpy() for s in states]
        log_em = [s["le"].to_numpy() for s in states]
        dts = np.concatenate([[0.0], np.diff(t)])
        gps_d = np.concatenate([[0.0], np.hypot(np.diff(px), np.diff(py))])
        force = dts > p["max_gap_s"]

        def trans(k: int) -> np.ndarray:
            return transition_matrix(net, arcs[k - 1], fracs[k - 1], arcs[k], fracs[k], float(gps_d[k]),
                                     float(dts[k]), **p)

        path, chain, reasons = viterbi_segmented(log_em, trans, force)
        out_arc = np.full(N, -1, np.int64)
        out_p = np.full(N, np.nan)
        for i in range(N):
            if path[i] >= 0:
                out_arc[i] = arcs[i][path[i]]
                out_p[i] = states[i]["prob"].to_numpy()[path[i]]
        diag = {"n_chains": int(chain.max() + 1) if (chain >= 0).any() else 0,
                "n_gap_breaks": sum(1 for r in reasons.values() if r == "gps_gap"),
                "n_transition_breaks": sum(1 for r in reasons.values() if r == "no_valid_transition"),
                "n_no_candidate_fixes": sum(1 for r in reasons.values() if r == "no_candidates")}
        return DecodeResult(out_arc, out_p, chain, reasons, time.perf_counter() - t0, diag)


def reconstruct_route(net, arc: np.ndarray, chain: np.ndarray, limit_m: float = 1500.0):
    """Turn per-fix decoded arcs into a drivable arc sequence.

    Inside one Viterbi chain, consecutive distinct arcs that are not adjacent are
    joined with the shortest network path (the same path the transition model
    scored). Across chain breaks nothing is invented: the route simply has a gap.
    Returns (route_arcs, inferred_flags, n_unfilled_gaps)."""
    route, inferred, gaps = [], [], 0
    prev_a, prev_c = -1, -1
    for a, c in zip(arc, chain):
        if a < 0:
            continue
        if a == prev_a and c == prev_c:
            continue
        if prev_a >= 0 and c == prev_c:
            mid = net.connecting_arcs(prev_a, int(a), limit_m)
            if mid is None:
                mid = net.connecting_arcs(prev_a, int(a), 4 * limit_m)
            if mid is None:
                gaps += 1
            else:
                route.extend(mid)
                inferred.extend([True] * len(mid))
        elif prev_a >= 0:
            gaps += 1
        route.append(int(a))
        inferred.append(False)
        prev_a, prev_c = int(a), c
    return np.asarray(route, np.int64), np.asarray(inferred, bool), gaps


def classifier_only(cands: pd.DataFrame, prob: np.ndarray, n_obs: int):
    """Independent per-fix argmax (the baseline the HMM is compared against)."""
    arc = np.full(n_obs, -1, np.int64)
    pr = np.full(n_obs, np.nan)
    if len(cands):
        df = cands[["obs", "arc"]].copy()
        df["prob"] = prob
        df = df.sort_values(["obs", "prob"], ascending=[True, False], kind="mergesort").drop_duplicates("obs")
        arc[df["obs"].to_numpy()] = df["arc"].to_numpy()
        pr[df["obs"].to_numpy()] = df["prob"].to_numpy()
    return arc, pr
