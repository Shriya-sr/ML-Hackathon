"""Shared trip-level comparison of classifier-only vs classifier+HMM decoding."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from ..decoding.hmm import HMMDecoder, classifier_only, reconstruct_route
from .metrics import selection_macro_f1
from .route_metrics import route_metrics


def _pred_frame(tid, arc):
    obs = np.arange(len(arc))
    m = arc >= 0
    return pd.DataFrame({"trip_id": tid, "obs": obs[m], "arc": arc[m]})


def error_types(pred: np.ndarray, lab: np.ndarray, net) -> dict:
    """Classify wrong per-fix decisions: reverse direction of the true arc, an arc
    sharing a node with it (junction/neighbour), no prediction, or another road."""
    m = (lab >= 0) & (pred != lab)
    p, t = pred[m], lab[m]
    none = p < 0
    pp = np.where(none, 0, p)
    rev = (~none) & (net.reverse_arc[t] == pp)
    adj = (~none) & (~rev) & ((net.u[pp] == net.u[t]) | (net.u[pp] == net.v[t]) | (net.v[pp] == net.u[t]) | (net.v[pp] == net.v[t]))
    return {"reverse": int(rev.sum()), "adjacent": int(adj.sum()), "none": int(none.sum()),
            "other": int((~none & ~rev & ~adj).sum())}


def compare_trip(tid, net, route, cands, prob, obs, hmm_params: dict) -> dict:
    """``obs`` must be the per-fix table of one trip sorted by obs (x, y, t, label)."""
    obs = obs.sort_values("obs")
    lab = obs["label"].to_numpy()
    m = lab >= 0
    px, py, t = obs["x"].to_numpy(), obs["y"].to_numpy(), obs["t"].to_numpy()
    t0 = time.perf_counter()
    a_clf, p_clf = classifier_only(cands, prob, len(obs))
    clf_s = time.perf_counter() - t0
    res = HMMDecoder(hmm_params).decode(net, cands, prob, px, py, t)
    t1 = time.perf_counter()
    route_hmm, inferred, unfilled = reconstruct_route(net, res.arc, res.chain)
    recon_s = time.perf_counter() - t1
    rm_clf = route_metrics(a_clf, net, route)
    rm_hmm = route_metrics(res.arc, net, route)
    rm_rec = route_metrics(route_hmm, net, route)
    cl = cands.assign(trip_id=tid)
    out = {
        "trip_id": tid, "n_fixes": int(len(obs)), "n_labelled": int(m.sum()),
        "miss_rate": float(1 - obs.loc[m, "true_in_candidates"].mean()) if "true_in_candidates" in obs else np.nan,
        "clf_top1": float((a_clf[m] == lab[m]).mean()), "hmm_top1": float((res.arc[m] == lab[m]).mean()),
        "clf_sel_macro_f1": selection_macro_f1(cl, _pred_frame(tid, a_clf)),
        "hmm_sel_macro_f1": selection_macro_f1(cl, _pred_frame(tid, res.arc)),
        "clf_route_continuity": rm_clf["route_continuity"], "hmm_route_continuity": rm_hmm["route_continuity"],
        "recon_route_continuity": rm_rec["route_continuity"],
        "clf_invalid_transitions": rm_clf["invalid_transitions"], "hmm_invalid_transitions": rm_hmm["invalid_transitions"],
        "recon_invalid_transitions": rm_rec["invalid_transitions"],
        "clf_route_f1": rm_clf["route_f1"], "hmm_route_f1": rm_hmm["route_f1"], "recon_route_f1": rm_rec["route_f1"],
        "clf_route_precision": rm_clf["route_precision"], "recon_route_precision": rm_rec["route_precision"],
        "clf_route_recall": rm_clf["route_recall"], "recon_route_recall": rm_rec["route_recall"],
        "clf_edit_similarity": rm_clf["edit_similarity"], "hmm_edit_similarity": rm_hmm["edit_similarity"],
        "recon_edit_similarity": rm_rec["edit_similarity"],
        "clf_lcs_recall": rm_clf["lcs_recall"], "recon_lcs_recall": rm_rec["lcs_recall"],
        "clf_missing_segments": rm_clf["missing_segments"], "recon_missing_segments": rm_rec["missing_segments"],
        "clf_extra_segments": rm_clf["extra_segments"], "recon_extra_segments": rm_rec["extra_segments"],
        "ref_continuity": rm_rec["ref_continuity"],
        "hmm_chains": res.diagnostics["n_chains"], "hmm_gap_breaks": res.diagnostics["n_gap_breaks"],
        "hmm_transition_breaks": res.diagnostics["n_transition_breaks"],
        "recon_unfilled_gaps": unfilled, "recon_inferred_arcs": int(inferred.sum()),
        "clf_seconds": clf_s, "hmm_seconds": res.seconds, "recon_seconds": recon_s,
        "n_changed_by_hmm": int((a_clf != res.arc).sum()),
        "n_fixed_by_hmm": int(((a_clf != lab) & (res.arc == lab) & m).sum()),
        "n_broken_by_hmm": int(((a_clf == lab) & (res.arc != lab) & m).sum()),
    }
    for name, arr in (("clf", a_clf), ("hmm", res.arc)):
        for k, v in error_types(arr, lab, net).items():
            out[f"{name}_err_{k}_by_hmm"] = v       # suffix -> summed by pooled()
    return out


COUNT_SUFFIXES = ("_seconds", "_breaks", "_segments", "_transitions", "_arcs", "_gaps", "_chains", "_by_hmm")


def pooled(df: pd.DataFrame) -> dict:
    """Aggregate per-trip rows: accuracies weighted by labelled fixes (pooled over
    fixes), counts summed, route-level ratios averaged over trips."""
    w = df["n_labelled"].to_numpy()
    out = {"n_trips": int(len(df)), "n_fixes": int(df["n_fixes"].sum()), "n_labelled": int(w.sum())}
    for c in df.columns:
        if c in ("trip_id", "n_fixes", "n_labelled") or df[c].dtype.kind not in "fiu":
            continue
        if c.endswith("_top1") or c == "miss_rate":
            out[c] = float(np.average(df[c], weights=w))
        elif c.endswith(COUNT_SUFFIXES):
            out[c] = float(df[c].sum())
        else:
            out[c] = float(df[c].mean())
    return out
