"""Route-level (sequence) evaluation.

Definitions (also documented in docs/methodology.md):

* predicted route  = per-fix predicted arcs in time order, fixes without a
  prediction removed, consecutive repeats collapsed.
* route_continuity = share of consecutive predicted-route pairs (a, b) that are
  drivable without a jump: to_node(a) == from_node(b). 1.0 for a route of length < 2.
* invalid_transitions = number of such pairs that are not connected.
* route_precision / route_recall / route_f1 = length-weighted overlap between the
  set of predicted arcs and the set of reference arcs (metres of correct road).
* missing_segments / extra_segments = reference arcs never predicted / predicted
  arcs not in the reference.
* edit_similarity = 1 - Levenshtein(pred, ref) / max(len(pred), len(ref)) on the
  collapsed arc sequences (penalises order errors, missing and extra arcs).
* lcs_recall / lcs_precision = longest common subsequence / len(ref) (resp. len(pred)):
  ordered coverage of the reference.
"""
from __future__ import annotations

import numpy as np


def collapse(seq) -> np.ndarray:
    s = np.asarray(seq, dtype=np.int64)
    s = s[s >= 0]
    if len(s) == 0:
        return s
    keep = np.concatenate([[True], s[1:] != s[:-1]])
    return s[keep]


def levenshtein(a: np.ndarray, b: np.ndarray) -> int:
    """Vectorised O(len(a)*len(b)) edit distance (unit costs)."""
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return max(n, m)
    j = np.arange(m + 1)
    prev = j.astype(np.int64)
    for i in range(1, n + 1):
        sub = prev[:-1] + (b != a[i - 1])
        tmp = np.empty(m + 1, np.int64)
        tmp[0] = i
        tmp[1:] = np.minimum(prev[1:] + 1, sub)
        prev = j + np.minimum.accumulate(tmp - j)
    return int(prev[-1])


def lcs_length(a: np.ndarray, b: np.ndarray) -> int:
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0
    prev = np.zeros(m + 1, np.int64)
    for i in range(1, n + 1):
        cur = np.empty(m + 1, np.int64)
        cur[0] = 0
        cur[1:] = np.maximum(prev[1:], prev[:-1] + (b == a[i - 1]))
        prev = np.maximum.accumulate(cur)
    return int(prev[-1])


def continuity(route: np.ndarray, net) -> tuple[float, int]:
    if len(route) < 2:
        return 1.0, 0
    ok = net.v[route[:-1]] == net.u[route[1:]]
    return float(ok.mean()), int((~ok).sum())


def route_metrics(pred_arcs, net, reference=None) -> dict:
    pr = collapse(pred_arcs)
    cont, n_bad = continuity(pr, net)
    out = {"pred_route_len": int(len(pr)), "route_continuity": cont, "invalid_transitions": n_bad}
    if reference is None:
        return out
    ref = np.asarray(reference, dtype=np.int64)
    ref_c = collapse(ref)
    ps, rs = np.unique(pr), np.unique(ref_c)
    inter = np.intersect1d(ps, rs)
    L = net.length
    p_len, r_len, i_len = L[ps].sum(), L[rs].sum(), L[inter].sum()
    prec = float(i_len / p_len) if p_len > 0 else 0.0
    rec = float(i_len / r_len) if r_len > 0 else 0.0
    lcs = lcs_length(pr, ref_c)
    ed = levenshtein(pr, ref_c)
    out.update({
        "ref_route_len": int(len(ref_c)),
        "route_precision": prec, "route_recall": rec,
        "route_f1": 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0,
        "missing_segments": int(len(np.setdiff1d(rs, ps))),
        "extra_segments": int(len(np.setdiff1d(ps, rs))),
        "edit_similarity": 1.0 - ed / max(len(pr), len(ref_c), 1),
        "lcs_recall": lcs / max(len(ref_c), 1),
        "lcs_precision": lcs / max(len(pr), 1),
        "ref_continuity": continuity(ref_c, net)[0],
    })
    return out
