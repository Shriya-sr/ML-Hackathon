"""Numerically stable Viterbi decoding in log space over variable-size state sets.

Recurrence for one chain (segment):
    delta_0(s)   = logE_0(s)
    delta_t(s)   = max_r [ delta_{t-1}(r) + logT_t(r, s) ] + logE_t(s)
    psi_t(s)     = argmax_r [ ... ]
Backtracking from argmax_s delta_T(s) recovers the MAP state sequence.

Fallback policy (never fabricates road segments):
  * a step with no states (no candidates)  -> output -1 there, start a new chain after it
  * a step where every transition is -inf  -> start a new chain at that step
  * a step flagged as a large GPS gap       -> start a new chain at that step
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np


def viterbi_chain(log_em: Sequence[np.ndarray], trans: Callable[[int], np.ndarray]) -> np.ndarray:
    """Decode one chain where every transition step has at least one finite path."""
    T = len(log_em)
    if T == 0:
        return np.array([], dtype=np.int64)
    delta = np.asarray(log_em[0], float)
    psis = []
    for t in range(1, T):
        M = delta[:, None] + trans(t)
        psi = np.argmax(M, axis=0)
        delta = M[psi, np.arange(M.shape[1])] + log_em[t]
        psis.append(psi)
    path = np.empty(T, np.int64)
    path[-1] = int(np.argmax(delta))
    for t in range(T - 1, 0, -1):
        path[t - 1] = psis[t - 1][path[t]]
    return path


def viterbi_segmented(log_em: Sequence[np.ndarray], trans: Callable[[int], np.ndarray],
                      force_break: Sequence[bool] | None = None):
    """Viterbi over a whole trip with automatic chain breaking.

    ``trans(t)`` returns the (|S_{t-1}|, |S_t|) log-transition matrix.
    Returns (path, chain_id, break_reasons) where path[t] = -1 if step t has no states.
    """
    T = len(log_em)
    path = np.full(T, -1, np.int64)
    chain = np.full(T, -1, np.int64)
    reasons: dict[int, str] = {}
    force_break = force_break if force_break is not None else [False] * T
    t, cid = 0, 0
    while t < T:
        if len(log_em[t]) == 0:
            reasons[t] = "no_candidates"
            t += 1
            continue
        # grow a chain [start, end) as long as transitions stay feasible
        start = t
        delta = np.asarray(log_em[t], float)
        psis, mats = [], []
        t += 1
        while t < T:
            if len(log_em[t]) == 0:
                break
            if force_break[t]:
                reasons[t] = "gps_gap"
                break
            Tm = trans(t)
            M = delta[:, None] + Tm
            if not np.isfinite(M).any():
                reasons[t] = "no_valid_transition"
                break
            psi = np.argmax(M, axis=0)
            new = M[psi, np.arange(M.shape[1])] + log_em[t]
            if not np.isfinite(new).any():
                reasons[t] = "no_valid_transition"
                break
            psis.append(psi)
            delta = new
            t += 1
        end = t
        seg = np.empty(end - start, np.int64)
        seg[-1] = int(np.argmax(delta))
        for k in range(end - start - 1, 0, -1):
            seg[k - 1] = psis[k - 1][seg[k]]
        path[start:end] = seg
        chain[start:end] = cid
        cid += 1
    return path, chain, reasons
