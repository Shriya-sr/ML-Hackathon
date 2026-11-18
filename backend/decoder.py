from __future__ import annotations

import numpy as np


def viterbi(candidate_ids: list[np.ndarray], emissions: list[np.ndarray], arcs: np.ndarray,
            transition_penalty: float = 6.0) -> np.ndarray:
    """Decode candidate road arcs with emissions and directed-arc continuity."""
    if not candidate_ids:
        return np.empty(0, dtype=np.int32)
    score = np.log(np.maximum(np.asarray(emissions[0], dtype=float), 1e-12))
    backpointers: list[np.ndarray] = []
    for t in range(1, len(candidate_ids)):
        prev, cur = candidate_ids[t - 1], candidate_ids[t]
        prev_end = arcs[prev, 1]
        cur_start = arcs[cur, 0]
        connected = prev_end[:, None] == cur_start[None, :]
        same = prev[:, None] == cur[None, :]
        trans = np.where(connected | same, 0.0, -float(transition_penalty))
        matrix = score[:, None] + trans
        bp = np.argmax(matrix, axis=0)
        score = matrix[bp, np.arange(len(cur))] + np.log(np.maximum(np.asarray(emissions[t], dtype=float), 1e-12))
        backpointers.append(bp.astype(np.int32))
    state = int(np.argmax(score))
    output = np.empty(len(candidate_ids), dtype=np.int32)
    output[-1] = candidate_ids[-1][state]
    for t in range(len(candidate_ids) - 1, 0, -1):
        state = int(backpointers[t - 1][state])
        output[t - 1] = candidate_ids[t - 1][state]
    return output


def continuity(route: np.ndarray, arcs: np.ndarray) -> float:
    if len(route) < 2:
        return 1.0
    a = arcs[np.asarray(route[:-1], dtype=int)]
    b = arcs[np.asarray(route[1:], dtype=int)]
    return float(np.mean((a[:, 1] == b[:, 0]) | (np.asarray(route[:-1]) == np.asarray(route[1:]))))
