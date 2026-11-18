"""Runs the real inference pipeline and shapes the API payload.

Nothing here is precomputed: every POST /api/predict/{trip_id} executes
cleaning -> candidates -> features -> classifier -> HMM/Viterbi -> reconstruction
on the raw trip files, then (server-side only) scores it against the reference.
"""
from __future__ import annotations

import logging
import threading
from collections import OrderedDict

import numpy as np

from pathsnap.features.feature_schema import FEATURE_COLUMNS
from pathsnap.inference.pipeline import MapMatcher, Prediction, RouteEvaluator

log = logging.getLogger("pathsnap.api")

EXPLAIN_FEATURES = ["dist_m", "cos_heading", "win_mean_dist_m", "dist_rank", "cos_long_bearing", "speed_mps"]


class PredictionService:
    def __init__(self, matcher: MapMatcher | None, registry, settings):
        self.matcher = matcher
        self.registry = registry
        self.settings = settings
        self.evaluator = RouteEvaluator()
        self._cache: OrderedDict[str, tuple[Prediction, dict]] = OrderedDict()
        self._lock = threading.Lock()

    # ------------------------------------------------------------- running
    def predict(self, trip_id: str) -> dict:
        if self.matcher is None:
            raise RuntimeError("model not loaded")
        tdir = self.registry.trip_dir(trip_id)
        pred = self.matcher.predict(tdir)
        metrics = self.evaluator.evaluate(tdir, pred)
        payload = self._payload(pred, metrics)
        with self._lock:
            self._cache[trip_id] = (pred, payload)
            self._cache.move_to_end(trip_id)
            while len(self._cache) > self.settings.result_cache_size:
                self._cache.popitem(last=False)
        m = metrics
        if m.get("available"):
            log.info("trip=%s fixes=%d clf_top1=%.4f hmm_top1=%.4f continuity=%.4f total=%.2fs", trip_id,
                     m["n_fixes"], m["classifier"]["top1_accuracy"] or 0, m["hmm"]["top1_accuracy"] or 0,
                     m["hmm"]["route_continuity"], pred.timings["total_s"])
        return payload

    def cached(self, trip_id: str) -> dict | None:
        with self._lock:
            hit = self._cache.get(trip_id)
        return hit[1] if hit else None

    # -------------------------------------------------------------- payload
    def _payload(self, pred: Prediction, metrics: dict) -> dict:
        ctx = pred.ctx
        net = ctx.net
        lat, lon = ctx.track.lat, ctx.track.lon
        used = set(np.unique(pred.route).tolist()) | set(np.unique(pred.clf_arc[pred.clf_arc >= 0]).tolist())
        cand_arcs = np.unique(ctx.cands["arc"].to_numpy()) if len(ctx.cands) else np.array([], np.int64)
        # draw each two-way road once
        rev = net.reverse_arc[cand_arcs]
        draw = cand_arcs[(rev < 0) | (cand_arcs < rev)][: self.settings.max_network_arcs]
        a, b = net.arc_lonlat(draw)
        network = [[[round(float(a[i, 1]), 6), round(float(a[i, 0]), 6)], [round(float(b[i, 1]), 6), round(float(b[i, 0]), 6)]]
                   for i in range(len(draw))]
        used_arr = np.array(sorted(used), dtype=np.int64)
        ua, ub = net.arc_lonlat(used_arr) if len(used_arr) else (np.zeros((0, 2)), np.zeros((0, 2)))
        arcs = {str(int(k)): [[round(float(ua[i, 1]), 6), round(float(ua[i, 0]), 6)],
                              [round(float(ub[i, 1]), 6), round(float(ub[i, 0]), 6)]] for i, k in enumerate(used_arr)}
        dt = np.diff(ctx.track.t)
        gaps = [{"index": int(i + 1), "dt_s": float(dt[i])} for i in np.where(dt > 30)[0]]
        low = np.where(np.nan_to_num(pred.hmm_prob, nan=0) < self.settings.low_confidence_threshold)[0]
        no_c = np.where(pred.hmm_arc < 0)[0]
        r3 = lambda x: [None if not np.isfinite(v) else round(float(v), 3) for v in x]  # noqa: E731
        return {
            "trip_id": pred.trip_id, "mode": self.registry.mode,
            "model": self.matcher.info() if self.matcher else {},
            "n_fixes": int(len(lat)),
            "gps": {"lat": np.round(lat, 6).tolist(), "lon": np.round(lon, 6).tolist(),
                    "t": ctx.track.t.tolist()},
            "fixes": {"hmm_arc": pred.hmm_arc.tolist(), "hmm_prob": r3(pred.hmm_prob),
                      "clf_arc": pred.clf_arc.tolist(), "clf_prob": r3(pred.clf_prob),
                      "chain": pred.chain.tolist(), "orig_index": ctx.track.orig_index.tolist()},
            "arcs": arcs, "network": network,
            "route": {"hmm": pred.route.tolist(), "hmm_inferred": pred.route_inferred.tolist(),
                      "classifier": pred.clf_route.tolist()},
            "robustness": {
                "gps_gaps": gaps,
                "chain_breaks": [{"index": int(k), "reason": v} for k, v in sorted(pred.breaks.items())],
                "no_candidate_fixes": no_c.tolist(),
                "low_confidence_fixes": low.tolist(),
                "unfilled_route_gaps": int(pred.unfilled_gaps),
                "n_dropped_invalid": int(ctx.track.n_dropped_invalid),
                "n_dropped_duplicate": int(ctx.track.n_dropped_duplicate),
                "was_resorted": bool(ctx.track.was_resorted),
            },
            "metrics": metrics,
            "timings": {k: round(float(v), 4) for k, v in pred.timings.items()},
            "diagnostics": {**pred.diagnostics, "n_candidates": int(len(ctx.cands)),
                            "n_network_arcs_drawn": len(network), "network_arcs_total": int(net.E),
                            "route_length_m": float(net.length[pred.route].sum()) if len(pred.route) else 0.0,
                            "utm_epsg": int(ctx.projector.epsg)},
        }

    def explain(self, trip_id: str, fix: int) -> dict | None:
        with self._lock:
            hit = self._cache.get(trip_id)
        if not hit:
            return None
        pred = hit[0]
        ctx = pred.ctx
        if fix < 0 or fix >= len(ctx.track):
            raise IndexError(fix)
        rows = ctx.cands.index[ctx.cands["obs"].to_numpy() == fix]
        sub = ctx.cands.loc[rows].copy()
        sub["prob"] = pred.prob[rows] if len(rows) else []
        sub = sub.sort_values("prob", ascending=False)
        cands = [{"arc": int(r.arc), "probability": round(float(r.prob), 4),
                  "chosen_by_hmm": bool(r.arc == pred.hmm_arc[fix]),
                  "chosen_by_classifier": bool(r.arc == pred.clf_arc[fix]),
                  "features": {f: round(float(getattr(r, f)), 3) for f in EXPLAIN_FEATURES if f in FEATURE_COLUMNS}}
                 for r in sub.itertuples()]
        return {"trip_id": trip_id, "fix_index": int(fix), "timestamp": float(ctx.track.t[fix]),
                "hmm_arc": int(pred.hmm_arc[fix]), "classifier_arc": int(pred.clf_arc[fix]), "candidates": cands}
