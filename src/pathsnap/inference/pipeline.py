"""End-to-end inference: raw trip folder -> matched route.

Leakage guarantees
------------------
* ``MapMatcher.predict`` loads the trip with ``with_route=False``: the .route
  file is never opened on the prediction path.
* ``RouteEvaluator`` is the only component that reads reference routes. It
  receives the *prediction* and returns aggregate metrics only; no reference
  arc ids or geometry are returned.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ..config import artifact_dir, load_config
from ..data.dataset import process_trip
from ..data.labeling import align_to_route
from ..data.loader import load_trip
from ..decoding.hmm import HMMDecoder, classifier_only, reconstruct_route
from ..evaluation.metrics import classification_metrics, selection_macro_f1
from ..evaluation.route_metrics import collapse, route_metrics
from ..features.candidate_generation import attach_labels, candidate_coverage
from ..features.feature_schema import FEATURE_COLUMNS, FEATURE_VERSION, SchemaError, to_matrix
from ..models.model_factory import positive_proba


class ArtifactError(RuntimeError):
    pass


@dataclass
class Prediction:
    trip_id: str
    ctx: object                 # TripContext (server-side only)
    prob: np.ndarray            # classifier P(correct) per candidate row
    clf_arc: np.ndarray         # per cleaned fix
    clf_prob: np.ndarray
    hmm_arc: np.ndarray
    hmm_prob: np.ndarray
    chain: np.ndarray
    breaks: dict
    route: np.ndarray           # reconstructed drivable arc sequence
    route_inferred: np.ndarray
    clf_route: np.ndarray
    unfilled_gaps: int
    timings: dict
    diagnostics: dict


class MapMatcher:
    def __init__(self, artifacts: Path | None = None):
        self.root = Path(artifacts or artifact_dir())
        mpath = self.root / "models" / "final_classifier.joblib"
        if not mpath.exists():
            raise ArtifactError(f"model artifact not found: {mpath}")
        try:
            self.model = joblib.load(mpath)
        except Exception as exc:
            raise ArtifactError(f"model artifact could not be loaded ({type(exc).__name__}): {exc}") from exc
        if not hasattr(self.model, "predict_proba"):
            raise ArtifactError("model artifact has no predict_proba")
        try:
            self.meta = json.loads((self.root / "metadata" / "classifier.json").read_text())
            self.hmm_meta = json.loads((self.root / "metadata" / "hmm_params.json").read_text())
        except FileNotFoundError as exc:
            raise ArtifactError(f"metadata missing: {exc.filename}") from exc
        if self.meta.get("feature_columns") != FEATURE_COLUMNS or self.meta.get("feature_version") != FEATURE_VERSION:
            raise SchemaError("artifact feature schema differs from code schema - retrain or check out matching code")
        n_in = getattr(self.model, "n_features_in_", len(FEATURE_COLUMNS))
        if n_in != len(FEATURE_COLUMNS):
            raise SchemaError(f"model expects {n_in} features, schema has {len(FEATURE_COLUMNS)}")
        fpath = self.root / "metadata" / "feature_config.json"
        self.feat_cfg = json.loads(fpath.read_text()) if fpath.exists() else load_config("features")
        self.hmm = HMMDecoder(self.hmm_meta["params"])

    def info(self) -> dict:
        return {"model": self.meta.get("variant"), "model_family": self.meta.get("model_name"),
                "feature_version": self.meta.get("feature_version"), "n_features": len(FEATURE_COLUMNS),
                "trained_at": self.meta.get("timestamp_utc"), "git_commit": self.meta.get("git_commit"),
                "cv_macro_f1": self.meta.get("cv_macro_f1"), "student_test_top1": self.meta.get("student_test_top1"),
                "hmm_params": self.hmm_meta.get("params")}

    def predict(self, trip_dir: Path) -> Prediction:
        t0 = time.perf_counter()
        trip = load_trip(Path(trip_dir), with_route=False)          # labels never loaded here
        ctx = process_trip(trip, with_labels=False, feat_cfg=self.feat_cfg)
        timings = dict(ctx.timings)
        n = len(ctx.track)
        t1 = time.perf_counter()
        if len(ctx.cands):
            prob = positive_proba(self.model, to_matrix(ctx.cands))
            if not np.isfinite(prob).all() or prob.min() < 0 or prob.max() > 1:
                raise ValueError("classifier returned invalid probabilities")
        else:
            prob = np.array([])
        timings["classifier_s"] = time.perf_counter() - t1
        clf_arc, clf_p = classifier_only(ctx.cands, prob, n)
        t2 = time.perf_counter()
        if len(ctx.cands):
            res = self.hmm.decode(ctx.net, ctx.cands, prob, ctx.px, ctx.py, ctx.track.t)
            hmm_arc, hmm_p, chain, breaks, diag = res.arc, res.prob, res.chain, res.breaks, res.diagnostics
        else:
            hmm_arc, hmm_p = np.full(n, -1), np.full(n, np.nan)
            chain, breaks, diag = np.full(n, -1), {}, {"n_chains": 0}
        timings["hmm_s"] = time.perf_counter() - t2
        t3 = time.perf_counter()
        route, inferred, unfilled = reconstruct_route(ctx.net, hmm_arc, chain)
        clf_route = collapse(clf_arc)          # classifier-only route: no path model, so no infill
        timings["reconstruction_s"] = time.perf_counter() - t3
        timings["total_s"] = time.perf_counter() - t0
        return Prediction(trip.trip_id, ctx, prob, clf_arc, clf_p, hmm_arc, hmm_p, chain, breaks, route, inferred,
                          clf_route, unfilled, timings, diag)


class RouteEvaluator:
    """Server-side scoring against the reference route. Returns aggregates only."""

    def __init__(self, data_cfg: dict | None = None):
        self.lc = (data_cfg or load_config("data"))["labelling"]

    def evaluate(self, trip_dir: Path, pred: Prediction) -> dict:
        route_file = Path(trip_dir) / f"{Path(trip_dir).name}.route"
        if not route_file.exists():
            return {"available": False, "reason": "no reference route for this trip on the server"}
        ref = load_trip(Path(trip_dir), with_route=True).route
        ctx = pred.ctx
        labels, _, _ = align_to_route(ctx.px, ctx.py, ctx.net, ref, self.lc["distance_cap_m"],
                                      self.lc["max_label_distance_m"])
        m = labels >= 0
        cov = candidate_coverage(ctx.cands, labels)
        lab_c = attach_labels(ctx.cands, labels)
        lab_c["trip_id"] = pred.trip_id
        rows = lab_c["true_arc"].to_numpy() >= 0

        def frame(arc):
            k = np.where(arc >= 0)[0]
            return pd.DataFrame({"trip_id": pred.trip_id, "obs": k, "arc": arc[k]})

        cm = classification_metrics(lab_c["label"].to_numpy()[rows], pred.prob[rows]) if rows.any() else {}
        rm_clf = route_metrics(pred.clf_arc, ctx.net, ref)
        rm_rec = route_metrics(pred.route, ctx.net, ref)
        return {
            "available": True,
            "n_fixes": int(len(labels)), "n_labelled_fixes": int(m.sum()),
            "candidate_miss_rate": cov["miss_rate"],
            "classifier": {
                "top1_accuracy": float((pred.clf_arc[m] == labels[m]).mean()) if m.any() else None,
                "macro_f1_threshold_0_5": cm.get("macro_f1"),
                "selection_macro_f1": selection_macro_f1(lab_c, frame(pred.clf_arc)),
                "log_loss": cm.get("log_loss"),
                "route_continuity": rm_clf["route_continuity"], "route_f1": rm_clf["route_f1"],
                "edit_similarity": rm_clf["edit_similarity"], "invalid_transitions": rm_clf["invalid_transitions"],
            },
            "hmm": {
                "top1_accuracy": float((pred.hmm_arc[m] == labels[m]).mean()) if m.any() else None,
                "selection_macro_f1": selection_macro_f1(lab_c, frame(pred.hmm_arc)),
                "route_continuity": rm_rec["route_continuity"], "route_f1": rm_rec["route_f1"],
                "route_precision": rm_rec["route_precision"], "route_recall": rm_rec["route_recall"],
                "edit_similarity": rm_rec["edit_similarity"], "lcs_recall": rm_rec["lcs_recall"],
                "invalid_transitions": rm_rec["invalid_transitions"],
                "missing_segments": rm_rec["missing_segments"], "extra_segments": rm_rec["extra_segments"],
            },
        }
