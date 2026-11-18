"""Train and evaluate every model in configs/models.yaml.

Protocol (identical for all models):
  1. 5-fold GroupKFold over the 80 training trips (group = trip_id); fit on
     training-fold rows (every 2nd fix), score on ALL labelled fixes of the
     validation trips. Train-fold scores are kept for the generalisation gap.
  2. Refit on all 80 training trips, score once on the 10 student-testing trips.
  3. Record configuration, seed, feature version, timings and git commit.

Usage:  python scripts/train_all.py [--models hist_gradient_boosting logistic_regression ...]
Outputs: experiments/results/*, experiments/logs/train_all.log, artifacts/models/experiments/*.joblib
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathsnap.config import SEED, load_config, processed_dir  # noqa: E402
from pathsnap.evaluation.metrics import per_trip_top1  # noqa: E402
from pathsnap.evaluation.networks import load_network  # noqa: E402
from pathsnap.evaluation.route_metrics import route_metrics  # noqa: E402
from pathsnap.features.feature_schema import FEATURE_COLUMNS, FEATURE_VERSION  # noqa: E402
from pathsnap.models.model_factory import MODEL_CARDS  # noqa: E402
from pathsnap.training.cross_validation import evaluate_model, fit_model, run_cv, training_rows, trip_folds  # noqa: E402
from pathsnap.training.tracking import run_metadata, write_json  # noqa: E402

RES = ROOT / "experiments" / "results"
LOGF = ROOT / "experiments" / "logs" / "train_all.log"


def log(msg):
    print(msg, flush=True)
    with open(LOGF, "a", encoding="utf-8") as fh:
        fh.write(msg + "\n")


def load_data():
    pd_ = processed_dir()
    return (pd.read_pickle(pd_ / "train_candidates.pkl"), pd.read_pickle(pd_ / "train_obs.pkl"),
            pd.read_pickle(pd_ / "student_test_candidates.pkl"), pd.read_pickle(pd_ / "student_test_obs.pkl"))


def per_fix_sequences(pred: pd.DataFrame, obs: pd.DataFrame) -> dict[str, np.ndarray]:
    m = obs[["trip_id", "obs"]].merge(pred[["trip_id", "obs", "arc"]], on=["trip_id", "obs"], how="left")
    m["arc"] = m["arc"].fillna(-1).astype(np.int64)
    return {tid: g.sort_values("obs")["arc"].to_numpy() for tid, g in m.groupby("trip_id")}


def student_route_metrics(pred, obs, nets):
    seqs = per_fix_sequences(pred, obs)
    rows = []
    for tid, seq in seqs.items():
        trip, net = nets[tid]
        rows.append({"trip_id": tid, **route_metrics(seq, net, trip.route)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--fresh", action="store_true", help="start a new model_summary.csv and log")
    args = ap.parse_args()
    RES.mkdir(parents=True, exist_ok=True)
    LOGF.parent.mkdir(parents=True, exist_ok=True)
    if args.fresh:
        LOGF.write_text("", encoding="utf-8")
        (RES / "model_summary.csv").unlink(missing_ok=True)
    (ROOT / "artifacts" / "models" / "experiments").mkdir(parents=True, exist_ok=True)
    mcfg = load_config("models")
    fcfg = load_config("features")
    cv_cfg = mcfg["cv"]
    stride = int(fcfg["training"]["obs_stride"])
    names = args.models or list(mcfg["models"].keys())

    tr_c, tr_o, st_c, st_o = load_data()
    folds = trip_folds(tr_c["trip_id"].unique(), cv_cfg["n_splits"], cv_cfg["seed"])
    write_json(ROOT / "experiments" / "configurations" / "cv_folds.json",
               {"n_splits": cv_cfg["n_splits"], "seed": cv_cfg["seed"], "group": "trip_id",
                "folds": [{"train": list(a), "val": list(b)} for a, b in folds]})
    nets = {tid: load_network("student_test", tid) for tid in st_o["trip_id"].unique()}
    log(f"=== train_all  models={names}  features={FEATURE_VERSION} ({len(FEATURE_COLUMNS)})  stride={stride}")
    log(f"train rows={len(tr_c)}  student-test rows={len(st_c)}")

    summary_path = RES / "model_summary.csv"
    summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    for name in names:
        spec = mcfg["models"][name]
        params = spec.get("params", {})
        extra = {k: v for k, v in spec.items() if k not in ("params", "family")}
        log(f"--- {name}: {params} {extra}")
        t0 = time.perf_counter()
        folds_df, oof = run_cv(name, params, extra, tr_c, tr_o, folds, stride=stride,
                               threshold=cv_cfg["decision_threshold"], seed=SEED, keep_oof=True, log=log)
        folds_df.to_csv(RES / f"cv_folds_{name}.csv", index=False)
        np.save(processed_dir() / f"oof_{name}.npy", oof)
        cv_s = time.perf_counter() - t0

        model, fit_s, n_fit = fit_model(name, params, extra, training_rows(tr_c, stride), seed=SEED)
        m_st, p_st, pred_st = evaluate_model(model, st_c, st_o, threshold=cv_cfg["decision_threshold"])
        np.save(processed_dir() / f"student_probs_{name}.npy", p_st)
        rm = student_route_metrics(pred_st, st_o, nets)
        rm.to_csv(RES / f"student_route_metrics_{name}.csv", index=False)
        per_trip_top1(st_o, pred_st).to_csv(RES / f"student_per_trip_top1_{name}.csv", index=False)
        joblib.dump(model, ROOT / "artifacts" / "models" / "experiments" / f"{name}.joblib", compress=3)

        hist = None
        est = model.named_steps["clf"] if hasattr(model, "named_steps") else model
        if hasattr(est, "loss_curve_"):
            hist = pd.DataFrame({"epoch": np.arange(1, len(est.loss_curve_) + 1), "train_loss": est.loss_curve_,
                                 "val_accuracy": getattr(est, "validation_scores_", None) or [np.nan] * len(est.loss_curve_)})
        elif hasattr(est, "train_score_") and len(getattr(est, "train_score_", [])):
            hist = pd.DataFrame({"iteration": np.arange(len(est.train_score_)),
                                 "train_neg_log_loss": est.train_score_,
                                 "val_neg_log_loss": est.validation_score_ if len(est.validation_score_) else np.nan})
        if hist is not None:
            hist.to_csv(RES / f"training_history_{name}.csv", index=False)
        if hasattr(est, "feature_importances_"):
            pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": est.feature_importances_}) \
                .sort_values("importance", ascending=False).to_csv(RES / f"feature_importance_{name}.csv", index=False)

        cvm = folds_df.mean(numeric_only=True)
        cvs = folds_df.std(numeric_only=True)
        row = {
            "model": name, "display": MODEL_CARDS[name]["display"], "family": spec["family"],
            "key_config": "; ".join(f"{k}={v}" for k, v in params.items() if k not in ("n_jobs", "max_iter")),
            "cv_macro_f1": cvm["val_macro_f1"], "cv_macro_f1_std": cvs["val_macro_f1"],
            "cv_top1": cvm["val_top1_accuracy"], "cv_top1_std": cvs["val_top1_accuracy"],
            "cv_log_loss": cvm["val_log_loss"], "cv_roc_auc": cvm["val_roc_auc"],
            "train_macro_f1": cvm.get("train_macro_f1"), "train_top1": cvm.get("train_top1_accuracy"),
            "train_log_loss": cvm.get("train_log_loss"),
            "gap_macro_f1": cvm.get("train_macro_f1", np.nan) - cvm["val_macro_f1"],
            "gap_top1": cvm.get("train_top1_accuracy", np.nan) - cvm["val_top1_accuracy"],
            "st_accuracy": m_st["accuracy"], "st_precision_pos": m_st["precision_pos"], "st_recall_pos": m_st["recall_pos"],
            "st_precision_macro": m_st["precision_macro"], "st_recall_macro": m_st["recall_macro"],
            "st_macro_f1": m_st["macro_f1"], "st_top1": m_st["top1_accuracy"], "st_log_loss": m_st["log_loss"],
            "st_roc_auc": m_st["roc_auc"], "st_brier": m_st["brier"],
            "st_tn": m_st["tn"], "st_fp": m_st["fp"], "st_fn": m_st["fn"], "st_tp": m_st["tp"],
            "st_route_continuity": rm["route_continuity"].mean(), "st_route_f1": rm["route_f1"].mean(),
            "st_edit_similarity": rm["edit_similarity"].mean(),
            "fit_s_full": fit_s, "n_fit_rows_full": n_fit, "cv_total_s": cv_s,
            "inference_s_student": m_st["inference_s"], "inference_ms_per_1k_fixes": m_st["inference_ms_per_1k_fixes"],
            "model_size_mb": (ROOT / "artifacts" / "models" / "experiments" / f"{name}.joblib").stat().st_size / 1e6,
        }
        log(f"  => {name}: CV macroF1={row['cv_macro_f1']:.4f}±{row['cv_macro_f1_std']:.4f} CV top1={row['cv_top1']:.4f} "
            f"| ST macroF1={row['st_macro_f1']:.4f} top1={row['st_top1']:.4f} logloss={row['st_log_loss']:.4f} "
            f"cont={row['st_route_continuity']:.4f} | fit={fit_s:.1f}s infer={m_st['inference_s']:.2f}s")
        write_json(RES / f"experiment_{name}.json", {
            **run_metadata(), "model": name, "card": MODEL_CARDS[name], "params": params, "extra": extra,
            "seed": SEED, "feature_version": FEATURE_VERSION, "feature_columns": FEATURE_COLUMNS,
            "obs_stride": stride, "cv": cv_cfg, "summary": row,
            "fold_metrics": folds_df.to_dict(orient="records"), "student_test_metrics": m_st,
        })
        if len(summary) and "model" in summary:
            summary = summary[summary["model"] != name]
        summary = pd.concat([summary, pd.DataFrame([row])], ignore_index=True)
        summary.to_csv(summary_path, index=False)
    log("=== done")


if __name__ == "__main__":
    main()
