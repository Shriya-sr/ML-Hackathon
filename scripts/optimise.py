"""Bayesian hyperparameter optimisation of the selected classifier.

Usage:  python scripts/optimise.py [--trials 40]
Steps:
  1. Optuna TPE search, objective = mean CV Macro-F1 on the SAME 5 trip-grouped
     folds as the baseline (training trips only; student/evaluator never used).
  2. Re-score the best configuration with the baseline protocol (stride 2) and
     keep out-of-fold probabilities (used later to tune the HMM).
  3. Refit on all 80 training trips, score once on the 10 student-testing trips.
  4. Choose between tuned and untuned by CV Macro-F1 (not by student-test score)
     and export the final classifier to artifacts/models/final_classifier.joblib.
Outputs: experiments/optimisation/*, artifacts/models/*, reports/figures/optimisation_*.png
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathsnap.config import SEED, load_config, processed_dir  # noqa: E402
from pathsnap.features.feature_schema import FEATURE_COLUMNS, FEATURE_VERSION  # noqa: E402
from pathsnap.optimisation.bayesian_search import run_study  # noqa: E402
from pathsnap.optimisation.search_space import normalise  # noqa: E402
from pathsnap.training.cross_validation import evaluate_model, fit_model, run_cv, training_rows, trip_folds  # noqa: E402
from pathsnap.training.tracking import run_metadata, write_json  # noqa: E402
from pathsnap.visualisation.plots import plt, save  # noqa: E402

OPT = ROOT / "experiments" / "optimisation"
FIG = ROOT / "reports" / "figures"
LOGF = ROOT / "experiments" / "logs" / "optimise.log"


def log(msg):
    print(msg, flush=True)
    with open(LOGF, "a", encoding="utf-8") as fh:
        fh.write(msg + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config("optimisation")
    mcfg = load_config("models")
    fcfg = load_config("features")
    name = cfg["model"]
    n_trials = args.trials or cfg["n_trials"]
    stride = int(fcfg["training"]["obs_stride"])
    OPT.mkdir(parents=True, exist_ok=True)
    pdir = processed_dir()
    tr_c, tr_o = pd.read_pickle(pdir / "train_candidates.pkl"), pd.read_pickle(pdir / "train_obs.pkl")
    st_c, st_o = pd.read_pickle(pdir / "student_test_candidates.pkl"), pd.read_pickle(pdir / "student_test_obs.pkl")
    folds = trip_folds(tr_c["trip_id"].unique(), mcfg["cv"]["n_splits"], mcfg["cv"]["seed"])

    db = ROOT / cfg["storage"]
    if db.exists():
        db.unlink()
    storage = f"sqlite:///{db.as_posix()}"
    log(f"=== optimise {name}: {n_trials} trials, search stride {cfg['search_obs_stride']}")
    t0 = time.perf_counter()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = run_study(name, tr_c, tr_o, folds, n_trials, int(cfg["search_obs_stride"]), storage=storage,
                      seed=cfg["seed"], log=log)
    search_s = time.perf_counter() - t0
    trials = study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs", "duration"))
    trials["duration_s"] = trials["duration"].dt.total_seconds()
    trials.drop(columns=["duration"]).to_csv(OPT / "trials.csv", index=False)
    best = normalise(name, {**study.best_params})
    fixed = {"early_stopping": True, "validation_fraction": 0.1, "n_iter_no_change": 20} \
        if name == "hist_gradient_boosting" else {}
    best_params = {**best, **fixed}
    try:
        importance = optuna.importance.get_param_importances(study)
    except Exception as exc:  # e.g. too few completed trials
        importance = {"error": str(exc)}
    log(f"best trial #{study.best_trial.number} value={study.best_value:.4f} params={best_params}")

    # re-score tuned config with the exact baseline protocol
    folds_df, oof = run_cv(name, best_params, {}, tr_c, tr_o, folds, stride=stride, keep_oof=True, log=log)
    folds_df.to_csv(OPT / f"cv_folds_{name}_tuned.csv", index=False)
    np.save(pdir / f"oof_{name}_tuned.npy", oof)
    model, fit_s, n_fit = fit_model(name, best_params, {}, training_rows(tr_c, stride), seed=SEED)
    m_st, p_st, _ = evaluate_model(model, st_c, st_o)
    np.save(pdir / f"student_probs_{name}_tuned.npy", p_st)
    joblib.dump(model, ROOT / "artifacts" / "models" / f"{name}_tuned.joblib", compress=3)

    base = pd.read_csv(ROOT / "experiments" / "results" / "model_summary.csv").set_index("model").loc[name]
    base_folds = pd.read_csv(ROOT / "experiments" / "results" / f"cv_folds_{name}.csv")
    tuned = {
        "cv_macro_f1": folds_df["val_macro_f1"].mean(), "cv_macro_f1_std": folds_df["val_macro_f1"].std(),
        "cv_top1": folds_df["val_top1_accuracy"].mean(), "cv_log_loss": folds_df["val_log_loss"].mean(),
        "st_top1": m_st["top1_accuracy"], "st_macro_f1": m_st["macro_f1"], "st_log_loss": m_st["log_loss"],
        "st_precision_macro": m_st["precision_macro"], "st_recall_macro": m_st["recall_macro"],
        "inference_ms_per_1k_fixes": m_st["inference_ms_per_1k_fixes"], "fit_s_full": fit_s,
        "train_macro_f1": folds_df["train_macro_f1"].mean(),
    }
    untuned = {
        "cv_macro_f1": base_folds["val_macro_f1"].mean(), "cv_macro_f1_std": base_folds["val_macro_f1"].std(),
        "cv_top1": base["cv_top1"], "cv_log_loss": base["cv_log_loss"], "st_top1": base["st_top1"],
        "st_macro_f1": base["st_macro_f1"], "st_log_loss": base["st_log_loss"],
        "st_precision_macro": base["st_precision_macro"], "st_recall_macro": base["st_recall_macro"],
        "inference_ms_per_1k_fixes": base["inference_ms_per_1k_fixes"], "fit_s_full": base["fit_s_full"],
        "train_macro_f1": base["train_macro_f1"],
    }
    comp = pd.DataFrame({"untuned": untuned, "tuned": tuned})
    comp["delta"] = comp["tuned"] - comp["untuned"]
    comp.to_csv(OPT / "tuned_vs_untuned.csv")
    log("\n" + comp.to_string())

    use_tuned = tuned["cv_macro_f1"] >= untuned["cv_macro_f1"]
    final_name = f"{name}_tuned" if use_tuned else name
    src = ROOT / "artifacts" / "models" / (f"{name}_tuned.joblib" if use_tuned else f"experiments/{name}.joblib")
    final_model = joblib.load(src)
    joblib.dump(final_model, ROOT / "artifacts" / "models" / "final_classifier.joblib", compress=3)
    write_json(OPT / "best_params.json", {"model": name, "best_trial": study.best_trial.number,
                                          "best_value_search": study.best_value, "best_params": best_params})
    write_json(OPT / "optimisation_summary.json", {
        **run_metadata(), "model": name, "objective": "mean CV Macro-F1 (5 trip-grouped folds)",
        "sampler": "TPE (n_startup_trials=10, seed=42)", "pruner": "MedianPruner(n_startup_trials=5, n_warmup_steps=1)",
        "n_trials": n_trials, "n_complete": int((trials["state"] == "COMPLETE").sum()),
        "n_pruned": int((trials["state"] == "PRUNED").sum()), "search_seconds": search_s,
        "search_obs_stride": cfg["search_obs_stride"], "best_params": best_params,
        "param_importance": importance, "tuned": tuned, "untuned": untuned,
        "selected_final": final_name, "selection_rule": "tuned if its CV Macro-F1 >= untuned CV Macro-F1",
    })
    write_json(ROOT / "artifacts" / "metadata" / "classifier.json", {
        **run_metadata(), "model_name": name, "variant": final_name,
        "params": best_params if use_tuned else mcfg["models"][name]["params"],
        "feature_version": FEATURE_VERSION, "feature_columns": FEATURE_COLUMNS, "obs_stride": stride,
        "cv_macro_f1": tuned["cv_macro_f1"] if use_tuned else untuned["cv_macro_f1"],
        "student_test_top1": tuned["st_top1"] if use_tuned else untuned["st_top1"],
        "oof_file": f"oof_{name}_tuned.npy" if use_tuned else f"oof_{name}.npy",
        "student_probs_file": f"student_probs_{name}_tuned.npy" if use_tuned else f"student_probs_{name}.npy",
    })
    write_json(ROOT / "artifacts" / "metadata" / "feature_config.json", fcfg)   # exact settings used at training
    plots(trials, importance, comp)
    log(f"=== final classifier: {final_name}")


def plots(trials, importance, comp):
    done = trials[trials["state"] == "COMPLETE"].sort_values("number")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].scatter(done["number"], done["value"], s=18, label="completed trial")
    ax[0].plot(done["number"], done["value"].cummax(), color="#e8833a", label="best so far")
    pr = trials[trials["state"] == "PRUNED"]
    if len(pr):
        ax[0].scatter(pr["number"], [done["value"].min()] * len(pr), marker="x", color="#999", s=14, label="pruned")
    ax[0].set(title="Optuna TPE optimisation history", xlabel="trial", ylabel="CV Macro-F1")
    ax[0].legend()
    if isinstance(importance, dict) and "error" not in importance:
        k = list(importance.keys())[::-1]
        ax[1].barh(k, [importance[x] for x in k], color="#2e9e6b")
        ax[1].set(title="Hyperparameter importance (fANOVA)", xlabel="importance")
    save(fig, FIG / "optimisation_history_importance.png")
    fig, ax = plt.subplots(figsize=(7, 3.6))
    rows = ["cv_macro_f1", "cv_top1", "st_macro_f1", "st_top1"]
    x = np.arange(len(rows))
    ax.bar(x - 0.2, comp.loc[rows, "untuned"], 0.4, label="untuned")
    ax.bar(x + 0.2, comp.loc[rows, "tuned"], 0.4, label="tuned")
    lo = min(comp.loc[rows, ["untuned", "tuned"]].min()) - 0.02
    ax.set(xticks=x, xticklabels=rows, ylim=(lo, 1.0), title="Untuned vs tuned (same folds / same test trips)")
    ax.legend()
    save(fig, FIG / "optimisation_tuned_vs_untuned.png")


if __name__ == "__main__":
    main()
