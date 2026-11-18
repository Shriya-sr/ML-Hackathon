"""Feature importance and group ablation (training trips only).

* Permutation importance: HistGradientBoosting fitted on the training folds,
  importance measured on the held-out trips of each fold (drop in log loss).
* Group ablation: remove one feature group at a time and re-run the 5-fold
  trip-grouped CV (Macro-F1 and Top-1).
The student-test and evaluator partitions are not used here, so feature
selection cannot leak test information.

Usage:  python scripts/feature_analysis.py
Outputs: experiments/results/feature_ablation.csv, permutation_importance.csv,
         reports/figures/feature_importance.png, feature_ablation.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathsnap.config import SEED, load_config, processed_dir  # noqa: E402
from pathsnap.features.feature_schema import FEATURE_COLUMNS, FEATURE_GROUPS, to_matrix  # noqa: E402
from pathsnap.training.cross_validation import fit_model, run_cv, training_rows, trip_folds  # noqa: E402
from pathsnap.visualisation.plots import plt, save  # noqa: E402

RES = ROOT / "experiments" / "results"
FIG = ROOT / "reports" / "figures"
MODEL = "hist_gradient_boosting"


def main():
    mcfg = load_config("models")
    params = mcfg["models"][MODEL]["params"]
    pdir = processed_dir()
    c, o = pd.read_pickle(pdir / "train_candidates.pkl"), pd.read_pickle(pdir / "train_obs.pkl")
    folds = trip_folds(c["trip_id"].unique(), mcfg["cv"]["n_splits"], mcfg["cv"]["seed"])

    # permutation importance on held-out trips of every fold
    rng = np.random.default_rng(SEED)
    imps = []
    for k, (tr, va) in enumerate(folds):
        model, _, _ = fit_model(MODEL, params, {}, training_rows(c[c.trip_id.isin(tr)], 2))
        v = training_rows(c[c.trip_id.isin(va)], 1)
        v = v.iloc[rng.choice(len(v), size=min(60000, len(v)), replace=False)]
        r = permutation_importance(model, to_matrix(v), v["label"].to_numpy(), scoring="neg_log_loss",
                                   n_repeats=3, random_state=SEED, n_jobs=1)
        imps.append(r.importances_mean)
        print(f"permutation importance fold {k} done", flush=True)
    imp = pd.DataFrame({"feature": FEATURE_COLUMNS, "importance_mean": np.mean(imps, 0), "importance_std": np.std(imps, 0)})
    imp["group"] = imp["feature"].map({f: g for g, fs in FEATURE_GROUPS.items() for f in fs})
    imp = imp.sort_values("importance_mean", ascending=False)
    imp.to_csv(RES / "permutation_importance.csv", index=False)

    # group ablation
    rows = []
    for drop in [None] + list(FEATURE_GROUPS):
        cols = [f for f in FEATURE_COLUMNS if drop is None or f not in FEATURE_GROUPS[drop]]
        df, _ = run_cv(MODEL, params, {}, c, o, folds, stride=4, feature_cols=cols, eval_train=False,
                       log=lambda *_: None)
        rows.append({"dropped_group": drop or "(none - full set)", "n_features": len(cols),
                     "cv_macro_f1": df["val_macro_f1"].mean(), "cv_macro_f1_std": df["val_macro_f1"].std(),
                     "cv_top1": df["val_top1_accuracy"].mean(), "cv_log_loss": df["val_log_loss"].mean()})
        print(rows[-1], flush=True)
    ab = pd.DataFrame(rows)
    full = ab.iloc[0]
    ab["delta_macro_f1"] = ab["cv_macro_f1"] - full["cv_macro_f1"]
    ab["delta_top1"] = ab["cv_top1"] - full["cv_top1"]
    ab.to_csv(RES / "feature_ablation.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {"geometric": "#2a6fdb", "heading": "#e8833a", "context": "#2e9e6b", "kinematic": "#c94f7c", "network": "#7b61c9"}
    s = imp.iloc[::-1]
    ax.barh(s["feature"], s["importance_mean"], xerr=s["importance_std"], color=[colors[g] for g in s["group"]])
    ax.set(title="Permutation importance (increase in log loss when shuffled)\nHGB, held-out training trips, 5 folds",
           xlabel="importance")
    for g, col in colors.items():
        ax.barh([], [], color=col, label=g)
    ax.legend(loc="lower right")
    save(fig, FIG / "feature_importance.png")
    fig, ax = plt.subplots(figsize=(8, 3.6))
    a = ab.iloc[1:]
    x = np.arange(len(a))
    ax.bar(x - 0.2, 100 * a["delta_macro_f1"], 0.4, label="Δ CV Macro-F1")
    ax.bar(x + 0.2, 100 * a["delta_top1"], 0.4, label="Δ CV Top-1")
    ax.axhline(0, color="k", lw=0.8)
    ax.set(xticks=x, xticklabels=[f"−{g}" for g in a["dropped_group"]], ylabel="percentage points",
           title="Group ablation: change vs full feature set (negative = group helps)")
    ax.legend()
    save(fig, FIG / "feature_ablation.png")


if __name__ == "__main__":
    main()
