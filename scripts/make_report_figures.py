"""Model-comparison figures, generated only from saved experiment records.

Usage:  python scripts/make_report_figures.py
Reads:  experiments/results/model_summary.csv, cv_folds_*.csv, training_history_*.csv
Writes: reports/figures/model_*.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathsnap.visualisation.plots import plt, save  # noqa: E402

RES = ROOT / "experiments" / "results"
FIG = ROOT / "reports" / "figures"


def main():
    s = pd.read_csv(RES / "model_summary.csv").sort_values("cv_macro_f1", ascending=False)
    names = s["display"].tolist()
    x = np.arange(len(s))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x, s["cv_macro_f1"], yerr=s["cv_macro_f1_std"], capsize=4, color="#2a6fdb")
    for i, v in enumerate(s["cv_macro_f1"]):
        ax.text(i, v + 0.003, f"{v:.4f}", ha="center", fontsize=8)
    ax.set(xticks=x, xticklabels=names, ylim=(0.85, 0.97), ylabel="Macro-F1",
           title="Cross-validation Macro-F1 (5 trip-grouped folds, mean ± std)")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    save(fig, FIG / "model_cv_macro_f1.png")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - 0.2, s["st_top1"], 0.4, label="Top-1 accuracy (fixes)", color="#2e9e6b")
    ax.bar(x + 0.2, s["st_macro_f1"], 0.4, label="Macro-F1 (candidates)", color="#e8833a")
    ax.set(xticks=x, xticklabels=names, ylim=(0.8, 0.97), title="Student-testing partition (10 trips), classifier only")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    ax.legend()
    save(fig, FIG / "model_student_top1_macro_f1.png")

    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.bar(x - 0.2, s["cv_log_loss"], 0.4, label="CV", color="#7b61c9")
    ax.bar(x + 0.2, s["st_log_loss"], 0.4, label="student-test", color="#c94f7c")
    ax.set(xticks=x, xticklabels=names, ylabel="log loss (lower is better)", title="Log loss (probability quality)")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    ax.legend()
    save(fig, FIG / "model_log_loss.png")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - 0.2, s["train_macro_f1"], 0.4, label="training folds", color="#999999")
    ax.bar(x + 0.2, s["cv_macro_f1"], 0.4, label="validation folds", color="#2a6fdb")
    for i, g in enumerate(s["gap_macro_f1"]):
        ax.text(i, max(s["train_macro_f1"].iloc[i], s["cv_macro_f1"].iloc[i]) + 0.004, f"gap {g:+.3f}", ha="center", fontsize=8)
    ax.set(xticks=x, xticklabels=names, ylim=(0.85, 1.01), ylabel="Macro-F1",
           title="Generalisation gap: training-fold vs validation-fold Macro-F1")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    ax.legend(loc="lower left")
    save(fig, FIG / "model_generalisation_gap.png")

    fig, ax = plt.subplots(figsize=(10, 4))
    for name, disp in zip(s["model"], names):
        f = pd.read_csv(RES / f"cv_folds_{name}.csv")
        ax.plot(f["fold"], f["val_macro_f1"], marker="o", label=disp)
    ax.set(xlabel="fold", ylabel="validation Macro-F1", xticks=range(5),
           title="Fold-level stability (same folds for every model)")
    ax.legend(fontsize=8, ncol=2)
    save(fig, FIG / "model_fold_stability.png")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(s["inference_ms_per_1k_fixes"], s["st_top1"], s=60, color="#2a6fdb")
    for _, r in s.iterrows():
        ax.annotate(r["display"], (r["inference_ms_per_1k_fixes"], r["st_top1"]), fontsize=8,
                    xytext=(4, 4), textcoords="offset points")
    ax.set(xscale="log", xlabel="inference time (ms per 1000 fixes, log)", ylabel="student-test Top-1",
           title="Accuracy vs inference cost")
    save(fig, FIG / "model_accuracy_vs_cost.png")

    n = len(s)
    cols = 4
    fig, axes = plt.subplots(int(np.ceil(n / cols)), cols, figsize=(13, 3.2 * int(np.ceil(n / cols))))
    for ax, (_, r) in zip(axes.ravel(), s.iterrows()):
        cm = np.array([[r["st_tn"], r["st_fp"]], [r["st_fn"], r["st_tp"]]], float)
        cmn = cm / cm.sum(1, keepdims=True)
        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, f"{int(v)}\n({cmn[i, j]:.2f})", ha="center", va="center", fontsize=8,
                    color="white" if cmn[i, j] > 0.5 else "black")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["pred 0", "pred 1"], yticklabels=["true 0", "true 1"],
               title=r["display"])
        ax.grid(False)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Confusion matrices on student-test candidate rows (threshold 0.5)")
    save(fig, FIG / "model_confusion_matrices.png")

    # epoch-wise histories (only models that really produce them)
    hist_files = {p.stem.replace("training_history_", ""): p for p in RES.glob("training_history_*.csv")}
    if hist_files:
        fig, axes = plt.subplots(1, len(hist_files), figsize=(6 * len(hist_files), 3.6))
        for ax, (name, p) in zip(np.atleast_1d(axes), sorted(hist_files.items())):
            h = pd.read_csv(p)
            if "train_loss" in h:
                ax.plot(h["epoch"], h["train_loss"], label="training loss (log loss + L2)")
                ax2 = ax.twinx()
                ax2.plot(h["epoch"], h["val_accuracy"], color="#e8833a", label="validation accuracy (10% hold-out)")
                ax2.set_ylabel("validation accuracy")
                ax2.grid(False)
                ax.set(xlabel="epoch", ylabel="training loss", title=f"{name}: epoch history (final fit)")
                ax.legend(loc="upper center", fontsize=8); ax2.legend(loc="center right", fontsize=8)
            else:
                ax.plot(h["iteration"], -h["train_neg_log_loss"], label="training log loss")
                ax.plot(h["iteration"], -h["val_neg_log_loss"], label="validation log loss (10% hold-out)")
                ax.set(xlabel="boosting iteration", ylabel="log loss", title=f"{name}: boosting history (final fit)")
                ax.legend(fontsize=8)
        save(fig, FIG / "model_training_histories.png")
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
