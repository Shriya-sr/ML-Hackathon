"""Candidate-level and observation-level metrics.

Two different questions are measured:

* Candidate classification (rows = (fix, candidate arc) pairs, binary label):
  accuracy, precision, recall, F1 (positive class and macro), log loss, ROC-AUC,
  confusion matrix at threshold 0.5. Heavily imbalanced (~1 positive per ~9 rows),
  so accuracy alone is misleading and Macro-F1 is reported.
* Map matching (rows = GPS fixes): Top-1 accuracy = the highest-scoring
  candidate equals the reference arc. The denominator is every labelled fix,
  including fixes whose reference arc was not generated (candidate miss) or that
  received no candidates, which therefore count as errors.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, brier_score_loss, confusion_matrix, f1_score, log_loss,
                             precision_score, recall_score, roc_auc_score)


def classification_metrics(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict:
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p, float), 1e-7, 1 - 1e-7)
    yhat = (p >= threshold).astype(int)
    cm = confusion_matrix(y, yhat, labels=[0, 1])
    out = {
        "accuracy": accuracy_score(y, yhat),
        "precision_pos": precision_score(y, yhat, zero_division=0),
        "recall_pos": recall_score(y, yhat, zero_division=0),
        "f1_pos": f1_score(y, yhat, zero_division=0),
        "precision_macro": precision_score(y, yhat, average="macro", zero_division=0),
        "recall_macro": recall_score(y, yhat, average="macro", zero_division=0),
        "macro_f1": f1_score(y, yhat, average="macro", zero_division=0),
        "log_loss": log_loss(y, p, labels=[0, 1]),
        "brier": brier_score_loss(y, p),
        "roc_auc": roc_auc_score(y, p) if len(np.unique(y)) == 2 else float("nan"),
        "tn": int(cm[0, 0]), "fp": int(cm[0, 1]), "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
    }
    return out


def select_top1(cands: pd.DataFrame, score: np.ndarray) -> pd.DataFrame:
    """Return one row per (trip_id, obs): the arc with the highest score."""
    df = cands[["trip_id", "obs", "arc"]].copy()
    df["score"] = score
    df = df.sort_values(["trip_id", "obs", "score"], ascending=[True, True, False], kind="mergesort")
    return df.drop_duplicates(["trip_id", "obs"], keep="first").reset_index(drop=True)


def top1_accuracy(obs: pd.DataFrame, pred: pd.DataFrame) -> dict:
    """``obs``: per-fix table with trip_id, obs, label (reference arc or -1).
    ``pred``: per-fix predicted arc (trip_id, obs, arc). Missing predictions count as wrong."""
    lab = obs[obs["label"] >= 0][["trip_id", "obs", "label"]]
    m = lab.merge(pred[["trip_id", "obs", "arc"]], on=["trip_id", "obs"], how="left")
    correct = (m["arc"].to_numpy() == m["label"].to_numpy())
    has_pred = m["arc"].notna().to_numpy()
    return {"top1_accuracy": float(correct.mean()) if len(m) else float("nan"),
            "n_labelled_obs": int(len(m)), "n_correct": int(correct.sum()),
            "n_no_prediction": int((~has_pred).sum())}


def per_trip_top1(obs: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    lab = obs[obs["label"] >= 0][["trip_id", "obs", "label"]]
    m = lab.merge(pred[["trip_id", "obs", "arc"]], on=["trip_id", "obs"], how="left")
    m["correct"] = m["arc"].to_numpy() == m["label"].to_numpy()
    return m.groupby("trip_id")["correct"].mean().rename("top1_accuracy").reset_index()


def selection_macro_f1(cands: pd.DataFrame, pred: pd.DataFrame) -> float:
    """Macro-F1 of the *decision* (selected candidate = 1, others = 0) against the
    candidate labels. Lets classifier-only and HMM outputs be compared on the
    same candidate rows even though the HMM outputs no threshold-able scores."""
    lab = cands[cands["true_arc"] >= 0]
    sel = lab[["trip_id", "obs", "arc"]].merge(pred[["trip_id", "obs", "arc"]].assign(_sel=1),
                                                on=["trip_id", "obs", "arc"], how="left")
    yhat = sel["_sel"].fillna(0).to_numpy().astype(int)
    return float(f1_score(lab["label"].to_numpy().astype(int), yhat, average="macro", zero_division=0))
