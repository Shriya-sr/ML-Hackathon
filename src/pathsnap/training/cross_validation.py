"""Trip-grouped cross-validation.

All candidate rows of a trip stay in the same fold (GroupKFold, group = trip_id),
so the validation score measures generalisation to *unseen trips/cities*, not
to unseen seconds of a trip already used for training.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from ..config import SEED
from ..evaluation.metrics import classification_metrics, select_top1, top1_accuracy
from ..features.feature_schema import FEATURE_COLUMNS, to_matrix
from ..models.model_factory import build_model, positive_proba


def trip_folds(trip_ids, n_splits: int = 5, seed: int = SEED) -> list[tuple[np.ndarray, np.ndarray]]:
    trips = np.array(sorted(set(trip_ids)))
    gkf = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(trips[a], trips[b]) for a, b in gkf.split(trips, groups=trips)]


def training_rows(cands: pd.DataFrame, stride: int = 1) -> pd.DataFrame:
    """Supervised rows: candidates of labelled fixes, optionally every ``stride``-th fix."""
    m = cands["true_arc"].to_numpy() >= 0
    if stride > 1:
        m &= (cands["obs"].to_numpy() % stride) == 0
    return cands[m]


def subsample_rows(df: pd.DataFrame, max_rows: int | None, seed: int = SEED) -> pd.DataFrame:
    """Trip-stratified subsample that keeps whole fixes (all their candidates)."""
    if not max_rows or len(df) <= max_rows:
        return df
    keys = df[["trip_id", "obs"]].drop_duplicates()
    frac = max_rows / len(df)
    rng = np.random.default_rng(seed)
    keep = keys.groupby("trip_id", group_keys=False).apply(
        lambda g: g.sample(frac=frac, random_state=int(rng.integers(1 << 31))))
    return df.merge(keep, on=["trip_id", "obs"])


def fit_model(name, params, extra, train_df, feature_cols=None, seed=SEED):
    feature_cols = feature_cols or FEATURE_COLUMNS
    tr = subsample_rows(train_df, extra.get("max_train_rows"), seed)
    X, y = to_matrix(tr, feature_cols), tr["label"].to_numpy().astype(int)
    model = build_model(name, params, extra, seed)
    t0 = time.perf_counter()
    model.fit(X, y)
    return model, time.perf_counter() - t0, len(tr)


def evaluate_model(model, cands_eval: pd.DataFrame, obs_eval: pd.DataFrame, feature_cols=None, threshold=0.5):
    feature_cols = feature_cols or FEATURE_COLUMNS
    X = to_matrix(cands_eval, feature_cols)
    t0 = time.perf_counter()
    p = positive_proba(model, X)
    infer_s = time.perf_counter() - t0
    lab = cands_eval["true_arc"].to_numpy() >= 0
    cm = classification_metrics(cands_eval["label"].to_numpy()[lab], p[lab], threshold)
    pred = select_top1(cands_eval, p)
    t1 = top1_accuracy(obs_eval, pred)
    n_obs = int(cands_eval[["trip_id", "obs"]].drop_duplicates().shape[0])
    return {**cm, **t1, "inference_s": infer_s, "inference_ms_per_1k_fixes": infer_s * 1e6 / max(n_obs, 1),
            "n_eval_rows": int(len(cands_eval))}, p, pred


def run_cv(name, params, extra, cands, obs, folds, stride=2, feature_cols=None, threshold=0.5,
           seed=SEED, keep_oof=False, eval_train=True, log=print):
    rows, oof = [], (np.full(len(cands), np.nan) if keep_oof else None)
    for k, (tr_trips, va_trips) in enumerate(folds):
        tr_mask = cands["trip_id"].isin(tr_trips).to_numpy()
        va_mask = cands["trip_id"].isin(va_trips).to_numpy()
        train_df = training_rows(cands[tr_mask], stride)
        model, fit_s, n_fit = fit_model(name, params, extra, train_df, feature_cols, seed)
        va_c = cands[va_mask]
        va_o = obs[obs["trip_id"].isin(va_trips)]
        m_va, p_va, _ = evaluate_model(model, va_c, va_o, feature_cols, threshold)
        rec = {"model": name, "fold": k, "n_train_trips": len(tr_trips), "n_val_trips": len(va_trips),
               "n_fit_rows": n_fit, "fit_s": fit_s, **{f"val_{a}": b for a, b in m_va.items()}}
        if eval_train:
            # training-fold score on the rows actually used for fitting -> generalisation gap
            tr_obs = obs[obs["trip_id"].isin(tr_trips)]
            tr_obs = tr_obs[tr_obs["obs"] % stride == 0] if stride > 1 else tr_obs
            m_tr, _, _ = evaluate_model(model, train_df, tr_obs, feature_cols, threshold)
            rec.update({f"train_{a}": m_tr[a] for a in ("macro_f1", "log_loss", "top1_accuracy", "accuracy")})
        rows.append(rec)
        if keep_oof:
            oof[va_mask] = p_va
        log(f"  [{name}] fold {k}: val macroF1={m_va['macro_f1']:.4f} top1={m_va['top1_accuracy']:.4f} "
            f"logloss={m_va['log_loss']:.4f} fit={fit_s:.1f}s")
    return pd.DataFrame(rows), oof
