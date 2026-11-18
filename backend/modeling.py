from __future__ import annotations

import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from .data import Trip
from .features import candidate_features, FEATURE_NAMES
from .decoder import viterbi, continuity


def model_zoo(seed: int = 42) -> dict:
    return {
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, class_weight="balanced", random_state=seed)),
        "gaussian_naive_bayes": make_pipeline(StandardScaler(), GaussianNB()),
        "random_forest": RandomForestClassifier(n_estimators=40, max_depth=18, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=1, random_state=seed),
        "extra_trees": ExtraTreesClassifier(n_estimators=40, max_depth=18, min_samples_leaf=2, class_weight="balanced", n_jobs=1, random_state=seed),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=40, learning_rate=0.1, max_depth=2, random_state=seed),
    }


def _build(trips: list[Trip], k: int = 8, max_observations_per_trip: int = 100):
    records = []
    for trip in trips:
        ids, feats, labels, misses = candidate_features(trip, k=k)
        sample = np.unique(np.linspace(0, len(ids) - 1, min(len(ids), max_observations_per_trip), dtype=int))
        for i in sample:
            candidates = ids[i]
            if misses[i]:
                continue
            y = (candidates == labels[i]).astype(np.int8)
            records.append((trip, i, candidates, feats[i], y, labels[i], bool(misses[i])))
    if not records:
        raise ValueError("No observations had their reference route arc in the candidate set")
    X = np.vstack([r[3] for r in records])
    y = np.concatenate([r[4] for r in records])
    groups = np.concatenate([np.full(len(r[2]), r[0].trip_id) for r in records])
    return X, y, groups, records


def _preds(model, X, records):
    probabilities = model.predict_proba(X)[:, 1]
    offset, pred, true, trip_ids, misses = 0, [], [], [], []
    for trip, i, candidates, feats, y, label, miss in records:
        p = probabilities[offset:offset + len(candidates)]
        offset += len(candidates)
        pred.append(int(candidates[int(np.argmax(p))]))
        true.append(int(label))
        trip_ids.append(trip.trip_id)
        misses.append(miss)
    return np.asarray(pred), np.asarray(true), np.asarray(trip_ids), np.asarray(misses)


def _score(model, X, y, groups, records, folds: int = 3):
    splitter = GroupKFold(n_splits=min(folds, len(np.unique(groups))))
    all_pred, all_true = [], []
    for tr, va in splitter.split(X, y, groups):
        model.fit(X[tr], y[tr])
        # Each observation has candidate rows kept contiguous; reconstruct the held-out rows
        # using the per-trip fold fit below to preserve candidate ranking.
        va_groups = set(groups[va])
        fold_records = [r for r in records if r[0].trip_id in va_groups]
        fold_X = np.vstack([r[3] for r in fold_records])
        pred, true, _, _ = _preds(model, fold_X, fold_records)
        all_pred.extend(pred.tolist()); all_true.extend(true.tolist())
    return float(f1_score(all_true, all_pred, average="macro", zero_division=0))


def _bayes_rf(X, y, groups, records, seed: int = 42, trials: int = 8):
    """Small Gaussian-process expected-improvement search over RF depth/leaf size."""
    rng = np.random.default_rng(seed)
    seen, scores, logs = [], [], []
    def objective(params):
        depth, leaf = int(round(params[0])), int(round(params[1]))
        candidate = RandomForestClassifier(n_estimators=70, max_depth=depth, min_samples_leaf=leaf,
            class_weight="balanced_subsample", n_jobs=1, random_state=seed)
        score = _score(candidate, X, y, groups, records, folds=3)
        return score
    for i in range(trials):
        if i < 3:
            point = [int(rng.integers(8, 31)), int(rng.integers(1, 6))]
        else:
            gp = GaussianProcessRegressor(kernel=ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(1e-4), normalize_y=True, random_state=seed)
            gp.fit(np.asarray(seen, dtype=float), np.asarray(scores))
            candidates = np.column_stack((rng.integers(8, 31, size=80), rng.integers(1, 6, size=80)))
            mu, sd = gp.predict(candidates, return_std=True)
            best = max(scores)
            z = (mu - best - 0.01) / np.maximum(sd, 1e-9)
            from scipy.stats import norm
            ei = (mu - best - 0.01) * norm.cdf(z) + sd * norm.pdf(z)
            for old in seen:
                ei[np.all(candidates == old, axis=1)] = -np.inf
            point = candidates[int(np.argmax(ei))].tolist()
        score = objective(point)
        seen.append(point); scores.append(score)
        logs.append({"trial": i + 1, "max_depth": int(point[0]), "min_samples_leaf": int(point[1]), "cv_macro_f1": score})
    best_i = int(np.argmax(scores))
    return {"max_depth": int(seen[best_i][0]), "min_samples_leaf": int(seen[best_i][1])}, logs


def _test_eval(model, trips: list[Trip], model_name: str = "model", k: int = 8, prepared=None):
    rows = []
    for trip in trips:
        ids, feats, labels, misses = prepared[trip.trip_id] if prepared is not None else candidate_features(trip, k=k)
        started = time.perf_counter()
        probs = [model.predict_proba(x)[:, 1] for x in feats]
        inference = time.perf_counter() - started
        direct = np.asarray([int(ids[i][int(np.argmax(probs[i]))]) for i in range(len(ids))])
        decoded = viterbi(ids, probs, trip.arcs)
        binary_y = np.concatenate([(ids[i] == labels[i]).astype(np.int8) for i in range(len(ids))])
        binary_p = np.concatenate(probs)
        candidate_log_loss = float(log_loss(binary_y, np.column_stack((1 - binary_p, binary_p)), labels=[0, 1]))
        for name, pred in (("classifier", direct), ("classifier_hmm", decoded)):
            rows.append({"trip_id": trip.trip_id, "model": model_name, "method": name,
                "top1_accuracy": float(accuracy_score(labels, pred)),
                "macro_f1": float(f1_score(labels, pred, average="macro", zero_division=0)),
                "route_continuity": continuity(pred, trip.arcs),
                "candidate_miss_rate": float(np.mean(misses)),
                "candidate_log_loss": candidate_log_loss if name == "classifier" else None,
                "inference_seconds": inference})
    return pd.DataFrame(rows)


def train(trips: list[Trip], output_dir: str | Path, results_dir: str | Path, bayes_trials: int = 8):
    train_trips = [t for t in trips if t.partition.lower() in {"train", "training"}]
    test_trips = [t for t in trips if t.partition.lower() in {"student_test", "student_testing"}]
    if len(train_trips) < 5 or not test_trips:
        raise ValueError("Expected the 80-trip train and 10-trip student_test partitions")
    output_dir, results_dir = Path(output_dir), Path(results_dir)
    output_dir.mkdir(parents=True, exist_ok=True); results_dir.mkdir(parents=True, exist_ok=True)
    X, y, groups, records = _build(train_trips)
    comparison = []
    test_rows = []
    test_features = {trip.trip_id: candidate_features(trip, k=8) for trip in test_trips}
    for name, model in model_zoo().items():
        start = time.perf_counter()
        cv_f1 = _score(model, X, y, groups, records)
        model.fit(X, y)
        elapsed = time.perf_counter() - start
        comparison.append({"model": name, "cv_macro_f1": cv_f1, "fit_seconds": elapsed})
        test_rows.extend(_test_eval(model, test_trips, model_name=name, prepared=test_features).to_dict(orient="records"))
    pd.DataFrame(comparison).to_csv(results_dir / "model_comparison.csv", index=False)
    tuned, trials = _bayes_rf(X, y, groups, records, trials=bayes_trials)
    (results_dir / "bayesian_trials.json").write_text(json.dumps(trials, indent=2))
    tuned_fit_started = time.perf_counter()
    selected = RandomForestClassifier(n_estimators=60, class_weight="balanced_subsample", n_jobs=1,
        random_state=42, **tuned)
    selected.fit(X, y)
    tuned_fit_seconds = time.perf_counter() - tuned_fit_started
    joblib.dump({"model": selected, "feature_names": FEATURE_NAMES, "candidate_k": 8,
                 "parameters": tuned}, output_dir / "pathsnap.joblib")
    evaluation = pd.DataFrame(test_rows + _test_eval(selected, test_trips, model_name="tuned_random_forest", prepared=test_features).to_dict(orient="records"))
    evaluation.to_csv(results_dir / "student_test_metrics.csv", index=False)
    cv_table = pd.DataFrame(comparison)
    cv_table = pd.concat([cv_table, pd.DataFrame([{"model": "tuned_random_forest",
        "cv_macro_f1": max(t["cv_macro_f1"] for t in trials), "fit_seconds": tuned_fit_seconds}])], ignore_index=True)
    test_table = evaluation.groupby(["model", "method"], as_index=False).mean(numeric_only=True)
    consolidated = test_table.merge(cv_table, on="model", how="left")
    consolidated.to_csv(results_dir / "model_comparison.csv", index=False)
    return comparison, trials, evaluation
