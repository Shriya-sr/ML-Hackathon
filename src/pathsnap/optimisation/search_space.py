"""Hyperparameter search spaces (justification in docs/methodology.md, section 7).

Ranges are centred on the baseline configuration and cover the parameters that
control model capacity (depth/leaves/iterations), the learning rate, and
regularisation, which together govern the bias-variance trade-off that the
fold-level results showed (small train/validation gap -> room for more capacity).
"""
from __future__ import annotations


def suggest(trial, model: str) -> dict:
    if model == "hist_gradient_boosting":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
            "max_iter": trial.suggest_int("max_iter", 100, 800, step=50),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 127, log=True),
            "max_depth": trial.suggest_categorical("max_depth", [None, 6, 10, 16]),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 400, log=True),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-4, 10.0, log=True),
            "max_features": trial.suggest_float("max_features", 0.5, 1.0),
            "early_stopping": True, "validation_fraction": 0.1, "n_iter_no_change": 20,
        }
    if model in ("random_forest", "extra_trees"):
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
            "max_depth": trial.suggest_categorical("max_depth", [None, 12, 20, 30]),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 50, log=True),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
            "n_jobs": -1,
        }
    if model == "logistic_regression":
        return {"C": trial.suggest_float("C", 1e-3, 1e2, log=True), "max_iter": 3000,
                "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"])}
    if model == "mlp":
        return {
            "hidden_layer_sizes": trial.suggest_categorical("hidden_layer_sizes", ["64", "64-32", "128-64", "128-64-32"]),
            "alpha": trial.suggest_float("alpha", 1e-6, 1e-2, log=True),
            "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
            "max_iter": 200, "early_stopping": True, "validation_fraction": 0.1, "n_iter_no_change": 10,
            "batch_size": 512,
        }
    raise KeyError(f"no search space for {model}")


def normalise(model: str, params: dict) -> dict:
    p = dict(params)
    if model == "mlp" and isinstance(p.get("hidden_layer_sizes"), str):
        p["hidden_layer_sizes"] = [int(x) for x in p["hidden_layer_sizes"].split("-")]
    return p
