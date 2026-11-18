"""Model zoo. Every model is a scikit-learn estimator exposing ``predict_proba``
whose column order is checked against ``classes_`` before use.

Scale-sensitive models (LR, KNN, SVM, MLP) wrap a StandardScaler inside a
Pipeline so the scaler is re-fitted on the training fold only.
"""
from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ..config import SEED

MODEL_CARDS = {
    "logistic_regression": {
        "display": "Logistic Regression", "family": "Linear / probabilistic",
        "loss": "Binary cross-entropy (log loss) + L2 penalty, lbfgs",
        "why": "Interpretable linear baseline with naturally calibrated probabilities that plug directly into HMM emissions.",
    },
    "gaussian_nb": {
        "display": "Gaussian Naive Bayes", "family": "Probabilistic (generative)",
        "loss": "Maximum-likelihood class-conditional Gaussians (no iterative loss)",
        "why": "Cheapest probabilistic baseline; tests whether independent per-feature evidence is enough.",
    },
    "knn": {
        "display": "k-Nearest Neighbours", "family": "Instance-based",
        "loss": "None (lazy learner); probability = distance-weighted vote of k neighbours",
        "why": "Non-parametric similarity baseline: candidates that look like past correct candidates are correct.",
    },
    "random_forest": {
        "display": "Random Forest", "family": "Ensemble (bagging)",
        "loss": "Gini impurity per split, bootstrap aggregation",
        "why": "Captures non-linear interactions (distance x heading x context) with low variance from bagging.",
    },
    "extra_trees": {
        "display": "Extra Trees", "family": "Ensemble (randomised trees)",
        "loss": "Gini impurity with random split thresholds",
        "why": "More randomisation than RF -> lower variance and faster training; checks whether RF overfits.",
    },
    "hist_gradient_boosting": {
        "display": "HistGradientBoosting", "family": "Boosting",
        "loss": "Binary log loss minimised by gradient boosting of histogram trees",
        "why": "Strong tabular learner, optimises log loss directly (good probabilities for the HMM), fast on ~1M rows.",
    },
    "svm_rbf": {
        "display": "RBF SVM (Platt-calibrated)", "family": "Kernel-based",
        "loss": "Hinge loss + L2 margin; probabilities via sigmoid (Platt) calibration",
        "why": "Smooth non-linear boundary in scaled feature space; tests kernel suitability on this geometry.",
    },
    "mlp": {
        "display": "MLP (64-32)", "family": "Neural network",
        "loss": "Binary cross-entropy minimised with Adam, L2 penalty alpha",
        "why": "Learns non-linear feature combinations; the only model with epoch-wise loss curves.",
    },
}


def build_model(name: str, params: dict | None = None, extra: dict | None = None, seed: int = SEED):
    p = dict(params or {})
    extra = extra or {}
    if name == "logistic_regression":
        return Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(random_state=seed, **p))])
    if name == "gaussian_nb":
        return GaussianNB(**p)
    if name == "knn":
        return Pipeline([("scale", StandardScaler()), ("clf", KNeighborsClassifier(n_jobs=-1, **p))])
    if name == "random_forest":
        return RandomForestClassifier(random_state=seed, **p)
    if name == "extra_trees":
        return ExtraTreesClassifier(random_state=seed, **p)
    if name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(random_state=seed, **p)
    if name == "svm_rbf":
        svc = Pipeline([("scale", StandardScaler()), ("clf", SVC(random_state=seed, **p))])
        return CalibratedClassifierCV(svc, method=extra.get("calibration", "sigmoid"), cv=3, ensemble=False)
    if name == "mlp":
        if "hidden_layer_sizes" in p:
            p["hidden_layer_sizes"] = tuple(p["hidden_layer_sizes"])
        return Pipeline([("scale", StandardScaler()), ("clf", MLPClassifier(random_state=seed, **p))])
    raise KeyError(f"unknown model: {name}")


def positive_proba(model, X: np.ndarray) -> np.ndarray:
    """P(label=1) with explicit class-column alignment."""
    proba = model.predict_proba(X)
    classes = list(model.classes_)
    if proba.shape != (len(X), len(classes)):
        raise ValueError(f"probability shape {proba.shape} does not match ({len(X)}, {len(classes)})")
    if 1 not in classes:
        return np.zeros(len(X))
    return proba[:, classes.index(1)]
