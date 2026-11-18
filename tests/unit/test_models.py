import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from pathsnap.features.feature_schema import FEATURE_COLUMNS
from pathsnap.models.model_factory import MODEL_CARDS, build_model, positive_proba


@pytest.mark.parametrize("name", list(MODEL_CARDS))
def test_every_model_builds_and_predicts_probabilities(name):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, len(FEATURE_COLUMNS)))
    y = (X[:, 0] + 0.3 * rng.normal(size=300) > 0).astype(int)
    params = {"n_estimators": 10} if name == "random_forest" else {}
    m = build_model(name, params)
    m.fit(X, y)
    p = positive_proba(m, X)
    assert p.shape == (300,) and (p >= 0).all() and (p <= 1).all()


def test_class_column_alignment():
    X = np.random.default_rng(0).normal(size=(100, 2)); y = (X[:, 0] > 0).astype(int)
    m = LogisticRegression().fit(X, 1 - y)          # labels flipped -> classes_ still [0, 1]
    m.classes_ = np.array([1, 0])                   # simulate reversed column order
    p_internal = m.predict_proba(X)
    assert np.allclose(positive_proba(m, X), p_internal[:, 0])


def test_probability_shape_mismatch():
    class Bad:
        classes_ = np.array([0, 1])
        def predict_proba(self, X):
            return np.zeros((len(X), 3))
    with pytest.raises(ValueError, match="shape"):
        positive_proba(Bad(), np.zeros((4, 2)))
