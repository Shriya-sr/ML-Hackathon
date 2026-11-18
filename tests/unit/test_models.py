import json

import joblib
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from pathsnap.features.feature_schema import FEATURE_COLUMNS, FEATURE_VERSION, SchemaError
from pathsnap.inference.pipeline import ArtifactError, MapMatcher
from pathsnap.models.model_factory import MODEL_CARDS, build_model, positive_proba


@pytest.mark.parametrize("name", list(MODEL_CARDS))
def test_every_model_builds_and_predicts_probabilities(name):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, len(FEATURE_COLUMNS)))
    y = (X[:, 0] + 0.3 * rng.normal(size=300) > 0).astype(int)
    params = {"n_estimators": 10} if name in ("random_forest", "extra_trees") else {}
    if name == "mlp":
        params = {"hidden_layer_sizes": [8], "max_iter": 50}
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


def _make_artifacts(tmp_path, model=None, meta_cols=None):
    (tmp_path / "models").mkdir(parents=True)
    (tmp_path / "metadata").mkdir()
    if model is not None:
        joblib.dump(model, tmp_path / "models" / "final_classifier.joblib")
    (tmp_path / "metadata" / "classifier.json").write_text(json.dumps(
        {"feature_columns": meta_cols or FEATURE_COLUMNS, "feature_version": FEATURE_VERSION}))
    (tmp_path / "metadata" / "hmm_params.json").write_text(json.dumps({"params": {}}))
    return tmp_path


def test_missing_model_artifact(tmp_path):
    with pytest.raises(ArtifactError, match="not found"):
        MapMatcher(_make_artifacts(tmp_path))


def test_corrupt_model_artifact(tmp_path):
    root = _make_artifacts(tmp_path)
    (root / "models" / "final_classifier.joblib").write_bytes(b"not a pickle")
    with pytest.raises(ArtifactError, match="could not be loaded"):
        MapMatcher(root)


def test_schema_mismatch_rejected(tmp_path):
    X = np.zeros((10, len(FEATURE_COLUMNS))); X[:5, 0] = 1
    m = LogisticRegression().fit(X, np.r_[np.ones(5), np.zeros(5)])
    with pytest.raises(SchemaError):
        MapMatcher(_make_artifacts(tmp_path, m, meta_cols=FEATURE_COLUMNS[:-1]))


def test_wrong_feature_count_rejected(tmp_path):
    X = np.zeros((10, 3)); X[:5, 0] = 1
    m = LogisticRegression().fit(X, np.r_[np.ones(5), np.zeros(5)])
    with pytest.raises(SchemaError, match="expects 3"):
        MapMatcher(_make_artifacts(tmp_path, m))
