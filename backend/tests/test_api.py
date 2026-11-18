"""API tests. Most run the REAL pipeline on the smallest student-testing trip
(00000052, 262 fixes) with the exported artifacts - nothing is mocked except
where a failure mode is being simulated."""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from backend import dependencies  # noqa: E402
from backend.main import create_app  # noqa: E402
from backend.settings import get_settings  # noqa: E402
from pathsnap.config import data_dir  # noqa: E402
from pathsnap.data.loader import load_trip  # noqa: E402

DATA = data_dir() / "student_test"   # honours PATHSNAP_DATA_DIR
HAVE_DATA = (DATA / "00000052" / "00000052.track").exists()
HAVE_MODEL = (ROOT / "artifacts" / "models" / "final_classifier.joblib").exists()
needs_real = pytest.mark.skipif(not (HAVE_DATA and HAVE_MODEL), reason="dataset or model artifacts not present")
SMALL = "00000052"


@pytest.fixture(scope="module", autouse=True)
def _serve_student_test():
    """Pin the API to the student-testing trips. Without this, the server switches to EVALUATION mode as soon
    as real evaluator trips are in place, and the expectations below (trip 00000052, 262 fixes) no longer apply."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("PATHSNAP_EVAL_PARTITION", "student_test")
        yield


@pytest.fixture(scope="module")
def client():
    app = create_app()
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in ("ok", "degraded") and "mode" in body


@needs_real
def test_system_and_model_info(client):
    s = client.get("/api/system/info").json()
    assert s["mode"] in ("EVALUATION", "DEVELOPMENT") and s["n_trips"] >= 1
    m = client.get("/api/model/info").json()
    assert m["n_features"] == len(m["feature_columns"]) and m["hmm_params"]


@needs_real
def test_list_trips_and_metadata(client):
    trips = client.get("/api/trips").json()
    assert len(trips) == 10
    md = client.get(f"/api/trips/{SMALL}/metadata").json()
    assert md["n_fixes"] == 262 and md["n_arcs"] > 0


@pytest.mark.parametrize("bad", ["99999999", "..%2F..%2Fetc", "abc$", "x" * 80])
def test_invalid_trip_ids(client, bad):
    assert client.post(f"/api/predict/{bad}").status_code == 404
    assert client.get(f"/api/trips/{bad}/metadata").status_code == 404


def test_wrong_method(client):
    assert client.get(f"/api/predict/{SMALL}").status_code == 405


@needs_real
def test_results_and_explain_before_predict(client):
    dependencies.get_state().service._cache.clear()
    assert client.get(f"/api/results/{SMALL}").status_code == 404
    assert client.get(f"/api/results/{SMALL}/explain/0").status_code == 404


@needs_real
def test_full_prediction_real_pipeline(client):
    r = client.post(f"/api/predict/{SMALL}")
    assert r.status_code == 200, r.text
    p = r.json()
    assert p["n_fixes"] == 262 and len(p["fixes"]["hmm_arc"]) == 262
    assert p["metrics"]["available"] is True
    h = p["metrics"]["hmm"]
    assert 0 <= h["top1_accuracy"] <= 1 and 0 <= h["route_continuity"] <= 1
    assert p["timings"]["total_s"] > 0
    # every arc id in the route has geometry
    assert all(str(a) in p["arcs"] for a in p["route"]["hmm"])
    # cached result + explanation
    assert client.get(f"/api/results/{SMALL}").json()["trip_id"] == SMALL
    e = client.get(f"/api/results/{SMALL}/explain/10").json()
    assert e["fix_index"] == 10 and len(e["candidates"]) >= 1
    assert client.get(f"/api/results/{SMALL}/explain/99999").status_code == 404


@needs_real
def test_no_ground_truth_leakage(client):
    """The response must not contain the reference route or per-fix labels."""
    p = client.post(f"/api/predict/{SMALL}").json()
    text = json.dumps(p)
    for key in ("true_arc", "\"label\"", "reference_route", "route_pos", "label_dist"):
        assert key not in text
    ref = load_trip(DATA / SMALL).route.tolist()
    # outside the model's own predicted routes, the reference sequence must not appear anywhere
    rest = {k: v for k, v in p.items() if k not in ("route", "fixes", "arcs")}
    assert json.dumps(ref) not in json.dumps(rest)
    # reference arcs that the model did NOT predict must not be leaked via the geometry dict
    predicted = set(p["route"]["hmm"]) | set(p["route"]["classifier"])
    assert {int(a) for a in p["arcs"]} <= predicted
    for k in ("classifier", "hmm"):
        assert all(not isinstance(v, list) for v in p["metrics"][k].values())


@needs_real
def test_repeated_and_concurrent_requests(client):
    a = client.post(f"/api/predict/{SMALL}").json()
    b = client.post(f"/api/predict/{SMALL}").json()
    assert a["fixes"]["hmm_arc"] == b["fixes"]["hmm_arc"]          # deterministic
    with ThreadPoolExecutor(max_workers=3) as ex:
        codes = list(ex.map(lambda _: client.post(f"/api/predict/{SMALL}").status_code, range(3)))
    assert codes == [200, 200, 200]


@needs_real
def test_prediction_timeout_returns_504(client, monkeypatch):
    st = dependencies.get_state()
    monkeypatch.setattr(st.settings, "prediction_timeout_s", 0.001)
    orig = st.service.predict
    monkeypatch.setattr(st.service, "predict", lambda tid: (time.sleep(0.2), orig(tid))[1])
    assert client.post(f"/api/predict/{SMALL}").status_code == 504


@needs_real
def test_missing_model_returns_503(client, monkeypatch):
    st = dependencies.get_state()
    monkeypatch.setattr(st.service, "matcher", None)
    monkeypatch.setattr(st, "model_error", "ArtifactError: simulated")
    assert client.post(f"/api/predict/{SMALL}").status_code == 503
    assert client.get("/health").json()["status"] == "degraded"
    assert client.get("/api/model/info").status_code == 503


def test_settings_env_override(monkeypatch):
    monkeypatch.setenv("PATHSNAP_PORT", "9123")
    monkeypatch.setenv("PATHSNAP_CORS_ORIGINS", "http://a:5173, http://b:5173")
    s = get_settings()
    assert s.port == 9123 and s.cors_origins == ["http://a:5173", "http://b:5173"]


def test_malformed_trip_files_return_422(tmp_path, monkeypatch):
    """A trip whose .arcs file is broken must give a clear 422, not a crash."""
    import shutil
    if not HAVE_DATA or not HAVE_MODEL:
        pytest.skip("needs data + model")
    root = tmp_path / "data"
    shutil.copytree(DATA / SMALL, root / "evaluator_test" / SMALL)
    (root / "evaluator_test" / SMALL / f"{SMALL}.arcs").write_text("0 1\n1 2.5\n")
    with monkeypatch.context() as mp:
        mp.setenv("PATHSNAP_DATA_DIR", str(root))
        mp.setenv("PATHSNAP_EVAL_PARTITION", "evaluator_test")
        app = create_app()
        with TestClient(app) as c:
            assert c.get("/api/system/info").json()["mode"] == "EVALUATION"
            assert c.post(f"/api/predict/{SMALL}").status_code == 422
    dependencies.init_state()   # restore the pinned default state for other tests
