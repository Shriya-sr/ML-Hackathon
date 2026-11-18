"""Integration: raw student-test trips -> full pipeline -> server-side metrics.
Checks consistency with the offline evaluation (scripts/evaluate_all.py) on several real trips."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pathsnap.config import data_dir
from pathsnap.inference.pipeline import MapMatcher, RouteEvaluator

ROOT = Path(__file__).resolve().parents[2]
DATA = data_dir() / "student_test"   # honours PATHSNAP_DATA_DIR
OFFLINE = ROOT / "experiments" / "results" / "decoder_comparison_final.csv"
pytestmark = pytest.mark.skipif(
    not ((ROOT / "artifacts" / "models" / "final_classifier.joblib").exists() and DATA.exists()),
    reason="needs artifacts and student_test data")


@pytest.mark.parametrize("tid", ["00000052", "00000091", "00000065"])
def test_api_pipeline_matches_offline_evaluation(tid):
    pred = MapMatcher().predict(DATA / tid)
    m = RouteEvaluator().evaluate(DATA / tid, pred)
    assert m["available"]
    if OFFLINE.exists():
        off = pd.read_csv(OFFLINE, dtype={"trip_id": str}).set_index("trip_id").loc[tid]
        assert abs(m["hmm"]["top1_accuracy"] - off["hmm_top1"]) < 1e-9
        assert abs(m["classifier"]["top1_accuracy"] - off["clf_top1"]) < 1e-9
        assert abs(m["hmm"]["route_continuity"] - off["recon_route_continuity"]) < 1e-9
