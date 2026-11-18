# Final Acceptance Test

Machine: Windows 11 laptop, Python 3.12.6, Node 22.19, single host (Laptop 1 role; the UI was also run on this host).
Date: 2026-11-18. Commands were run from the repository root unless stated. Outcome legend: **PASS**, **FAIL**,
**BLOCKED** (missing external input), **TEAM** (needs the team / two physical laptops).

## Summary

| # | Acceptance requirement | Outcome |
|---|---|---|
| 1 | Data-leakage audit (splitting, preprocessing, feature selection, HPO, evaluation) | PASS |
| 2 | Route-level evaluation with documented metric definitions | PASS |
| 3 | Clean-clone reproducibility test | PASS (see §3 for scope) |
| 4 | Controlled failure recovery | PASS |
| 5 | Genuine two-laptop LAN demonstration | **TEAM**: single-machine LAN-interface test PASS; two-laptop run not possible from one machine |
| 6 | Report metrics match experiment outputs; screenshots are of the real application | PASS |
| 7 | Final requirements audit against all five documents | PASS (`FINAL_DELIVERABLE_AUDIT.md`, `REQUIREMENTS_TRACEABILITY.md`) |
| 8 | This document | PASS |
| — | System works for the ten **evaluator** trips | **BLOCKED**: evaluator trips not supplied; all 10 student-test trips pass through the API |

## 1. Data-leakage audit

Full table: `reports/final/DATA_LEAKAGE_AUDIT.md`. Executed checks:

```powershell
grep -rn "evaluator_test" --include=*.py src scripts backend     # only config docs, EDA inventory, API registry
python -m pytest tests/unit/test_candidates_features.py::test_labels_do_not_change_candidates   # PASS
python -m pytest tests/unit/test_data.py::test_inference_load_never_reads_route                  # PASS
python -m pytest backend/tests/test_api.py::test_no_ground_truth_leakage                          # PASS
```

Fold membership (`experiments/configurations/cv_folds.json`) shows disjoint train/validation trip lists in all 5 folds.
The HMM grid used out-of-fold probabilities of training trips only (`artifacts/metadata/hmm_params.json` → `tuned_on`).

## 2. Route-level evaluation

Implemented in `src/pathsnap/evaluation/route_metrics.py`; definitions in `docs/methodology.md` §9 (ordering via edit
similarity and LCS, missing/extra segments, invalid transitions, continuity, length-weighted route P/R/F1). Unit tests
compare Levenshtein/LCS with reference O(n·m) implementations on 50 random cases (`tests/unit/test_metrics.py`, PASS).
Student-test results (`experiments/results/decoder_summary.json`, final classifier + tuned HMM):

| Metric | Classifier only | HMM + reconstruction |
|---|---|---|
| Top-1 accuracy | 0.9000 | 0.9544 |
| Route continuity | 0.7820 | 0.9967 |
| Invalid transitions (10 trips) | 1108 | 12 |
| Route F1 | 0.9491 | 0.9889 |
| Edit similarity | 0.6654 | 0.9420 |

## 3. Clean-clone reproducibility

Executed in an empty scratch folder (not the working copy):

```powershell
git clone -b develop https://github.com/Shriya-sr/ML-Hackathon.git cleanclone
cd cleanclone
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pip install -e .
$env:PATHSNAP_DATA_DIR = "D:\ML hackathon"           # original course data folder
.\.venv\Scripts\python.exe -m pytest -p no:warnings
.\.venv\Scripts\python.exe scripts/build_features.py --jobs 6
.\.venv\Scripts\python.exe -W ignore scripts/train_all.py --models logistic_regression hist_gradient_boosting
cd frontend; npm ci; npx tsc -b; npx vitest run; npx vite build
```

Results:

| Step | Result |
|---|---|
| install from requirements | PASS |
| pytest (commit 88fe2b9) | 80 passed, 12 skipped. The skipped API/integration tests hard-coded `data/raw`; fixed to honour `PATHSNAP_DATA_DIR` and re-run after the final push (see below) |
| feature rebuild | identical to the original run: train 1 246 339 rows, 184 089 positives, miss rate 0.0114; student-test 131 544 rows, miss rate 0.0015 |
| retraining (LR, HGB) | **identical** to the recorded results: LR CV Macro-F1 0.9262 ± 0.0159, CV Top-1 0.8820, student Top-1 0.8834, log loss 0.1019; HGB CV Macro-F1 0.9332 ± 0.0138, CV Top-1 0.8840, student Top-1 0.8950, log loss 0.0828 (only wall-clock timings differ) |
| frontend `npm ci`, `tsc -b`, `vitest`, `vite build` | PASS (12 tests) |

Scope note: the full 8-model training (~45 min) and the Optuna search (~25 min) were not repeated in the clean clone. Two
models were retrained to confirm deterministic reproduction; all commands are in `scripts/run_all.py`.

## 4. Controlled failure recovery

| Failure | Behaviour | Evidence |
|---|---|---|
| malformed / missing / empty files | `DataFormatError` naming the file → API 422 | `tests/unit/test_data.py` (11 cases), `backend/tests/test_api.py::test_malformed_trip_files_return_422` |
| invalid coordinates, duplicates, unsorted timestamps | cleaned, counts reported in the response | `test_out_of_order_duplicate_and_invalid_rows` |
| missing / corrupt model artifact, schema mismatch | `ArtifactError` / `SchemaError`; API starts degraded, predict → 503 | `tests/unit/test_models.py`, `test_missing_model_returns_503` |
| invalid predictions (probabilities outside [0,1] / NaN) | pipeline raises `ValueError` → 422 | `inference/pipeline.py` check; probability-shape test |
| disconnected road network | Viterbi restarts chain, no fabricated link | `test_disconnected_network_components`, `test_disconnected_transition_breaks_chain` |
| fixes far from any road | no prediction, empty route | `test_all_fixes_far_from_network`, `test_partial_dropout_far_fixes` |
| API timeout | 504 | `test_prediction_timeout_returns_504` |
| invalid trip id / path traversal | 404 | `test_invalid_trip_ids` |
| frontend cannot reach API | "API offline" + Retry, 10 s heartbeat reconnect | `frontend/src/test/app.test.tsx::reports an unreachable API` |

## 5. LAN demonstration

Executed on this machine with the API bound to `0.0.0.0:8000`, requests sent to the machine's own Wi-Fi IPv4 address
(redacted as `<wifi-ip>`), so traffic used the LAN interface and the binding:

```powershell
python -m backend.main          # console captured to experiments/logs/api_demo_console.log
python scripts/lan_check.py http://<wifi-ip>:8000 --all
```

Output (`experiments/logs/lan_check_wifi_interface.log`): health OK, 10 trips listed, `POST /api/predict/<id>` → 200 for
all 10 student-test trips (0.20–2.21 s each), invalid id → 404, **ALL CHECKS PASSED**. UI screenshots were captured from
the running Vite app calling this API (`reports/figures/ui_trip_*.png`).

**TEAM action (cannot be done from one machine):** run `docs/deployment.md` steps 1–7 with two laptops. On Laptop 2
run `python scripts/lan_check.py http://<laptop-1-ip>:8000 --all` (stdlib only), open the UI, and take the screenshots of
the UI and of Laptop 1's console for the report (§9 of the PDF notes this).

## 6. Report ↔ outputs consistency

* Every table and figure in the PDF is produced by `scripts/build_report.py` / `make_report_figures.py` from
  `experiments/` CSV/JSON files. No number is typed by hand.
* The API produces the same metrics as the offline evaluation: `tests/integration/test_student_trips.py` asserts equality
  of classifier Top-1, HMM Top-1 and route continuity for three real trips (PASS).
* Screenshots: Folium maps rendered from `maps/html/*.html`; UI screenshots of the live UI + API (headless Chrome,
  `scripts/capture_screenshots.py`); console image rendered from the captured real log (`scripts/render_console_log.py`).

## 7. Test totals (working copy, final code)

```powershell
python -m pytest -p no:warnings          # 92 passed
cd frontend; npx vitest run; npx tsc -b   # 12 passed, type-check clean
```

## Remaining limitations

1. Evaluator trips not supplied: final-evaluation metrics cannot be reported (BLOCKED).
2. The two-laptop demonstration and its screenshots must be done by the team (TEAM).
3. Arcs are straight segments; candidate misses (1.1% train) cap accuracy; labels come from aligning to an imperfect reference.
4. The best HMM grid values for w_clf and the U-turn penalty lie on the grid edge; a wider grid may help slightly.
5. Repository visibility (public) and real GitHub usernames in `TEAM_ROSTER.md` are for the team to set.
