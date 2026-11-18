# Requirements Traceability Matrix

Sources: **MG** = Main Guidelines, **MT** = Models and Training Guidelines, **DL** = Deliverables (Technical Report &
Submission Guidelines), **AP** = Assessment and Plagiarism Policy, **DE** = Dataset Exploration.
Status: **PASS** (implemented and verified by the stated method) · **PARTIAL** · **BLOCKED** (needs an external input) ·
**TEAM** (an action only the team can perform).

| # | Requirement | Source | Implementation | Verification | Status | Limitation |
|---|---|---|---|---|---|---|
| 1 | Inspect raw data: trips per partition, fixes per trip, file availability/sizes, types | DE §4 | `scripts/run_eda.py`, `data/loader.py`, `data/validator.py` | `reports/eda/dataset_inventory.csv` (90 trips, 360 files, 0 missing) | PASS | evaluator trips arrived later; held out (not part of EDA) |
| 2 | Missing/invalid fixes, timestamp ordering, irregular gaps | DE §4 | `validator.py`, `preprocessing.py` | `timestamp_analysis.csv`, `data_quality_report.md`; `tests/unit/test_data.py` | PASS | |
| 3 | Geographic extent, network size, connectivity, isolated nodes, components | DE §4 | `validator.py` (scipy connected_components) | `network_connectivity.csv`, fig. 04 | PASS | |
| 4 | Reference route structure and limitations | DE §4, §6 | `validator.py`, `labeling.py` | route breaks/continuity in CSV; `docs/methodology.md` §3 | PASS | |
| 5 | Proper geographic distances (no raw-degree Euclidean) | DE §5 | per-trip UTM `geography/projection.py` | `test_projection_matches_haversine`, `test_utm_zones` | PASS | |
| 6 | Do not assume GPS rows ↔ network rows; preserve relationships | DE §3 | monotone alignment, candidate tables keyed (trip, obs, arc) | `test_alignment_recovers_route`, `test_alignment_handles_loop` | PASS | |
| 7 | Independent preprocessing + feature engineering | MG §4, DL §1 | `src/pathsnap/data`, `features/` | unit tests; ablation | PASS | |
| 8 | Candidate generation with miss-rate measurement, no GT insertion | MT §5 | `candidate_generation.py` | `candidate_radius_sweep_train.csv`; `test_labels_do_not_change_candidates` | PASS | |
| 9 | ≥ 5 distinct models from different families, configs recorded | MT §2, §4; AP §3 | `models/model_factory.py`, `configs/models.yaml`, `scripts/train_all.py` | `experiments/results/model_summary.csv` (8 models) | PASS | KNN/SVM on subsamples (documented) |
| 10 | 80 train trips for fitting + CV; 10 student trips for comparison | MT §4 | `train_all.py` | fold file `experiments/configurations/cv_folds.json` | PASS | |
| 11 | Group-aware CV (trip never split) | MT §7, DE §7 | `training/cross_validation.py` (GroupKFold by trip_id) | fold json lists disjoint trips | PASS | |
| 12 | Metrics: accuracy, precision, recall, F1, confusion, log loss, Top-1, miss rate, continuity, sequence-level, train/infer time | MT §5 | `evaluation/metrics.py`, `route_metrics.py` | per-model JSON + summary CSV; `test_metrics.py` | PASS | |
| 13 | Explain primary selection metric | MT §5 | `docs/methodology.md` §6, report §3 | report | PASS | |
| 14 | Comparison table (model, family, config, CV Macro-F1, Top-1, continuity, student Macro-F1, log loss, inference time) | DL §3.1, MT §6 | `build_report.py` | report §3.1 tables generated from CSV | PASS | |
| 15 | Per-model answers A1–2, B5–6, C7–9 | DL §3.2 | `build_report.py` → `docs/model_explanations.md` | generated from records | PASS | |
| 16 | Visualisations: CV Macro-F1, student Top-1/Macro-F1, log loss, confusion, epoch curves only where real | DL §3.3 | `make_report_figures.py` | `reports/figures/model_*.png` | PASS | |
| 17 | Overfitting/underfitting evidence | DL §3.2 C9 | train-fold vs val-fold metrics in `run_cv` | `model_generalisation_gap.png`, per-model C9 text | PASS | |
| 18 | Bayesian optimisation: space, objective, grouped CV, trials, best, refit, student eval, tuned vs untuned, cost | MT §7, DL §4, AP §3 | `optimisation/`, `scripts/optimise.py`, `configs/optimisation.yaml` | `experiments/optimisation/*` (SQLite study, trials.csv, summary json) | PASS | 40 trials, tuned vs untuned in experiments/optimisation |
| 19 | Evaluator partition never used for selection/tuning | MT §7, DE §7, AP §7 | scripts read only train/student_test | code review + `data/README.md` | PASS | |
| 20 | HMM: hidden states, emissions, transitions, Viterbi, missing candidates, gaps, disconnected | MT §8–9, DL §5 | `decoding/` | `tests/unit/test_decoding.py`, edge-case tests | PASS | |
| 21 | Classifier-only vs classifier+HMM, pointwise + sequence + continuity | MT §9, DL §5 | `evaluation/decoder_eval.py`, `scripts/evaluate_all.py` | `decoder_comparison_*.csv`, `decoder_summary.json` | PASS | |
| 22 | Decoder params tuned on train/student data only | MT §9 | `scripts/tune_hmm.py` (train OOF) | `artifacts/metadata/hmm_params.json` | PASS | |
| 23 | Folium maps: GPS, network, reference, prediction; colours, legend, extent, layer control | DL §6, DE §8 | `visualisation/folium_maps.py`, `scripts/generate_maps.py` | `maps/html/*.html`, screenshots | PASS | |
| 24 | Export model + preprocessing + HMM config artifacts | MG §4, MT §5 | `artifacts/models/final_classifier.joblib`, `artifacts/metadata/*.json` | `MapMatcher` loads and validates schema | PASS | |
| 25 | FastAPI backend processing evaluation requests | MT §5–6 | `backend/` | `backend/tests/test_api.py` | PASS | |
| 26 | API computes accuracy from internal reference data, labels stay server-side | MT §6, AP §7 | `inference/pipeline.py::RouteEvaluator` | `test_no_ground_truth_leakage` | PASS | |
| 27 | Works for all ten reserved trips | MT §6 | registry over `evaluator_test/` | `experiments/logs/lan_check_evaluator_trips.log` (10/10 → 200, EVALUATION mode); `scripts/evaluate_heldout.py` | PASS | |
| 28 | API logs visible on Laptop 1 | MT §6 | access-log middleware in `backend/main.py` | console output | PASS | |
| 29 | UI shows 10 trips, sends id, displays route + results | MG §3, MT §6 | `frontend/` | vitest + headless screenshots | PASS | |
| 30 | Two-laptop deployment | MG §3 | configurable host/port/API URL; `docs/deployment.md` | TEAM: must be demonstrated on two machines | TEAM | single-machine verified |
| 31 | No hardcoded predictions / no API bypass | MT §6, AP §7 | pipeline runs per request | code review; `test_repeated_and_concurrent_requests` | PASS | |
| 32 | Public GitHub repo with all code, results, README, organised folders, no secrets | DL §1 | this repo, `.gitignore` | `git ls-files` review | PASS | repo visibility is the owner's setting |
| 33 | Consolidated technical report PDF following the required structure | DL §1–7 | `scripts/build_report.py` | `reports/technical_report/PathSnap_Technical_Report.pdf` | PASS | two-laptop screenshots to be added by team |
| 34 | UI screenshots + API console screenshots from receiving laptop | DL §7 | `scripts/capture_screenshots.py` | single-machine captures | PARTIAL | two-laptop captures: TEAM |
| 35 | Disclose AI assistance; team can explain everything | DL §8, AP §5 | README disclosure, `TEAM.md` | viva material | PASS / TEAM | understanding is the team's responsibility |
