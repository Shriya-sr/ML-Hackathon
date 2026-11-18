# Final Deliverable Audit

Statuses: **PASS** (verified, evidence given) · **PARTIAL** · **FAIL** · **BLOCKED BY MISSING INPUT** · **TEAM ACTION**.
Verification date: 2026-11-18, Windows 11, Python 3.12.6, Node 22.19.

## Deliverable 1: GitHub repository

| Requirement | Status | Evidence / verification command |
|---|---|---|
| Data-processing code | PASS | `src/pathsnap/data/`, `scripts/run_eda.py`; `pytest tests/unit/test_data.py` |
| Feature-engineering code | PASS | `src/pathsnap/features/`, `scripts/build_features.py`, `scripts/feature_analysis.py` |
| Training code for every model (8) | PASS | `src/pathsnap/models/model_factory.py`, `scripts/train_all.py`; records `experiments/results/experiment_*.json` |
| Evaluation/comparison code, plots, confusion matrices | PASS | `src/pathsnap/evaluation/`, `scripts/make_report_figures.py`, `reports/figures/model_*.png` |
| Bayesian optimisation code + configs + trial results | PASS | `scripts/optimise.py`, `configs/optimisation.yaml`, `experiments/optimisation/{trials.csv,optuna_study.db,optimisation_summary.json,tuned_vs_untuned.csv}` |
| HMM + Viterbi | PASS | `src/pathsnap/decoding/`; `pytest tests/unit/test_decoding.py` (Viterbi == brute force) |
| FastAPI backend incl. model loading | PASS | `backend/`; `pytest backend/tests` (16 passed) |
| Evaluator UI | PASS | `frontend/`; `npx vitest run` (12 passed), `npx tsc -b` clean; screenshots `reports/figures/ui_*.png` |
| Deployment config + instructions | PASS | `docs/deployment.md`, `configs/api.yaml`, `.env.example`, `Dockerfile`, `docker-compose.yml` |
| Folium map code + HTML | PASS | `src/pathsnap/visualisation/folium_maps.py`, `scripts/generate_maps.py`, `maps/html/*.html` (6 maps) |
| Supporting CSVs, logs, configs | PASS | `experiments/results/*.csv`, `experiments/logs/*.log`, `configs/*.yaml` |
| Saved model + inference assets | PASS | `artifacts/models/final_classifier.joblib` (+ tuned HGB), `artifacts/metadata/{classifier,hmm_params,feature_config}.json` |
| README with install / run / deploy | PASS | `README.md` |
| Organised folders, no secrets | PASS | `.gitignore`; secret scan over tracked files found nothing |
| Repository public / accessible | TEAM ACTION | visibility is a GitHub setting of the owner (Shriya-sr) |

## Deliverable 2: Technical report

| Requirement | Status | Evidence |
|---|---|---|
| One consolidated PDF, required structure (§2.2–7) | PASS | `reports/technical_report/PathSnap_Technical_Report.pdf` (28 pages, headings in the brief's numbering 2.2–8), built by `scripts/build_report.py` from records |
| EDA: preprocessing answer, patterns, features, missing values, trajectory plots, network plots, ≥1 Folium map, partitions named | PASS | report §2 (2.2–2.4); Fig. 1–11 |
| Comparison table in required columns | PASS | report §3.1 (Tables 3.1a–c, incl. precision/recall) |
| Per-model questions A1–2, B5–6, C7–9 | PASS | report §3.2 / `docs/model_explanations.md` |
| Model visualisations (CV Macro-F1, student Top-1/F1, log loss, confusion, real training curves only) | PASS | Fig. 12–19 |
| Bayesian optimisation answers 1–6 + comparison | PASS | report §4 |
| HMM answers 1–6 + classifier vs HMM comparison | PASS | report §5 (5.1 held-out evaluator trips) |
| Folium section with screenshots | PASS | report §6 (whole-trip + zoomed maps) |
| UI screenshots | PASS | `reports/figures/ui_trip_*.png` (real running system, single machine) |
| API console screenshots on the receiving laptop | PARTIAL / TEAM ACTION | `reports/figures/api_console_log.png` is the real console of this machine; the two-laptop capture must be taken on demo day |

## Models & training guidelines

| Requirement | Status | Evidence |
|---|---|---|
| ≥ 5 distinct models, multiple families | PASS | 8 models, 7 families (`model_summary.csv`) |
| 80 trips for fit/CV, 10 for comparison | PASS | `cv_folds.json`, `train_all.py` |
| Group-aware CV | PASS | GroupKFold by trip |
| Metrics incl. Top-1, miss rate, continuity, sequence-level, timing | PASS | summary CSV, decoder summary |
| Primary metric explained, selection justified with evidence | PASS | report §3 (class-imbalance rationale) and §3.4 (paired fold table) |
| Bayesian search, grouped CV, trials recorded, refit, student eval, cost | PASS | `optimisation_summary.json` (40 trials: 18 complete, 22 pruned; 22 min) |
| Evaluator data not used for tuning/selection | PASS | `reports/final/DATA_LEAKAGE_AUDIT.md` |
| HMM requirements + decoder tuned on train/student only | PASS | `tune_hmm.py` (train OOF), `hmm_params.json` |
| System works for all ten reserved trips | PASS | held-out evaluator trips scored once with the frozen pipeline (`scripts/evaluate_heldout.py`): Top-1 0.9080 → 0.9590; all 10 served in EVALUATION mode via the API (`experiments/logs/lan_check_evaluator_trips.log`) |
| API on Laptop 1 + UI on Laptop 2, logs shown | TEAM ACTION | single-machine run verified, including a request through the Wi-Fi interface; procedure in `docs/deployment.md` |

## Academic integrity

| Requirement | Status | Evidence |
|---|---|---|
| No fabricated results; every number reproducible | PASS | all tables and figures are generated by scripts from saved records; `FINAL_ACCEPTANCE_TEST.md` |
| No hardcoded predictions | PASS | `prediction_service.py` runs the pipeline per request; deterministic-repeat and per-trip timing evidence |
| AI assistance disclosed | PASS | README section "AI-assistance disclosure" |
| Team can explain all work | TEAM ACTION | `reports/viva/*` |
