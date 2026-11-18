# Review 1: Intermediate Checkpoint

**Project:** PathSnap, map-matching & route reconstruction · UE24CS352A · Team: Sujal, Shriya, Sudheeksh, Shubashitha
**Branch:** `develop` · **Commit:** recorded in the Git history as `docs: add review one checkpoint`

All numbers below come from files in this repository (`experiments/results/`, `reports/eda/`) and were produced by
the scripts named next to them. Nothing is estimated or typed in by hand.

## A. Work completed

| Component | State | Where |
|---|---|---|
| Dataset audit (90 trips, 360 files, 0 missing/invalid values) | done | `scripts/run_eda.py`, `reports/eda/` |
| EDA figures: trip sizes, sampling intervals, gaps, locations, network size/connectivity, data quality, trajectories, fix-to-reference distance, candidate sweep | done | `reports/eda/figures/01–08` |
| Robust loaders + validator + cleaning (`orig_index` kept) | done, unit-tested | `src/pathsnap/data/` |
| Per-trip UTM projection, geometry, STRtree index, directed road graph with bounded Dijkstra | done, unit-tested | `src/pathsnap/geography/` |
| Per-fix labels by monotone DP alignment (with heading term) | done, unit-tested | `src/pathsnap/data/labeling.py` |
| Candidate generation (R = 50 m, K = 10, fallback 150 m) + miss-rate measurement | done | `src/pathsnap/features/candidate_generation.py` |
| 21 features in 5 groups, single schema for train and inference | done | `src/pathsnap/features/` |
| Trip-grouped 5-fold CV framework + experiment tracking (JSON/CSV, git commit, seed) | done | `src/pathsnap/training/` |
| Model experiments | 6 of 8 complete at the time of writing (SVM and MLP running) | `experiments/results/` |
| HMM + segmented Viterbi + route reconstruction (preliminary, default parameters) | done, unit-tested | `src/pathsnap/decoding/` |
| FastAPI backend (all endpoints, server-side evaluator, leakage guard) | implemented; API tests written | `backend/` |
| React + TypeScript + Leaflet UI | implemented, type-checks, 12 UI tests pass | `frontend/` |
| Unit tests | 64 passing | `tests/unit/` |

## B. Actual experimental results

**Data** (`reports/eda/data_quality_report.md`): 80 training trips (208 417 fixes), 10 student-test trips (19 160 fixes);
fixes per trip 262–19 313; networks 299–155 003 nodes. No invalid coordinates, duplicates or unordered timestamps;
135 gaps > 30 s in 30 trips (max 153 064 s); 14 route breaks in 11 trips. Routes have ~5.3 fixes per route arc, so labels are
derived by alignment. Candidate miss rate: **1.14% (train)**, **0.15% (student-test)**.

**Models**: 5-fold trip-grouped CV on 80 training trips; student-test = the 10 held-out trips; classifier-only (argmax):

| Model | CV Macro-F1 | CV Top-1 | Student Top-1 | Student Macro-F1 | Student log loss | Route continuity | Fit (s) |
|---|---|---|---|---|---|---|---|
| Random Forest | 0.9336 ± 0.0144 | 0.8840 | 0.8942 | 0.9344 | 0.0784 | 0.7713 | 66.1 |
| HistGradientBoosting | 0.9332 ± 0.0138 | 0.8840 | **0.8950** | 0.9347 | 0.0828 | 0.7701 | 3.2 |
| Extra Trees | 0.9316 ± 0.0148 | 0.8802 | 0.8872 | 0.9314 | 0.0836 | 0.7502 | 25.7 |
| Logistic Regression | 0.9262 ± 0.0159 | 0.8820 | 0.8834 | 0.9256 | 0.1019 | 0.7562 | 2.0 |
| KNN (k=25, 60k rows) | 0.9223 ± 0.0167 | 0.8711 | 0.8809 | 0.9241 | 0.1344 | 0.7371 | 0.0 |
| Gaussian NB | 0.9013 ± 0.0118 | 0.8596 | 0.8593 | 0.8966 | 0.4316 | 0.7176 | 0.2 |

*Unfinished at this checkpoint:* RBF-SVM (Platt) and MLP (running). Their rows are added to
`experiments/results/model_summary.csv` when finished. Bayesian optimisation and HMM tuning have not been run yet.

**Preliminary HMM** (untuned HGB + default decoder parameters, `scripts/evaluate_all.py --preliminary`,
`experiments/results/decoder_summary_preliminary.json`):

| Student-test (10 trips) | Classifier only | + HMM | + HMM + reconstruction |
|---|---|---|---|
| Top-1 accuracy | 0.8950 | **0.9375** | |
| Selection Macro-F1 | 0.9488 | 0.9710 | |
| Route continuity | 0.7701 | 0.9049 | **0.9974** |
| Route F1 | 0.9424 | | 0.9894 |
| Edit similarity | 0.6406 | | 0.9348 |

The HMM corrected 1 165 fixes and broke 352. Classifier errors: 748 wrong direction, 853 junction neighbour, 408 other road.
After the HMM: 624 / 483 / 89. Sequence decoding removes most "other road" and junction errors; direction errors
remain the hardest (stationary periods without a usable heading).

## C. Architecture

See `reports/review_1/ARCHITECTURE.md`.

## D. Team contributions

See `reports/review_1/TEAM_CONTRIBUTIONS.md`.

## E. Viva preparation

See `reports/review_1/VIVA_PREPARATION.md`.

## F. Remaining work (Phase 2)

1. Finish SVM and MLP experiments → final comparison and model selection.
2. Bayesian optimisation of HistGradientBoosting (Optuna TPE, 40 trials) and tuned vs untuned comparison.
3. HMM parameter grid on training trips using out-of-fold probabilities; final classifier-only vs HMM comparison.
4. Export final artifacts; run API tests and edge-case tests on the real artifacts.
5. Folium comparison maps and screenshots; UI screenshots from the running system.
6. Technical report PDF; final viva material; requirements audit; acceptance tests; clean-clone test.

## G. Risks

| Risk | Mitigation |
|---|---|
| Evaluator trips not supplied yet | API auto-switches from DEVELOPMENT to EVALUATION mode when `data/raw/evaluator_test` is filled; tested with a temporary evaluator folder |
| Two-laptop network issues on demo day (firewall, isolated Wi-Fi) | `docs/deployment.md` checklist: `ipconfig`, firewall rule, `Test-NetConnection`, `/health`; hotspot fallback |
| No internet → no basemap tiles | UI still draws road network + routes on a dark background |
| Large trips are slow | worst-case student trip takes a few seconds; timeout and loading states in the UI |
| Labels derived by alignment are approximate | fixes > 60 m from their aligned arc excluded; limitation documented |

## H. Demonstration readiness at this checkpoint

* Can demonstrate: EDA outputs, feature pipeline, model comparison for 6 models, preliminary HMM gains, unit tests.
* API and UI code complete; final artifacts are exported in Phase 2 after optimisation (the API needs them to start in `ok` state).
