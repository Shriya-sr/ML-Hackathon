# Team Member Explanations

Ownership follows the proposed split in `TEAM_CONTRIBUTIONS.md`. It is a responsibility plan for review and
defence, not an authorship claim (the implementation was AI-assisted, as disclosed in the README). Every member
should also read `COMPLETE_VIVA_GUIDE.md` and `QUICK_REVISION.md`.

---

## Shriya: data, preprocessing, candidates, features

**Files:** `src/pathsnap/data/{loader,validator,preprocessing,labeling,dataset}.py`, `src/pathsnap/geography/*`,
`src/pathsnap/features/*`, `scripts/run_eda.py`, `scripts/build_features.py`, `scripts/feature_analysis.py`, `reports/eda/`.

**What it does.** Reads the four raw file types robustly, checks data quality, cleans the track, projects each trip
to UTM, derives per-fix labels, generates candidate arcs and computes 21 features.

**How it works.**
* `loader.read_*`: pandas whitespace parser, column-count and integer checks, raises `DataFormatError(file, reason)`.
* `validator.validate_trip`: duplicates, monotonic timestamps, gaps, isolated nodes, weak/strong components (scipy),
  two-way share, route breaks.
* `labeling.align_to_route`: DP `C[i,j] = cost(i,j) + min_{j'≤j} C[i-1,j']` with prefix minima, then back-pointers.
* `SegmentIndex` (STRtree) → `generate_candidates` (radius, K, fallback) → `extract_features`.

**Design decisions.** Per-trip UTM (data is world-wide); labels only via alignment (`.route` ≠ per fix);
heading term for out-and-back trips; −1 labels beyond 60 m (parked cars); R = 50 m / K = 10 from the sweep.

**Likely questions.**
* *Why is the miss rate higher on train than student-test?* Some training trips have parked periods and detours
  where the reference arc is far from the fixes. Trip-level rates are in `candidate_coverage_train.csv`.
* *Show that labels don't leak into candidates.* `tests/unit/test_candidates_features.py::test_labels_do_not_change_candidates`.
* *What does `dist_ratio` capture?* How much farther a candidate is than the nearest one, independent of absolute noise.

## Sudheeksh: models, cross-validation, Bayesian optimisation

**Files:** `src/pathsnap/models/model_factory.py`, `src/pathsnap/training/*`, `src/pathsnap/evaluation/metrics.py`,
`src/pathsnap/optimisation/*`, `scripts/train_all.py`, `scripts/optimise.py`, `configs/models.yaml`, `configs/optimisation.yaml`.

**What it does.** Trains eight classifiers with identical features and trip-grouped folds, records every metric,
selects the final model and tunes it with Optuna.

**How it works.**
* `trip_folds`: `GroupKFold(5, shuffle=True, random_state=42)` over trip ids, saved to `experiments/configurations/cv_folds.json`.
* `run_cv`: fit on training-fold rows (every 2nd fix), evaluate on all labelled fixes of the validation trips,
  and also score the training rows (generalisation gap). Out-of-fold probabilities are saved for HMM tuning.
* `evaluate_model`: candidate metrics at threshold 0.5 + Top-1 via `select_top1`.
* `run_study`: TPE sampler (10 random start-up trials), MedianPruner, objective = mean CV Macro-F1.

**Design decisions.** Group = trip; Macro-F1 as the primary metric plus Top-1/log loss/cost; subsampling only
for KNN/SVM; HGB chosen over the statistically tied RF for log loss, speed and size; tuned vs untuned chosen by CV.

**Likely questions.**
* *Why is the student-test score sometimes above CV?* Ten trips are a small sample. It lies within fold variation.
* *What does the pruner do?* It stops a trial after a fold whose running mean is below the median of earlier trials.
* *How do you get SVM probabilities?* Platt scaling (`CalibratedClassifierCV(method="sigmoid", cv=3, ensemble=False)`).

## Shubashitha: HMM/Viterbi, maps, UI

**Files:** `src/pathsnap/decoding/*`, `scripts/tune_hmm.py`, `scripts/evaluate_all.py`, `src/pathsnap/visualisation/*`,
`scripts/generate_maps.py`, `frontend/src/*`.

**What it does.** Turns per-fix probabilities into a coherent route, compares it with classifier-only output,
draws Folium maps, and builds the evaluator UI.

**How it works.**
* `HMMDecoder.decode`: top-8 states per fix → `log_emission` → `transition_matrix` (bounded Dijkstra via
  `RoadNetwork.network_distance`) → `viterbi_segmented` (chain breaks: no candidates / no valid transition / gap > 120 s).
* `reconstruct_route`: shortest-path infill inside a chain (`RoadNetwork.connecting_arcs`).
* `tune_hmm.py`: grid over w_clf, β, w_dist, U-turn penalty on 30 training trips with OOF probabilities.
* UI: `App.tsx` (state, heartbeat, deep link `?trip=`), `MapView.tsx` (layers, snap animation, replay), `MetricsPanel.tsx`, `ExplainPanel.tsx`.

**Likely questions.**
* *Why bounded Dijkstra?* Consecutive fixes are seconds apart, so only nearby nodes matter. Bounding by distance and
  speed keeps it fast. Results are cached per source node.
* *What does the U-turn penalty fix?* Flip-flopping between a road's two directions when heading is unreliable.
* *Why do reverse-direction errors remain?* When the car is stationary the heading is unknown and both directions
  explain the fixes equally well. See the error breakdown in QUICK_REVISION.

## Sujal: inference pipeline, backend, deployment, testing

**Files:** `src/pathsnap/inference/pipeline.py`, `backend/*`, `docs/deployment.md`, `Dockerfile`, `docker-compose.yml`,
`tests/`, `backend/tests/`.

**What it does.** Packages the trained artifacts into an API that runs the full pipeline per request, scores it
server-side and serves it to the UI over the LAN.

**How it works.**
* `MapMatcher`: loads `final_classifier.joblib`, checks feature schema/version and feature count, loads HMM params,
  `predict()` reads the trip **without** `.route`.
* `RouteEvaluator`: the only reader of `.route`; returns aggregates only.
* `backend/main.py`: CORS, gzip, access-log middleware. `routers/trips.py`: id whitelist (404), timeout (504),
  malformed data (422), missing model (503).
* `TripRegistry`: EVALUATION vs DEVELOPMENT mode by checking `data/raw/evaluator_test`.

**Likely questions.**
* *How do you show the evaluator it's live?* The console log line per request, with client IP and duration.
* *What if Laptop 2 cannot connect?* Check the IP (`ipconfig`), the firewall rule, `Test-NetConnection`, same network, `/health`.
* *How is leakage prevented in the API?* Schemas have no reference fields, and `test_no_ground_truth_leakage` asserts it.
