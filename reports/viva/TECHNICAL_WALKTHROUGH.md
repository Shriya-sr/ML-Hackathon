# Technical Walkthrough: one trip, end to end

Follow a single request (`POST /api/predict/00000052`, 262 fixes) through the code. Open each file as you go.

1. **Request arrives**: `backend/routers/trips.py::predict`. The id is checked by `TripRegistry.trip_dir` (regex +
   membership), then `PredictionService.predict` runs in a worker thread under `asyncio.wait_for`.
2. **Load without labels**: `pathsnap/inference/pipeline.py::MapMatcher.predict` → `load_trip(dir, with_route=False)`.
   `.track` (262×3), `.nodes`, `.arcs` are parsed by `data/loader.py`; `.route` is not opened.
3. **Clean + project**: `data/dataset.py::process_trip` → `clean_track` (invalid rows, sort, duplicates) →
   `LocalProjector.for_points` picks the UTM zone of the trip centroid (EPSG printed in the UI diagnostics) →
   `RoadNetwork` builds projected segments, lengths, bearings, degrees, reverse-arc map and CSR adjacency.
4. **Candidates**: `features/candidate_generation.py` → one STRtree `dwithin` query (50 m) for all fixes → keep the 10
   nearest per fix → fallback nearest ≤ 150 m.
5. **Features**: `features/feature_extraction.py` computes 21 columns; `feature_schema.to_matrix` orders them exactly as
   at training time.
6. **Classifier**: `positive_proba(model, X)` returns P(class 1) with an explicit `classes_` check.
7. **Baseline**: `decoding/hmm.py::classifier_only` gives the argmax per fix.
8. **HMM**: `HMMDecoder.decode` keeps the top-8 states per fix, computes emissions (`emissions.py`) and, per step,
   transitions (`transitions.py`, driving distances via `RoadNetwork.network_distance` → cached bounded Dijkstra),
   then `viterbi.py::viterbi_segmented` returns the best state per fix, chain ids and break reasons.
9. **Reconstruction**: `reconstruct_route` fills shortest paths between consecutive arcs inside each chain.
10. **Server-side scoring**: `RouteEvaluator.evaluate` opens `.route` (only here), aligns it to the fixes
    (`labeling.align_to_route`), computes candidate miss rate, Top-1 (classifier and HMM), selection Macro-F1, log loss,
    route continuity/F1/edit similarity, and returns numbers only.
11. **Response**: `PredictionService._payload` packs GPS fixes, per-fix decisions, geometry of predicted arcs and nearby
    roads, robustness flags, timings and metrics. It caches the result for `/api/results` and `/explain`.
12. **Log line**: the middleware prints `client POST /api/predict/00000052 -> 200 (… ms)`; the service prints the accuracy line.
13. **UI**: `frontend/src/App.tsx::runPrediction` stores the response; `MapView` draws the layers and `MetricsPanel`
    the metrics. Clicking a fix calls `/api/results/{id}/explain/{fix}` and `ExplainPanel` shows the candidate table.

## Offline experiment chain (what produced the numbers)

`run_eda.py` → `build_features.py` → `train_all.py` (8 models × 5 folds + student test) → `feature_analysis.py` →
`optimise.py` (Optuna; exports `final_classifier.joblib`, `classifier.json`) → `tune_hmm.py` (`hmm_params.json`) →
`evaluate_all.py` (classifier vs HMM) → `generate_maps.py` → `make_report_figures.py` → `build_report.py`.
