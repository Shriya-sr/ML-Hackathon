# Completed Work at Review 1

* **Repository architecture**: `src/pathsnap` package (data, geography, features, models, training, evaluation,
  optimisation, decoding, inference, visualisation), `scripts/`, `configs/`, `backend/`, `frontend/`, `tests/`, `docs/`.
* **Data**: loaders with explicit error messages, validator, cleaning with raw-row pointers, inventory + quality report.
* **Geography**: per-trip UTM, vectorised point-segment geometry, STRtree, directed graph with reverse-arc map and cached bounded Dijkstra.
* **Labels**: monotone DP alignment of fixes to the reference route, with a heading term and an unreliable-label threshold.
* **Candidates & features**: radius/K sweep, chosen 50 m/10, miss rate measured; 21 features; versioned schema.
* **Training**: GroupKFold by trip, OOF probabilities, train-fold vs validation-fold metrics, JSON/CSV experiment records.
* **Models**: LR, Gaussian NB, KNN, RF, Extra Trees, HGB finished; SVM and MLP running.
* **Decoding**: emissions, transitions, segmented Viterbi with documented fallbacks, route reconstruction; preliminary evaluation.
* **API/UI**: complete implementation; UI type-checks and passes 12 tests; API tests are ready and run after artifact export.
* **Tests**: 64 unit tests passing (data, geography, candidates/features, decoding vs brute force, metrics vs reference implementations, models/artifacts).
