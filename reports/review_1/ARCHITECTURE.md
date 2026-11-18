# Architecture at Review 1

The full diagram and code map are in [`docs/architecture.md`](../../docs/architecture.md).

**How the components communicate**

1. Offline: `scripts/build_features.py` uses `data/` + `geography/` + `features/` to write candidate tables →
   `scripts/train_all.py` uses `training/` + `models/` + `evaluation/` to write experiment records and OOF
   probabilities → (Phase 2) `optimise.py` exports the final classifier, and `tune_hmm.py` exports the HMM parameters.
2. Online: `backend/` loads `artifacts/` through `inference.pipeline.MapMatcher`. Each `POST /api/predict/{id}` runs
   the same `process_trip` code path as training (without labels), then `decoding/`, then `RouteEvaluator` (server-side
   labels) → JSON → `frontend/` over HTTP.
3. One schema (`features/feature_schema.py`) and one config set (`configs/`) are shared by both paths, so training and
   serving cannot drift apart. The artifact metadata stores the schema, and the API refuses a mismatch.
