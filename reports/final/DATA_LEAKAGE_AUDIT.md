# Data-Leakage Audit

Method: code inspection (`grep` over `src/`, `scripts/`, `backend/`), automated tests and the experiment
configuration files. Every row states the evidence.

| Stage | Possible leak | Finding | Evidence |
|---|---|---|---|
| Partitioning | the same trip in train and test | **None.** Partitions are separate folders; no script moves trips between them. | `data/raw/{train,student_test}`; `scripts/build_features.py --partitions train student_test` processes each folder separately |
| Cross-validation | fixes of one trip in both training and validation folds | **None.** `GroupKFold(groups=trip_id)`; fold membership saved. | `src/pathsnap/training/cross_validation.py::trip_folds`; `experiments/configurations/cv_folds.json` (train/val trip lists are disjoint) |
| Labels → candidates | the reference arc inserted into or shaping the candidate set | **None.** Candidates are generated before labels are attached, from GPS + network only. | `features/candidate_generation.py`; test `test_labels_do_not_change_candidates` (identical candidates with and without `.route`) |
| Labels → features | a feature computed from the reference route | **None.** All 21 features use the track and the network only. | `features/feature_extraction.py` (no `true_arc`/`label`/`route` inputs); schema `feature_schema.py` excludes key/label columns |
| Learned preprocessing | scaler/statistics fitted on validation/test data | **None.** The only learned transforms are StandardScalers inside sklearn Pipelines, re-fitted on each training fold and on the full training set. | `models/model_factory.py` |
| Feature selection | test data used to choose features | **None.** Permutation importance and ablation use held-out *training* folds only. | `scripts/feature_analysis.py` (reads `train_*.pkl` only) |
| Model selection | choosing a model on student-test scores | **Avoided.** Selection uses CV Macro-F1 (+ CV Top-1, log loss, cost). Student-test is reported for every model, never optimised. | `scripts/optimise.py` selection rule; report §5.2 |
| Hyperparameter optimisation | tuning on student-test or evaluator data | **None.** Optuna objective = CV on the 80 training trips; student-test scored once after the search. | `optimisation/bayesian_search.py`, `scripts/optimise.py` |
| Decoder tuning | HMM parameters tuned on test data, or on optimistic in-sample probabilities | **None.** Grid on 30 training trips using out-of-fold probabilities. | `scripts/tune_hmm.py` (`oof_*.npy`), `artifacts/metadata/hmm_params.json` (`tuned_on`) |
| Evaluator partition | used for training, selection or tuning | **Never used.** The trips arrived after development was finished (EDA had already run). They share no trip ID or `.track` file with train/student-test (checked by `evaluate_heldout.py`). They are read only by the API registry and by `scripts/evaluate_heldout.py`, which scores the frozen artifacts once; nothing reads its output for training, tuning or selection. | `scripts/evaluate_heldout.py`, `experiments/results/heldout_evaluator_summary.json` (`overlap_check`, `used_for_model_selection: false`) |
| Inference path | `.route` visible to the predictor | **None.** `MapMatcher.predict` calls `load_trip(..., with_route=False)`. | `inference/pipeline.py`; test `test_inference_load_never_reads_route` |
| API / UI | reference route or labels sent to the browser | **None.** Response schema has no reference fields; geometry is only sent for predicted arcs and nearby roads. | `backend/schemas`, `services/prediction_service.py`; test `test_no_ground_truth_leakage` |
| Logs / errors | labels in error messages or logs | **None.** Logs print aggregate accuracy only; errors name the file and the problem, not its content. | `backend/main.py`, `services/prediction_service.py` |

**Residual caveat.** The student-testing partition was used for development-stage comparison of all models, as the brief
specifies. Repeatedly looking at it carries a mild risk of indirect selection pressure. We limited this by fixing
all selection rules to CV before computing student-test scores. The evaluator partition is the only fully untouched estimate: Top-1 0.9080 → 0.9590 with the HMM, in line with
student-test (0.9000 → 0.9544), which suggests the student-test numbers were not inflated by this pressure.
