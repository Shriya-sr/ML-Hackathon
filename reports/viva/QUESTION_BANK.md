# Viva Question Bank (120 questions)

Short model answers. Numbers live in `reports/viva/QUICK_REVISION.md` (generated from the experiment records),
so quote them from there. ★ = harder question.

## A. Problem & data (1–20)

1. **What is map matching?** Assigning each noisy GPS fix to the road segment the vehicle was actually on, and producing a drivable route.
2. **Why is raw GPS not enough?** Multipath near buildings, noise of a few to tens of metres, dropouts and gaps put fixes on parallel streets or between roads.
3. **What does a `.track` row contain?** longitude, latitude, timestamp in seconds. No header.
4. **What is a node, and what is an arc?** Node = road-network point (row index = node_id, lon/lat). Arc = directed straight segment between two nodes (row index = arc_id).
5. **What is in `.route`?** The ordered sequence of reference arc ids the vehicle traversed. It is a chain of arcs, not one row per fix.
6. **Does row i of `.track` correspond to row i of `.route`?** No. Routes have ~1 entry per 5 fixes. We derive per-fix labels with a monotone alignment (`data/labeling.py`).
7. **How did you derive per-fix labels?** DP that assigns each fix a route position, non-decreasing in time, minimising capped distance plus a heading penalty. Exact, O(N·M).
8. ★ **Why the heading term in the alignment?** A road's two directions have identical geometry. On out-and-back trips, distance alone cannot tell them apart.
9. **Why mark some fixes −1?** If the aligned arc is > 60 m away (typically a car parked off-network), the reference is unreliable for that fix, so we exclude it from training and scoring.
10. **How many trips per partition?** 80 train, 10 student-test, 10 evaluator (received at the end, held out: Top-1 0.9080 → 0.9590 with the HMM).
11. **What sampling rate?** Nominally 1 Hz. Most steps are 1 s, but there are gaps up to ~42 h (trip 90).
12. **Were there missing or invalid values?** None in the supplied files (see `reports/eda/data_quality_report.md`). The loaders still handle them.
13. **What anomalies did EDA find?** Long gaps in 30 trips, parked periods off-network, 14 route breaks in 11 trips, duplicate arcs, self-loops.
14. **Why is the dataset world-wide relevant?** Degree-based distances would be wrong and inconsistent, so we project each trip into its own UTM zone.
15. **Why UTM per trip and not one projection?** One projection cannot be accurate world-wide. Per-trip UTM keeps scale error < 0.1%.
16. **Why not haversine everywhere?** Possible, but segment projection and bearings are simpler and faster in planar metres. Accuracy is equivalent at trip scale (tested in `test_projection_matches_haversine`).
17. **What are disconnected components?** Groups of nodes with no path between them. Most networks have one large strongly-connected component plus small fragments.
18. **Why do two-way roads matter?** Each direction is a separate arc. The model must pick the correct direction, which is where heading features come in.
19. **Is the reference route perfect ground truth?** No. It has breaks, and GPS sometimes runs where the route does not. We treat it as reference and say so.
20. **Which partition did you use for EDA?** Train and student-test (labelled as such). The evaluator partition arrived after EDA and was not part of it; it was only scored once at the end with the frozen model.

## B. Preprocessing, candidates, features (21–45)

21. **What preprocessing do you do?** Drop invalid coordinates/timestamps, stable re-sort by time, drop duplicate timestamps, project to UTM, flag gaps.
22. **What is `orig_index` for?** It maps every cleaned fix back to its raw row, so outputs refer to the original observation.
23. **Why generate candidates at all?** Scoring every arc (up to 307k) for every fix is infeasible. Candidates restrict the decision to plausible nearby roads.
24. **How are candidates found efficiently?** A Shapely STRtree (R-tree) over all segments, with one vectorised `dwithin` query for all fixes.
25. **How were radius and K chosen?** From the training-partition sweep. 50 m / K=10 sits near the knee of the miss-rate curve.
26. **What is candidate miss rate?** Share of labelled fixes whose reference arc is not a candidate. No downstream model can be right for those fixes.
27. ★ **Could you lower the miss rate by adding the true arc?** That would be leakage. Candidate generation never sees labels (tested in `test_labels_do_not_change_candidates`).
28. **What happens with no candidates?** Fallback: the nearest arc within 150 m. Beyond that, no prediction for that fix and the HMM chain restarts.
29. **List your feature groups.** Geometric, heading, context, kinematic, network (21 features).
30. **Why relative distance features?** Whether an arc is the closest matters more than the absolute metres, which vary with local GPS noise.
31. **What is `cos_heading`?** Cosine of the angle between the motion direction (fixes i±2) and the arc direction. +1 means same direction, −1 means opposite.
32. **Why is heading sometimes invalid?** When the vehicle barely moves (< 3 m over the window), the direction estimate is noise, so it is set to neutral.
33. **What are context features?** Mean/max distance of the neighbouring fixes (±3) to the candidate arc: does the road fit the local trajectory?
34. ★ **Are context features leakage?** No. They use only GPS fixes and geometry, never labels.
35. **Why network degree features?** Junctions and one-way structure change how likely an arc is.
36. **How did you validate the features?** Permutation importance and group ablation on held-out *training* folds.
37. **Which features matter most?** See `experiments/results/permutation_importance.csv` (distance/context/heading features lead).
38. **Do you scale features?** Only for LR, KNN, SVM, MLP, inside a Pipeline re-fitted per training fold. Trees don't need it.
39. **How do you guarantee identical train/inference schemas?** `feature_schema.FEATURE_COLUMNS` + `to_matrix`. The artifact stores the column list and version, and `MapMatcher` refuses a mismatch.
40. **Why train on every second fix?** 1 Hz fixes are highly autocorrelated. Striding halves cost with little information loss. Validation and test use all fixes.
41. **How imbalanced are the classes?** About 1 positive per 6–7 candidate rows.
42. **Did you use class weights?** Not by default (probability calibration for the HMM matters more). `class_weight` was part of the LR search space.
43. **What is the label of a candidate row?** 1 if its arc equals the fix's aligned reference arc, else 0.
44. **What if two fixes have the same candidate?** Rows are per (fix, arc), so the same arc can be a candidate for many fixes.
45. **Why `proj_frac`?** Where along the arc the fix projects. It feeds the HMM's driving-distance computation.

## C. Models & evaluation (46–75)

46. **Which models did you train?** LR, Gaussian NB, KNN, Random Forest, Extra Trees, HistGradientBoosting, RBF-SVM (Platt), MLP: eight models, seven families.
47. **Why GroupKFold?** Random row splits put fixes from the same trip in train and validation, which inflates scores. Grouping by trip measures generalisation to new trips.
48. **What is Macro-F1?** The unweighted mean of the F1 of class 0 and class 1, so the minority positive class counts equally.
49. **Why not plain accuracy?** With ~85% negatives, predicting "no" everywhere already gives ~85% accuracy.
50. **What is log loss?** The mean negative log-probability of the true class. It punishes confident mistakes and measures probability quality.
51. **What is Top-1 accuracy?** For each fix, the highest-scoring candidate equals the reference arc. This is the map-matching objective.
52. ★ **Why can Macro-F1 and Top-1 disagree?** Macro-F1 depends on the 0.5 threshold over all rows. Top-1 only depends on the *ranking* within each fix.
53. **What was your primary selection metric and why?** CV Macro-F1 (as the brief requires), read together with CV Top-1, log loss (HMM emissions need good probabilities), fold stability and cost.
54. **Which model did you select?** HistGradientBoosting (see QUICK_REVISION for numbers).
55. ★ **RF and HGB are within noise. Why HGB?** The difference is far below the fold std. HGB optimises log loss directly, trains ~20× faster (making Bayesian search feasible), has a smaller artifact and faster inference.
56. **How does SVM give probabilities?** Platt scaling: a sigmoid fitted on cross-validated decision values (`CalibratedClassifierCV`, sigmoid, cv=3).
57. **Why subsample for SVM and KNN?** Kernel SVM training is O(n²)–O(n³). KNN prediction cost grows with stored rows. Subsampling is trip-stratified and documented.
58. **How does Gaussian NB fail here?** It assumes independent Gaussian features. Ours are strongly correlated and skewed (e.g. distance ratios), so its probabilities are overconfident (worst log loss).
59. **Why is LR competitive?** The strongest signals (relative distance, heading cosine, context distance) are close to monotone, so a linear boundary already separates them well.
60. **What does the MLP training curve show?** Training loss per epoch and accuracy on a 10% validation hold-out. Early stopping ends training when that accuracy stops improving.
61. **Do RF/HGB have training curves?** RF has none (no iterations). HGB has a per-iteration loss with early stopping, which we plot. No artificial curves were drawn.
62. **How did you check overfitting?** Training-fold vs validation-fold metrics (gap), fold variability, and student-test vs CV.
63. ★ **RF fits training rows better than validation rows. Is that overfitting?** Some memorisation (min_samples_leaf=5), but its validation score stays top-tier and stable, so variance is controlled by bagging.
64. **Any underfitting?** Gaussian NB is the clearest case (low train *and* validation scores). LR is mildly underfit compared with the trees.
65. **Why does fold 1 score higher for every model?** Its validation trips are easier (cleaner GPS, sparser networks). That is trip difficulty, not model behaviour.
66. **How is inference time measured?** Wall-clock `predict_proba` on all student-test candidate rows, reported per 1000 fixes.
67. **What is the confusion matrix telling you?** How positives (true arcs) and negatives split into TP/FN/FP/TN at threshold 0.5. FN = true arc scored < 0.5, FP = wrong arc scored ≥ 0.5. Read the per-model counts in `model_confusion_matrices.png`; the error-type breakdown is in QUICK_REVISION.
68. **What is route continuity?** Share of consecutive output arcs that are connected in the network (to_node(a) = from_node(b)).
69. **Why is classifier-only continuity low?** Independent per-fix choices flip between parallel arcs, directions and junction neighbours.
70. **What is selection Macro-F1?** Macro-F1 of the final decision (chosen = 1). It makes the classifier-only and HMM outputs comparable.
71. **What is edit similarity?** 1 − normalised Levenshtein distance between predicted and reference arc sequences.
72. **What is route F1?** Length-weighted overlap between the predicted and reference arc sets.
73. **Why report several route metrics?** Each catches a different failure: continuity (jumps), F1 (wrong roads), edit/LCS (order, missing and extra arcs).
74. **What are the upper bounds on Top-1?** 1 − miss rate (≈ 98.9% train, 99.85% student-test).
75. **Is the student-test set used for training?** Never. It is scored once per final model, after CV-based choices.

## D. Bayesian optimisation (76–88)

76. **What is Bayesian optimisation?** Sequential model-based search: a surrogate of objective vs hyperparameters proposes the next most promising configuration.
77. **Which sampler?** Optuna TPE. It models good and bad trials as two densities and picks points that maximise their ratio.
78. **What objective?** Mean Macro-F1 over the same 5 trip-grouped folds of the training trips.
79. **Which hyperparameters?** learning_rate, max_iter, max_leaf_nodes, max_depth, min_samples_leaf, l2_regularization, max_features.
80. **Why those ranges?** Centred on the baseline. They span low to high capacity and weak to strong regularisation. Log scales are used where the effect is multiplicative.
81. **How many trials?** 40 (see `experiments/optimisation/optimisation_summary.json` for completed and pruned counts).
82. **What is pruning?** The MedianPruner stops a trial early if its running fold mean is below the median of earlier trials at that fold.
83. **Why every 4th fix during search?** Speed. The best configuration is re-scored with the exact baseline protocol, so the comparison is fair.
84. **Did tuning help?** See `tuned_vs_untuned.csv`. The answer is whatever the measured deltas say, including if they are within noise.
85. ★ **How do you avoid overfitting the CV during tuning?** Few, smooth hyperparameters, a fixed fold assignment, and confirmation on the untouched student-test trips.
86. **How do you choose tuned vs untuned for deployment?** By CV Macro-F1 only (rule in `optimise.py`). The student-test score is reported, not used.
87. **Computational cost?** `search_seconds` in the summary JSON. Each trial is 5 fold fits.
88. **Which hyperparameter mattered most?** fANOVA importance in `optimisation_summary.json` / `optimisation_history_importance.png`.

## E. HMM & Viterbi (89–108)

89. **What is an HMM?** Hidden states evolve as a Markov chain. We see observations generated by the states, and we infer the most likely state sequence.
90. **Hidden states here?** The candidate arcs of each fix (top 8 by emission), each keeping its arc_id, projection fraction and probability.
91. **Observations?** The GPS fixes.
92. **Emission probability?** `w_clf · log P_classifier`, plus an optional Gaussian distance term, with the probability floored at 1e-6.
93. **Transition probability?** `exp(−|d_route − d_gps| / β)`, minus a U-turn penalty. −∞ if unreachable or if the implied speed is > 55 m/s.
94. **How is d_route computed?** Bounded Dijkstra on the directed graph from the end of arc a to the start of arc b, plus the partial lengths on both arcs. Results are cached per source node.
95. ★ **Why compare d_route with d_gps?** On the true path the driving distance between two fixes is close to their straight-line distance. Big differences mean detours, i.e. implausible paths.
96. **Viterbi recurrence?** δ_t(s) = max_r [δ_{t−1}(r) + log T(r,s)] + log E_t(s), with back-pointers ψ. Backtrack from argmax δ_T.
97. **Why log space?** Products of hundreds of probabilities underflow. Sums of logs are stable.
98. **Complexity?** O(T · K²) transitions per trip (K ≤ 8), plus cached Dijkstra searches.
99. **Missing candidates?** No prediction for that fix. The chain ends and restarts after it.
100. **Large GPS gaps?** dt > 120 s forces a new chain. We don't claim a path across an unobserved gap.
101. **Disconnected consecutive candidates?** If every transition is −∞, a new chain starts at that fix (`no_valid_transition`). We never draw a fake link.
102. **One-observation trip?** Argmax of the emissions.
103. **How did you tune the HMM?** Grid over w_clf, β, w_dist, uturn_penalty on 30 training trips, using out-of-fold classifier probabilities.
104. ★ **Why out-of-fold probabilities?** In-sample probabilities are over-confident. The decoder must be tuned on probabilities like the ones it will see on new trips.
105. **Did the HMM help?** Yes on the student-test trips (see QUICK_REVISION). Per-trip gains and the fixes it broke are in `decoder_comparison_final.csv`.
106. **Where can the HMM hurt?** Where the classifier was right but a long low-probability path is smoother, e.g. around the reference's own route breaks, or at U-turns penalised by the model.
107. **What is route reconstruction?** Filling in arcs between consecutive matched arcs with the shortest path (inside a chain), so the output is drivable.
108. **Is reconstruction cheating?** No. It uses only the network and the predicted arcs, and gaps across chain breaks are left visible.

## F. System, API, UI, integrity (109–120)

109. **Why FastAPI?** Typed Pydantic schemas, automatic docs (`/docs`), async with thread offloading, and simple to deploy with Uvicorn.
110. **Which endpoints exist?** /health, /api/system/info, /api/model/info, /api/trips, /api/trips/{id}/metadata, POST /api/predict/{id}, /api/results/{id}, /api/results/{id}/explain/{fix}.
111. **How does Laptop 2 reach Laptop 1?** HTTP to `http://<laptop-1-LAN-IP>:8000`. The API binds 0.0.0.0, CORS allows the UI origin, and the Windows firewall must allow port 8000.
112. **How do evaluators see real inference happening?** The Laptop 1 console logs every request (client IP, endpoint, status, ms) and the server-side accuracy line.
113. **How is ground truth kept off the UI?** The prediction path loads trips without `.route`. Only `RouteEvaluator` reads it, and it returns aggregates. A test asserts no labels or reference sequence in responses.
114. **What if evaluator data is missing?** DEVELOPMENT mode serves the student-test trips with a banner. Dropping the files into `evaluator_test/` switches to EVALUATION mode.
115. **What if the model file is missing?** The API still starts, /health reports `degraded` with the reason, and predict returns 503.
116. **Timeouts?** Prediction runs in a thread under `asyncio.wait_for`. It returns 504 after `PATHSNAP_PREDICT_TIMEOUT`.
117. **Malformed input?** Loaders raise `DataFormatError` naming the file and problem → 422.
118. **Are outputs hardcoded?** No. Every POST runs the pipeline on the raw files. Repeated calls are deterministic, and timings differ per trip.
119. **How would you improve the system?** Curved road geometry, learned transition models, online/streaming Viterbi, more candidates where the miss rate is high, and probability calibration per region.
120. **What did AI tools do in this project, and what did you verify?** The code was AI-assisted (disclosed). We verify through tests, re-running scripts and being able to explain every component (this guide).
