# Results Summary (generated from experiment records)

## Model comparison (classifier only)

| Model | Family | CV Macro-F1 | CV Top-1 | Student Top-1 | Student Macro-F1 | Student log loss | Route continuity | Inference ms/1k fixes |
|---|---|---|---|---|---|---|---|---|
| Random Forest | Ensemble (bagging) | 0.9336 ± 0.0144 | 0.8840 | 0.8942 | 0.9344 | 0.0784 | 0.7713 | 22.1 |
| HistGradientBoosting | Boosting | 0.9332 ± 0.0138 | 0.8840 | 0.8950 | 0.9347 | 0.0828 | 0.7701 | 10.4 |
| Extra Trees | Ensemble (randomised trees) | 0.9316 ± 0.0148 | 0.8802 | 0.8872 | 0.9314 | 0.0836 | 0.7502 | 26.8 |
| MLP (64-32) | Neural network | 0.9296 ± 0.0145 | 0.8836 | 0.8868 | 0.9295 | 0.0878 | 0.7668 | 5.1 |
| Logistic Regression | Linear / probabilistic | 0.9262 ± 0.0159 | 0.8820 | 0.8834 | 0.9256 | 0.1019 | 0.7562 | 0.9 |
| RBF SVM (Platt-calibrated) | Kernel-based | 0.9244 ± 0.0167 | 0.8789 | 0.8789 | 0.9251 | 0.1073 | 0.7488 | 1151.1 |
| k-Nearest Neighbours | Instance-based | 0.9223 ± 0.0167 | 0.8711 | 0.8809 | 0.9241 | 0.1344 | 0.7371 | 314.8 |
| Gaussian Naive Bayes | Probabilistic (generative) | 0.9013 ± 0.0118 | 0.8596 | 0.8593 | 0.8966 | 0.4316 | 0.7176 | 3.9 |

## Bayesian optimisation (HistGradientBoosting)

| Metric | Untuned | Tuned | Δ |
|---|---|---|---|
| cv_macro_f1 | 0.9332 | 0.9343 | +0.0011 |
| cv_top1 | 0.8840 | 0.8861 | +0.0021 |
| cv_log_loss | 0.0896 | 0.0853 | -0.0044 |
| st_top1 | 0.8950 | 0.9000 | +0.0050 |
| st_macro_f1 | 0.9347 | 0.9373 | +0.0026 |
| st_log_loss | 0.0828 | 0.0767 | -0.0061 |
| inference_ms_per_1k_fixes | 10.4434 | 100.7272 | +90.2838 |

## Classifier-only vs classifier + HMM (student-test, final classifier, tuned HMM)

| Metric | Classifier only | + HMM | + HMM + reconstruction |
|---|---|---|---|
| Top-1 accuracy (pooled fixes) | 0.9000 | 0.9544 | (same as HMM) |
| Selection Macro-F1 (mean of trips) | 0.9517 | 0.9765 | |
| Route continuity | 0.7820 | 0.9009 | 0.9967 |
| Route F1 | 0.9491 | 0.9839 | 0.9889 |
| Edit similarity | 0.6654 | 0.8771 | 0.9420 |
| Invalid transitions (sum) | 1108 | 343 | 12 |
| Candidate miss rate | 0.0015 | 0.0015 | |

## Held-out evaluator trips (frozen pipeline, scored once after all modelling decisions)

Source: `experiments/results/heldout_evaluator_summary.json` (`scripts/evaluate_heldout.py`). The 10 trips share no trip ID or track file with train/student-test and were never used for training, tuning or selection.

| Metric | Student-test: classifier | Student-test: + HMM | Evaluator: classifier | Evaluator: + HMM |
|---|---|---|---|---|
| Top-1 accuracy (pooled fixes) | 0.9000 | 0.9544 | 0.9080 | 0.9590 |
| Selection Macro-F1 (mean) | 0.9517 | 0.9765 | 0.9577 | 0.9734 |
| Route continuity (mean, + recon.) | 0.7820 | 0.9967 | 0.7889 | 0.9937 |
| Route F1 (mean, + recon.) | 0.9491 | 0.9889 | 0.9261 | 0.9867 |
| Edit similarity (mean, + recon.) | 0.6654 | 0.9420 | 0.6886 | 0.9094 |
| Invalid transitions (total) | 1108 | 12 | 1070 | 12 |
| Candidate miss rate | 0.0015 | | 0.0023 | |

Classifier on evaluator candidates: Macro-F1 (threshold 0.5) 0.9422, log loss 0.0766, ROC-AUC 0.9926. The HMM fixed 1145 fixes and broke 142.

| Trip | fixes | miss rate | clf Top-1 | HMM Top-1 | clf contin. | recon contin. | recon route F1 |
|---|---|---|---|---|---|---|---|
| 00000015 | 6854 | 0.0015 | 0.8383 | 0.9546 | 0.749 | 0.973 | 0.952 |
| 00000017 | 1346 | 0.0149 | 0.9146 | 0.9257 | 0.686 | 0.998 | 0.982 |
| 00000023 | 458 | 0.0044 | 0.9410 | 0.9541 | 0.823 | 0.987 | 0.992 |
| 00000030 | 654 | 0.0138 | 0.8150 | 0.8853 | 0.645 | 0.997 | 0.974 |
| 00000049 | 2642 | 0.0000 | 0.9871 | 0.9921 | 0.909 | 1.000 | 0.999 |
| 00000059 | 1031 | 0.0029 | 0.9253 | 0.9389 | 0.739 | 0.998 | 0.994 |
| 00000067 | 916 | 0.0000 | 0.9727 | 0.9421 | 0.811 | 0.987 | 0.986 |
| 00000073 | 1144 | 0.0009 | 0.9729 | 0.9790 | 0.905 | 0.996 | 0.999 |
| 00000087 | 1399 | 0.0000 | 0.9492 | 0.9907 | 0.836 | 1.000 | 0.998 |
| 00000094 | 3230 | 0.0003 | 0.9378 | 0.9610 | 0.786 | 1.000 | 0.990 |
