# Check of the delivered test folders (2026-11-18)

Run end-to-end on the folders in `D:\ML hackathon\` with the frozen final model (`hist_gradient_boosting_tuned`
+ HMM, artifacts unchanged). For each trip: validate the files, predict **without** the `.route` file, then score
on the server side against `.route`. Nothing was retrained or tuned after seeing these results.

## Data checks

| Check | Result |
|---|---|
| `student_test/` (10 trips, 40 files) | byte-identical to the student-testing data used during development |
| `train/` (80 trips) | byte-identical to the training data used |
| `evaluator_test/` (10 trips: 00000015, 17, 23, 30, 49, 59, 67, 73, 87, 94) | new; no trip ID and no `.track` content shared with train or student test |
| File validation | all 20 test trips valid |

## Student-test trips (same as reported earlier)

| trip | fixes | miss % | clf Top-1 | HMM Top-1 | clf continuity | HMM continuity | HMM route F1 | edit sim. | invalid trans. |
|---|---|---|---|---|---|---|---|---|---|
| 00000000 | 2503 | 0.12 | 0.9517 | 0.9676 | 0.771 | 0.985 | 0.996 | 0.877 | 3 |
| 00000007 | 1354 | 0.00 | 0.9727 | 0.9970 | 0.975 | 1.000 | 0.997 | 0.984 | 0 |
| 00000009 | 1049 | 1.14 | 0.8704 | 0.9123 | 0.741 | 0.993 | 0.963 | 0.922 | 3 |
| 00000040 | 2995 | 0.00 | 0.9429 | 0.9756 | 0.785 | 1.000 | 0.998 | 0.966 | 0 |
| 00000052 | 262 | 0.00 | 0.9847 | 0.9924 | 0.852 | 1.000 | 0.998 | 0.979 | 0 |
| 00000056 | 3958 | 0.08 | 0.9158 | 0.9657 | 0.752 | 0.998 | 0.991 | 0.985 | 3 |
| 00000065 | 768 | 0.00 | 0.9193 | 0.9453 | 0.787 | 1.000 | 0.986 | 0.948 | 0 |
| 00000076 | 1486 | 0.40 | 0.9253 | 0.9670 | 0.716 | 0.998 | 0.979 | 0.916 | 1 |
| 00000077 | 3911 | 0.10 | 0.7832 | 0.9095 | 0.746 | 0.994 | 0.986 | 0.854 | 2 |
| 00000091 | 874 | 0.00 | 0.8947 | 0.9531 | 0.694 | 1.000 | 0.995 | 0.991 | 0 |

Pooled over 19,135 labelled fixes: Top-1 **0.9000 → 0.9544**; the HMM improves 10/10 trips. Matches
`reports/final/RESULTS_SUMMARY.md`.

## Evaluator trips (held out, first and only run)

| trip | fixes | miss % | clf Top-1 | HMM Top-1 | clf continuity | HMM continuity | HMM route F1 | edit sim. | invalid trans. |
|---|---|---|---|---|---|---|---|---|---|
| 00000015 | 6854 | 0.15 | 0.8383 | 0.9546 | 0.749 | 0.973 | 0.952 | 0.711 | 5 |
| 00000017 | 1346 | 1.49 | 0.9146 | 0.9257 | 0.686 | 0.998 | 0.982 | 0.901 | 2 |
| 00000023 | 458 | 0.44 | 0.9410 | 0.9541 | 0.823 | 0.987 | 0.992 | 0.738 | 1 |
| 00000030 | 654 | 1.38 | 0.8150 | 0.8853 | 0.645 | 0.997 | 0.974 | 0.912 | 1 |
| 00000049 | 2642 | 0.00 | 0.9871 | 0.9921 | 0.909 | 1.000 | 0.999 | 0.998 | 0 |
| 00000059 | 1031 | 0.29 | 0.9253 | 0.9389 | 0.739 | 0.998 | 0.994 | 0.923 | 1 |
| 00000067 | 916 | 0.00 | 0.9727 | 0.9421 | 0.811 | 0.987 | 0.986 | 0.976 | 1 |
| 00000073 | 1144 | 0.09 | 0.9729 | 0.9790 | 0.905 | 0.996 | 0.999 | 0.975 | 1 |
| 00000087 | 1399 | 0.00 | 0.9492 | 0.9907 | 0.836 | 1.000 | 0.998 | 0.998 | 0 |
| 00000094 | 3230 | 0.03 | 0.9378 | 0.9610 | 0.786 | 1.000 | 0.990 | 0.963 | 0 |

Pooled over 19,672 labelled fixes: Top-1 **0.9080 → 0.9590**. The HMM improves 9/10 trips; on 00000067 it is
worse (0.9727 → 0.9421). The held-out result is in line with the student-test result, so the model generalises
to unseen trips. Weakest trips: 00000030 (0.885, highest candidate miss rate) and 00000015 (longest trip,
lowest edit similarity 0.711).

## System checks

| Check | Result |
|---|---|
| Python tests (`pytest`), project data and with evaluator trips present | 92 passed / 92 passed |
| UI tests (`vitest`) + type check (`tsc -b`) | 12 passed, OK |
| Live API with `PATHSNAP_DATA_DIR=D:\ML hackathon` | mode **EVALUATION**, 10 trips; `lan_check --all` ALL CHECKS PASSED on 127.0.0.1 and over Wi-Fi |
| API server log | 0 errors, every request 200; invalid trip id → 404 |
| Ground-truth leakage | prediction responses contain no reference-route or label fields |

Fix made during this check: `backend/tests/test_api.py` assumed the server shows the student-test trips. With
real evaluator trips present it (correctly) switches to EVALUATION mode, so 4 API tests failed. The tests now pin
the partition to `student_test`; the system code itself was not changed.
