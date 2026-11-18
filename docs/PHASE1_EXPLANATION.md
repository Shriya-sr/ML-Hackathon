# PathSnap: Phase 1 Explanation (Data → EDA → Features → 5 Baseline Models → Evaluation)

UE24CS352A Machine Learning hackathon · PES University · Team: Sujal, Shriya, Sudheeksh, Shubashitha

This document explains everything done up to the first evaluation: what data we received, how we explored it,
how we turned it into a supervised learning problem, which five baseline models we trained, and how we evaluated
them. Every number comes from files in this branch (`reports/eda/`, `experiments/results/`) produced by the scripts
in `scripts/`.

---

## 1. The problem in one paragraph

A car records its position with GPS about once per second, but GPS is noisy: fixes wander a few metres to tens of
metres, jump onto parallel streets and sometimes disappear for a while. We are given the road network around each
trip. The task, **map matching**, is to decide for every GPS fix which road segment the car was actually on.
In Phase 1 we solve it as **supervised candidate classification**: for each GPS fix we list the nearby road segments
("candidates"), describe each candidate with features, and train models that estimate the probability that a candidate
is the correct segment. The highest-probability candidate is the prediction for that fix.

---

## 2. The data we received

### 2.1 Partitions

| Partition | Trips | GPS fixes | Use |
|---|---|---|---|
| Training | 80 | 208 417 | model fitting and cross-validation |
| Student testing | 10 | 19 160 | development-stage comparison of models |
| Evaluator testing | 10 | not supplied yet | final evaluation only (never used for development) |

Each trip folder `<id>/` contains four whitespace-separated, header-less files. **All 90 trips × 4 = 360 files are present.**

| File | Columns | Meaning |
|---|---|---|
| `<id>.track` | longitude, latitude, timestamp (s) | the GPS fixes, in time order, about 1 Hz |
| `<id>.nodes` | longitude, latitude | road-network points; **row number = node_id** |
| `<id>.arcs` | from_node_id, to_node_id | directed straight road segments; **row number = arc_id** |
| `<id>.route` | arc_id | the reference (ground-truth) sequence of arcs the car drove |

### 2.2 How the files relate (a key finding)

* Arcs point to nodes; the route points to arcs; the track points to nothing. **There is no row-to-row link between
  `.track` and `.route`.**
* The route has **far fewer rows than the track** (median **5.3 GPS fixes per route arc**), because one road
  segment is driven for several seconds. So the dataset does not directly tell us "fix i is on arc j".
* Most roads are two-way and appear as **two opposite arcs** (median **95.1%** of arcs have a reverse twin).
  The correct answer is a *directed* arc, so the model must also get the **direction** right.

---

## 3. Exploratory Data Analysis (EDA)

Script: `scripts/run_eda.py` → `reports/eda/*.csv`, `reports/eda/data_quality_report.md`, `reports/eda/figures/01–08`.
Partitions used: training + student-test (marked by colour in the figures). The evaluator partition was only checked
for file presence.

### 3.1 What we checked and how

| Check | How it was calculated | Result |
|---|---|---|
| File availability, sizes | listed every trip folder, checked the 4 expected files exist | 360/360 present (`dataset_inventory.csv`) |
| Missing / invalid GPS values | non-numeric cells → NaN; longitude outside ±180 or latitude outside ±90 → invalid | **0** invalid coordinates, **0** invalid timestamps |
| Duplicates | identical rows; repeated timestamps | **0** and **0** |
| Timestamp ordering | sign of consecutive time differences `dt` | all trips strictly increasing |
| Sampling rate / gaps | histogram of `dt`; gaps = `dt > 30 s` | median **99.1%** of steps are exactly 1 s per trip; **135 gaps** in **30 trips**; largest **153 064 s** (~42 h) (`timestamp_analysis.csv`, Fig. 02) |
| Trip size | fixes per trip, route length (sum of arc lengths), duration | 262–19 313 fixes (median 1 496); route 5.0–94.3 km (median 14.3 km) (Fig. 01) |
| Geographic extent | min/max lon/lat per trip | trips on several continents: lon −100.4…137.2, lat −34.6…61.7 (Fig. 03); **25 different UTM zones** |
| Network size | rows in `.nodes` / `.arcs` | 299–155 003 nodes, 649–307 392 arcs (Fig. 04) |
| Connectivity | `scipy.sparse.csgraph.connected_components` on the arc graph, weak and strong | 0 isolated nodes; median 19 weakly connected pieces; largest strongly connected component = median 96.4% of nodes (`network_connectivity.csv`) |
| Network anomalies | self-loops, duplicate arcs | 24 self-loops, 1 903 duplicate arcs |
| Reference route validity | arc ids in range; consecutive arcs share a node (to_node(a) = from_node(b)) | all ids valid; **14 route breaks in 11 trips** → the reference is not perfect |
| GPS noise | distance from each fix to its reference arc (after alignment, §4.3) | median **4.7 m** per trip; 2.03% of fixes more than 60 m away (Fig. 06) |
| Trajectories | GPS fixes drawn over the road network and the reference route | Fig. 08 and the Folium maps |

**Folium maps** (interactive, `maps/html/`, screenshots in `maps/screenshots/`): training trips 00000001 and
00000054 showing raw GPS fixes, the road network around the trajectory, the reference route, start/end markers,
GPS gaps, a legend and a layer control. Trip 00000054 shows a typical anomaly: a car **parked off the road network** for
a long time, producing a cloud of jittering fixes.

### 3.2 What EDA told us (and what we did about it)

| Observation | Consequence for our design |
|---|---|
| Data is world-wide; lon/lat are angles | Distances in degrees are wrong. **Each trip is projected into the UTM zone of its own centre**, so everything is in metres (error < 0.1%) |
| `.route` is shorter than `.track` | We **derive one label per fix** by aligning the fixes to the route (§4.3) |
| Two-way roads = two identical-looking arcs | We need **heading features** (direction of motion vs direction of the arc) |
| Gaps, parked periods, route breaks | Flag gaps; mark fixes far from the reference as unlabelled; treat the reference as "reference", not perfect truth |
| Up to 307k arcs per trip | Use a **spatial index** instead of comparing every fix with every arc |
| No missing values in the supplied files | Loaders still validate everything (the evaluator data might be messier) |

---

## 4. Preprocessing and problem construction

### 4.1 Loading and cleaning (`src/pathsnap/data/loader.py`, `preprocessing.py`)
* Files are parsed as whitespace-separated, header-less tables. Wrong column counts, non-integer ids, ids that point
  outside the node/arc tables, and empty or missing files raise a clear error naming the file.
* Cleaning: drop invalid coordinates/timestamps, sort by time, drop duplicate timestamps. Each cleaned fix keeps its
  original row number.

### 4.2 Projection (`src/pathsnap/geography/projection.py`)
`pyproj` converts lon/lat into the trip's UTM zone (EPSG 326xx north / 327xx south). All distances, bearings and
speeds afterwards are in metres. A unit test checks the projected distance against the great-circle (haversine)
distance (difference < 0.2%).

### 4.3 Per-fix labels by monotone alignment (`src/pathsnap/data/labeling.py`)
To know which arc each fix belongs to, we align the ordered fixes to the ordered route arcs. The rule is that the car
never goes backwards along its own route, so the assigned route position must not decrease over time. Cost of putting
fix i on route position j:

```
cost(i, j) = min(distance(fix i, arc j), 100 m)  +  15 m × (1 − cos(angle between motion and arc direction)) / 2
```

Dynamic programming finds the lowest-cost assignment exactly: `C[i, j] = cost(i, j) + min over j' ≤ j of C[i−1, j']`.
The heading term separates a road's two directions on out-and-back trips. Fixes whose aligned arc is **more than
60 m** away (mostly parked cars) get label −1 and are excluded from training and from accuracy denominators.
**The reference route is used only to create labels. It never influences candidates or features.**

### 4.4 Candidate generation (`src/pathsnap/features/candidate_generation.py`)
* All arcs are inserted into an **STRtree** (an R-tree spatial index). One query returns, for every fix, the arcs
  within radius **R**; we keep the **K** nearest. If none are in range, the single nearest arc within 150 m is used.
* **Choosing R and K**: we measured the **candidate miss rate** (share of labelled fixes whose true arc is not among the
  candidates) on the training trips for R ∈ {10…100 m} and K ∈ {2…24} (`candidate_radius_sweep_train.csv`, Fig. 07):

  | R (K = 10) | 25 m | 35 m | **50 m** | 75 m | 100 m |
  |---|---|---|---|---|---|
  | miss rate | 3.09% | 1.81% | **1.22%** | 1.05% | 1.05% |
  | candidates per fix | 4.5 | 5.5 | **6.6** | 7.8 | 8.7 |

  **R = 50 m, K = 10** is the knee of the curve: larger radii add candidates (more confusion and cost) for little
  coverage. With the 150 m fallback, the final miss rate is **1.14% (train)** and **0.15% (student test)**. The miss rate is
  a ceiling: no model can be right on those fixes. We never insert the true arc into the candidate list (that would be
  cheating).
* Result (`experiments/logs/build_features.log`): **1 246 339** candidate rows from training trips (184 089 correct =
  14.8%) and **131 544** from student-test trips.

### 4.5 Features (`src/pathsnap/features/feature_extraction.py`), 21 per (fix, candidate)

| Group | Features | Why |
|---|---|---|
| Geometric (9) | distance to segment, rank by distance, distance minus nearest distance, distance ratio to nearest, number of candidates, projection position along the arc, beyond-the-end flag, distance to nearest endpoint, arc length | closeness *relative to the other candidates* matters more than raw metres |
| Heading (4) | cos(motion direction over fixes i±2 vs arc direction), the same angle in degrees, heading-valid flag (moved ≥ 3 m), cos of a longer ±5-fix bearing | tells the two directions of a road apart, and separates crossing streets |
| Context (2) | mean and max distance of the neighbouring fixes (i−3…i+3) to this arc | a road that fits the neighbours is more plausible than one that fits a single noisy fix |
| Kinematic (3) | speed, time to previous fix, time to next fix | heading is unreliable when slow; gaps reduce context |
| Network (3) | in-degree of start node, out-degree of end node, two-way flag | junction structure, one-way roads |

**Which features matter?** (`scripts/feature_analysis.py`, training trips only, HistGradientBoosting)
* **Permutation importance** (increase in log loss when a feature is shuffled): `dist_minus_min` 0.251,
  `cos_long_bearing` 0.174, `win_mean_dist_m` 0.040, `endpoint_dist_m` 0.035, `cos_heading` 0.032, `speed_mps` 0.021.
* **Group ablation** (drop a group, re-run cross-validation):

  | Removed group | Δ CV Macro-F1 | Δ CV Top-1 |
  |---|---|---|
  | heading | −20.6 pts | **−33.2 pts** |
  | geometric | −5.3 pts | −4.1 pts |
  | context | −0.3 pts | −0.6 pts |
  | network | −0.3 pts | −0.1 pts |
  | kinematic | −0.1 pts | −0.2 pts |

  Every group helps. Direction is by far the most important information, because most wrong candidates are the
  same road in the opposite direction or a neighbouring road.

---

## 5. Evaluation protocol

* **Unit of classification**: one row per (GPS fix, candidate arc); label 1 if the arc is the fix's reference arc, else 0.
* **Training rows**: candidates of every 2nd labelled fix (1 Hz fixes are almost duplicates; this halves the cost),
  **603 593 rows** for the full training set. Validation and test always use **all** labelled fixes.
* **Cross-validation**: **5-fold GroupKFold with group = trip** (16 trips per fold, seed 42; fold lists in
  `experiments/configurations/cv_folds.json`). All fixes of a trip stay in the same fold. A random split would put
  near-identical neighbouring fixes in both training and validation and inflate the scores.
* **Student test**: each model is refit on all 80 training trips and scored once on the 10 student-test trips.
* **Same features, same folds, same test trips for every model.** Scale-sensitive models standardise features inside
  a pipeline that is re-fitted on each training fold, so no information leaks from validation data.

## 6. Evaluation metrics (`src/pathsnap/evaluation/metrics.py`, `route_metrics.py`)

| Metric | Level | Definition | Why we use it |
|---|---|---|---|
| Accuracy | candidate rows | correct / all rows at threshold 0.5 | required, but misleading: only ~15% of rows are positive |
| Precision / Recall (positive, macro) | candidate rows | TP/(TP+FP), TP/(TP+FN) | how many predicted "correct roads" are right / how many correct roads are found |
| **Macro-F1** (primary CV metric) | candidate rows | mean of the F1 of class 0 and class 1 | balances the rare positive class with the common negative class |
| Confusion matrix | candidate rows | TN, FP, FN, TP at threshold 0.5 | shows the error types |
| **Log loss** | candidate rows | −mean[y·log p + (1−y)·log(1−p)] | quality of the probabilities; punishes confident mistakes |
| ROC-AUC | candidate rows | ranking quality independent of the threshold | supporting metric |
| **Top-1 accuracy** | GPS fixes | highest-probability candidate = reference arc. Fixes whose true arc is missing, or that have no candidates, count as wrong | the real map-matching objective |
| Candidate miss rate | GPS fixes | true arc not among the candidates | upper bound on Top-1 |
| Route continuity | trip route | share of consecutive predicted arcs that are connected (end node of one = start node of next) | is the predicted route drivable? |
| Route F1 | trip route | length-weighted overlap of predicted vs reference arcs | did we pick the right roads? |
| Training / inference time | model | wall-clock fit time; prediction time per 1 000 fixes | computational efficiency |

Train-fold scores are also recorded to measure the **generalisation gap** (overfitting check).

---

## 7. The five baseline models (`src/pathsnap/models/model_factory.py`, `configs/models.yaml`)

| # | Model | Family | What it does | Configuration |
|---|---|---|---|---|
| 1 | **Logistic Regression** | linear / probabilistic | weighted sum of features → sigmoid → probability; fits a straight decision boundary | C = 1.0 (L2), lbfgs, standardised features |
| 2 | **k-Nearest Neighbours** | instance-based | finds the 25 most similar training candidates; probability = distance-weighted share that were correct | k = 25, distance weights, standardised, 60 000 stored rows (prediction cost grows with stored rows) |
| 3 | **Random Forest** | ensemble (bagging) | 200 decision trees on bootstrap samples, each split on random feature subsets; averages their votes | 200 trees, unlimited depth, min 5 samples per leaf, √features per split |
| 4 | **HistGradientBoosting** | boosting | adds small trees one after another, each correcting the previous errors by minimising log loss; features binned into histograms for speed | learning rate 0.1, up to 300 trees with early stopping, 31 leaves, min 20 samples per leaf |
| 5 | **RBF SVM** | kernel-based | finds the maximum-margin boundary in a non-linear (Gaussian kernel) feature space; probabilities via Platt scaling (a sigmoid fitted on cross-validated scores) | C = 1, gamma = scale, standardised, 20 000-row trip-stratified subsample (kernel training is O(n²)) |

Loss / objective: LR = log loss + L2; KNN = none (lazy learner); RF = Gini impurity per split; HGB = log loss
(gradient boosting); SVM = hinge loss + margin, then a sigmoid fitted by log loss.

---

## 8. Results (real, from `experiments/results/model_summary.csv`)

### 8.1 Comparison table

| Model | CV Macro-F1 (mean ± std) | CV Top-1 | CV log loss | Student Top-1 | Student Macro-F1 | Student log loss | Student ROC-AUC | Route continuity | Fit time | Inference (ms / 1 000 fixes) |
|---|---|---|---|---|---|---|---|---|---|---|
| Random Forest | **0.9336 ± 0.0144** | 0.8840 | **0.0862** | 0.8942 | 0.9344 | **0.0784** | **0.9918** | **0.7713** | 66.1 s | 22.1 |
| HistGradientBoosting | 0.9332 ± 0.0138 | 0.8840 | 0.0896 | **0.8950** | **0.9347** | 0.0828 | 0.9913 | 0.7701 | 3.2 s | 10.4 |
| Logistic Regression | 0.9262 ± 0.0159 | 0.8820 | 0.1086 | 0.8834 | 0.9256 | 0.1019 | 0.9852 | 0.7562 | 2.0 s | **0.9** |
| RBF SVM (Platt) | 0.9244 ± 0.0167 | 0.8789 | 0.1108 | 0.8789 | 0.9251 | 0.1073 | 0.9829 | 0.7488 | 6.9 s (20k rows) | 1 151 |
| k-Nearest Neighbours | 0.9223 ± 0.0167 | 0.8711 | 0.1525 | 0.8809 | 0.9241 | 0.1344 | 0.9808 | 0.7371 | 0.02 s (lazy) | 315 |

Student test = 10 trips, 19 135 labelled fixes, 131 544 candidate rows. Route continuity is that of the
independent per-fix predictions (no sequence model yet).

### 8.2 Student-test confusion matrices (threshold 0.5)

| Model | TN | FP | FN | TP | Precision (pos) | Recall (pos) |
|---|---|---|---|---|---|---|
| Random Forest | 110 721 | 1 661 | 2 544 | 16 563 | 0.909 | 0.867 |
| HistGradientBoosting | 110 469 | 1 913 | 2 314 | 16 793 | 0.898 | 0.879 |
| Logistic Regression | 109 854 | 2 528 | 2 352 | 16 755 | 0.869 | 0.877 |
| RBF SVM | 110 500 | 1 882 | 2 904 | 16 203 | 0.896 | 0.848 |
| k-NN | 109 971 | 2 411 | 2 536 | 16 571 | 0.873 | 0.867 |

### 8.3 Paired comparison on the same folds

Because all models use the same five folds, we compare them fold by fold (`cv_folds_*.csv`):

| HGB vs … | folds where HGB has higher Macro-F1 | folds where HGB has lower log loss |
|---|---|---|
| Random Forest | 2 / 5 | 0 / 5 |
| Logistic Regression | 5 / 5 | 5 / 5 |
| RBF SVM | 5 / 5 | 5 / 5 |
| k-NN | 5 / 5 | 5 / 5 |

The two tree ensembles beat the linear, kernel and instance-based models in **every** fold. Random Forest and
HistGradientBoosting are effectively **tied** (differences far smaller than the fold-to-fold spread).

### 8.4 Overfitting / underfitting (training-fold vs validation-fold Macro-F1)

| Model | Train-fold | Validation | Gap | Diagnosis |
|---|---|---|---|---|
| Random Forest | 0.9709 | 0.9336 | +0.037 | overfits (fully grown trees memorise training rows), yet validation is still the best, so bagging keeps it generalising |
| HistGradientBoosting | 0.9513 | 0.9332 | +0.018 | mild overfitting, controlled by early stopping (curve in `model_training_histories.png`) |
| k-NN | 0.9407 | 0.9223 | +0.018 | mild overfitting (neighbours include the training points themselves) |
| RBF SVM | 0.9305 | 0.9244 | +0.006 | no overfitting; slightly underfits compared with the trees |
| Logistic Regression | 0.9286 | 0.9262 | +0.002 | no overfitting; **mild underfitting** (high bias): even its training score is below the trees' validation score, because a linear boundary cannot model interactions such as "heading only matters while moving" |

For every model the student-test Macro-F1 lies within two fold standard deviations of its CV score, so CV
generalised to unseen trips.

### 8.5 What we conclude at this stage

1. **Primary metric**: CV Macro-F1 (as the brief asks), read together with CV Top-1 (the actual map-matching goal),
   log loss (probability quality) and cost.
2. **Best baselines**: Random Forest and HistGradientBoosting (tied). HGB trains ~20× faster (3.2 s vs 66 s),
   predicts ~2× faster and its model file is ~300× smaller (0.19 MB vs 59 MB). That makes it the practical
   candidate for the next step (Bayesian hyperparameter optimisation needs many refits).
3. **Accuracy ceiling**: Top-1 ≈ 0.88–0.90 while the candidate miss rate allows ~0.99. The remaining errors are mostly
   **the reverse direction of the right road** and **neighbouring arcs at junctions**, which independent per-fix
   decisions cannot resolve. For HistGradientBoosting on the student test, the 2 009 wrong fixes split into 748 wrong
   direction, 853 junction neighbour and 408 different road (breakdown computed by the next phase's evaluation script).
4. **Route continuity is only ~0.74–0.77**: consecutive independent predictions often jump between unconnected arcs.
   This is the motivation for the next phase: **sequence decoding with an HMM + Viterbi** over the road network.

---

## 9. Next steps (not part of this phase)

* Bayesian hyperparameter optimisation (Optuna TPE, grouped CV) of the selected model.
* HMM/Viterbi route reconstruction using the classifier's probabilities and network connectivity.
* FastAPI backend (Laptop 1) and evaluator UI (Laptop 2), Folium prediction maps, technical report.

---

## 10. How to reproduce this phase

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pip install -e .
$env:PATHSNAP_DATA_DIR = "D:\ML hackathon"          # folder containing train\ and student_test\
.\.venv\Scripts\python.exe scripts\run_eda.py              # EDA tables + figures      (~5 min)
.\.venv\Scripts\python.exe scripts\build_features.py --jobs 6   # candidates + features (~2 min)
.\.venv\Scripts\python.exe -W ignore scripts\train_all.py --fresh   # 5 models, CV + student test (~35 min, SVM is the slowest)
.\.venv\Scripts\python.exe scripts\feature_analysis.py     # importance + ablation      (~10 min)
.\.venv\Scripts\python.exe scripts\make_report_figures.py  # model comparison figures
.\.venv\Scripts\python.exe scripts\generate_maps.py        # Folium EDA maps
.\.venv\Scripts\python.exe -m pytest -p no:warnings        # unit tests
```

## 11. Where things are

| What | Where |
|---|---|
| EDA tables and quality report | `reports/eda/` |
| EDA figures | `reports/eda/figures/` |
| Folium maps | `maps/html/`, `maps/screenshots/` |
| Model comparison figures | `reports/figures/model_*.png` |
| Feature importance / ablation | `reports/figures/feature_*.png`, `experiments/results/permutation_importance.csv`, `feature_ablation.csv` |
| Per-model records (config, seed, fold metrics, timings) | `experiments/results/experiment_<model>.json`, `cv_folds_<model>.csv` |
| Summary table | `experiments/results/model_summary.csv` |
| Code | `src/pathsnap/` (data, geography, features, models, training, evaluation, visualisation) |
