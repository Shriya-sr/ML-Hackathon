# PathSnap: Phase 1 (Data, EDA, Features, 5 Baseline Models, Evaluation)

UE24CS352A Machine Learning hackathon · PES University, Department of CSE, RR Campus
Team: Sujal · Shriya · Sudheeksh · Shubashitha

This branch (`phase1-baseline`) contains **only the work presented at the first evaluation**: dataset
understanding, EDA, preprocessing, candidate generation, feature engineering, five baseline models and their
evaluation. Sequence decoding (HMM/Viterbi), Bayesian optimisation, the API and the UI belong to the next phase.

**Start here → [`docs/PHASE1_EXPLANATION.md`](docs/PHASE1_EXPLANATION.md)** (what data we got, what EDA we did and
how, how labels/candidates/features are built, what the five models do, which metrics we used, and the results).

## Results at a glance (5-fold trip-grouped CV on 80 training trips; student test = 10 held-out trips)

| Model | Family | CV Macro-F1 | CV Top-1 | Student Top-1 | Student Macro-F1 | Student log loss |
|---|---|---|---|---|---|---|
| Random Forest | Bagging ensemble | 0.9336 ± 0.0144 | 0.8840 | 0.8942 | 0.9344 | 0.0784 |
| HistGradientBoosting | Boosting | 0.9332 ± 0.0138 | 0.8840 | 0.8950 | 0.9347 | 0.0828 |
| Logistic Regression | Linear | 0.9262 ± 0.0159 | 0.8820 | 0.8834 | 0.9256 | 0.1019 |
| RBF SVM (Platt) | Kernel | 0.9244 ± 0.0167 | 0.8789 | 0.8789 | 0.9251 | 0.1073 |
| k-Nearest Neighbours | Instance-based | 0.9223 ± 0.0167 | 0.8711 | 0.8809 | 0.9241 | 0.1344 |

Source: `experiments/results/model_summary.csv`. Candidate miss rate: 1.14% (train), 0.15% (student test).

## Layout

| Path | Content |
|---|---|
| `src/pathsnap/data/` | loaders, validation, cleaning, per-fix label alignment, trip processing |
| `src/pathsnap/geography/` | UTM projection, point-segment geometry, STRtree index, road network |
| `src/pathsnap/features/` | candidate generation, 21 features, feature schema |
| `src/pathsnap/models/` | the five baseline models |
| `src/pathsnap/training/` | trip-grouped cross-validation, experiment tracking |
| `src/pathsnap/evaluation/` | classification, Top-1 and route metrics |
| `src/pathsnap/visualisation/` | plotting helpers, Folium maps |
| `scripts/` | `run_eda.py`, `build_features.py`, `train_all.py`, `feature_analysis.py`, `make_report_figures.py`, `generate_maps.py`, `capture_screenshots.py` |
| `configs/` | data, feature and model settings |
| `reports/eda/` | inventory, data-quality report, timestamp/network/trajectory tables, candidate sweep, EDA figures |
| `reports/figures/` | model comparison, confusion matrices, generalisation gap, feature importance/ablation |
| `experiments/` | CV folds, per-model records (config, seed, fold metrics, timings), summary |
| `maps/` | interactive Folium EDA maps + screenshots |
| `tests/unit/` | unit tests for data, geometry, candidates/features, metrics, models |

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pip install -e .
$env:PATHSNAP_DATA_DIR = "D:\ML hackathon"        # folder containing train\ and student_test\ (see data/README.md)
.\.venv\Scripts\python.exe scripts\run_eda.py
.\.venv\Scripts\python.exe scripts\build_features.py --jobs 6
.\.venv\Scripts\python.exe -W ignore scripts\train_all.py --fresh
.\.venv\Scripts\python.exe scripts\feature_analysis.py
.\.venv\Scripts\python.exe scripts\make_report_figures.py
.\.venv\Scripts\python.exe -m pytest -p no:warnings
```

## AI-assistance disclosure

As required by the submission guidelines, we disclose that the implementation was developed with the help of an
AI coding assistant under the team's direction. All results come from running the scripts in this repository on the
supplied data.
