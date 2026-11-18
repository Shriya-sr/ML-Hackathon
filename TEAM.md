# Team PathSnap

UE24CS352A Machine Learning hackathon · PES University, Department of CSE, RR Campus

| Member | Responsible for | Main code and outputs |
|---|---|---|
| **Sujal** | Inference pipeline, FastAPI backend, two-laptop deployment, testing | `src/pathsnap/inference/`, `backend/`, `docs/deployment.md`, `tests/`, `scripts/lan_check.py` |
| **Shriya** | Data exploration, preprocessing, label alignment, candidate generation, features; repository owner | `src/pathsnap/data/`, `geography/`, `features/`, `scripts/run_eda.py`, `build_features.py`, `reports/eda/` |
| **Sudheeksh** | Model zoo, trip-grouped cross-validation, metrics, Bayesian optimisation | `src/pathsnap/models/`, `training/`, `evaluation/`, `optimisation/`, `scripts/train_all.py`, `optimise.py` |
| **Shubashitha** | HMM / Viterbi decoding, route reconstruction, Folium maps, evaluator UI | `src/pathsnap/decoding/`, `visualisation/`, `scripts/tune_hmm.py`, `generate_maps.py`, `frontend/` |

The technical report, the documentation and the review of all results were shared by the whole team. Each member answers
for their own area and can explain the others.

## Demonstration roles

| Role | Member |
|---|---|
| Laptop 1: API console and request logs | Sujal |
| Laptop 2: evaluator UI walkthrough | Shubashitha |
| Data, preprocessing and features | Shriya |
| Models, cross-validation and Bayesian optimisation | Sudheeksh |

## Repository history

* `main` / `develop`: the repository was created by Shriya S R. The PathSnap implementation commits were made from
  Sujal's machine (git identity `SUJAL`). Full log: `git log --format="%h %an %ad %s" --date=short`.
* `sudeeksh`: an earlier exploratory implementation by Sudeeksh Nayak, kept unchanged (see report §3.1).
* `phase1-baseline`: the Phase 1 review snapshot (EDA and five baseline models).

## AI assistance

The implementation was produced with substantial help from an AI coding assistant (Claude Code) under the team's
direction, as disclosed in the [README](README.md#ai-assistance-disclosure). Every number in the report is produced by
the repository's scripts from the supplied data.
