# PathSnap

PathSnap maps noisy GPS observations to a directed road network and reconstructs a coherent road route. The code reads the organizer dataset provided in the sibling `Student folder\Student folder` directory; raw data is excluded from the public repository.

## Project layout

```text
backend/       data loading, candidate generation, training, decoding, API
frontend/      small evaluator UI served by FastAPI
data/          place the organizer-provided partitions here (not committed)
artifacts/     trained model and preprocessing artifacts (generated locally)
results/       experiment records and evaluation summaries (generated locally)
```

## Setup

Use Python 3.10 or newer.

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The loader expects the organizer's partition folders (here `train` and `student_test`) with one trip folder containing `.track`, `.nodes`, `.arcs`, and `.route` files. Override the default sibling data location with `PATHSNAP_DATA_DIR` when needed. Keep evaluator-testing data separate; do not use its labels for training or development selection.

## Explore and train

```powershell
python -m backend.explore --data-dir "..\Student folder\Student folder" --output results/dataset_inventory.csv
python -m backend.train --data-dir "..\Student folder\Student folder" --output-dir artifacts --results-dir results
```

Training aligns GPS fixes monotonically to each supplied reference route, uses up to 100 evenly sampled observations per training trip to control runtime on Windows, compares five classifiers with three-fold trip-grouped cross-validation, and applies a Gaussian-process expected-improvement search to Random Forest settings. Student-test route features are computed once and reused for the model comparison. It evaluates classifier-only and classifier-plus-Viterbi outputs on the student-test partition. Generated CSV/JSON files contain the actual run results; no metric should be reported until the run finishes.

Generate the representative interactive and report maps, then build the consolidated report after training completes:

```powershell
python -m backend.export_maps --data-dir "..\Student folder\Student folder"
python scripts/build_report.py
```

The report is written to `output/pdf/PathSnap_Technical_Report.pdf`. Rebuild it after changing the pipeline or regenerating experiment outputs.

## Run the demo

After training, start the API on laptop 1:

```powershell
uvicorn backend.api:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000/` on laptop 2 (replace `localhost` with laptop 1's LAN address). The current local data only includes student-test trips, which are used for development. Evaluator trips need to be supplied by the organizers before the final evaluation; do not present the student-test list as the reserved evaluator set. The UI shows the prediction and server-side metrics without returning reference labels.

## Current limitations

- The repository does not include raw GPS data. Keep the organizer dataset in the provided local folder and do not commit it.
- The supplied package has no evaluator-test partition. End-to-end evaluator flow and metrics must be checked once that reserved partition is provided.
- No evaluator-testing data should be used to choose models, features, or decoder settings.

## Team work split

- Sudeeksh: integrate the pipeline, training/evaluation protocol, and release branch.
- Shubashitha Gowtham: dataset audit, preprocessing, and feature/candidate analysis.
- Shriya: API and evaluator UI integration.
- Sujal Yadav: HMM/Viterbi decoder, continuity analysis, and map outputs.

Each member should review, run, and commit their own contribution using their own Git identity. Keep experiment evidence reproducible and explainable by the team.
