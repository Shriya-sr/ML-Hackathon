# Data

The raw dataset is distributed by the course and is **not committed** to Git.

```
data/raw/
├── train/            80 trip folders   (model fitting, CV, optimisation)
├── student_test/     10 trip folders   (development-stage evaluation)
└── evaluator_test/   EMPTY until the organisers supply the 10 reserved trips
```

Each trip folder `<id>/` contains `<id>.track`, `<id>.nodes`, `<id>.arcs`, `<id>.route`
(formats in `docs/data_dictionary.md`).

Setup on a fresh clone (Windows PowerShell), assuming the course archive was extracted to `D:\ML hackathon`:

```powershell
Copy-Item -Recurse "D:\ML hackathon\train"        data\raw\train
Copy-Item -Recurse "D:\ML hackathon\student_test" data\raw\student_test
New-Item -ItemType Directory -Force data\raw\evaluator_test
```

Or point the code at any folder with `PATHSNAP_DATA_DIR`:

```powershell
$env:PATHSNAP_DATA_DIR = "D:\ML hackathon"
```

## Evaluator trips

Put the reserved trips in `data/raw/evaluator_test/<id>/` (same four-file layout) on **Laptop 1 only** and
restart the API. The API switches from `DEVELOPMENT` to `EVALUATION` mode automatically. Their `.route` files
are read only by the server-side scorer (`RouteEvaluator`) and are never sent to the UI. They are never used
by any training, tuning or selection script: those scripts read only `train/` and `student_test/`.

`data/processed/` holds reproducible intermediate tables written by `scripts/build_features.py`.
