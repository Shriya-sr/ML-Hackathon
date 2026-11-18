"""Single source of truth for the model input schema.

Training, evaluation and the API all call ``to_matrix`` so the column order can
never drift between training and inference."""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_VERSION = "v1"

FEATURE_GROUPS: dict[str, list[str]] = {
    "geometric": ["dist_m", "dist_rank", "dist_minus_min", "dist_ratio", "n_candidates",
                  "proj_frac", "at_endpoint", "endpoint_dist_m", "arc_length_m"],
    "heading": ["cos_heading", "heading_diff_deg", "heading_valid", "cos_long_bearing"],
    "context": ["win_mean_dist_m", "win_max_dist_m"],
    "kinematic": ["speed_mps", "dt_prev", "dt_next"],
    "network": ["from_in_degree", "to_out_degree", "is_two_way"],
}

FEATURE_COLUMNS: list[str] = [c for cols in FEATURE_GROUPS.values() for c in cols]

# columns that identify a row but are NOT model inputs
KEY_COLUMNS = ["trip_id", "obs", "arc"]
LABEL_COLUMN = "label"          # 1 if arc == reference arc of that observation
TRUE_ARC_COLUMN = "true_arc"    # reference arc of the observation (-1 unknown)


class SchemaError(ValueError):
    pass


def to_matrix(df: pd.DataFrame, columns: list[str] | None = None) -> np.ndarray:
    cols = columns or FEATURE_COLUMNS
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SchemaError(f"missing feature columns: {missing}")
    X = df[cols].to_numpy(dtype=np.float64)
    if not np.isfinite(X).all():
        bad = [cols[j] for j in np.where(~np.isfinite(X).all(0))[0]]
        raise SchemaError(f"non-finite values in features: {bad}")
    return X
