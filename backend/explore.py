from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from .data import discover_trips, partition_name


def inventory(root: str | Path, output: str | Path) -> pd.DataFrame:
    trips = discover_trips(root)
    rows = []
    for t in trips:
        gaps = np.diff(t.gps[:, 2])
        adjacency = coo_matrix((np.ones(len(t.arcs) * 2, dtype=np.uint8),
            (np.r_[t.arcs[:, 0], t.arcs[:, 1]], np.r_[t.arcs[:, 1], t.arcs[:, 0]])),
            shape=(len(t.nodes), len(t.nodes))).tocsr()
        component_count, labels = connected_components(adjacency, directed=False, return_labels=True)
        component_sizes = np.bincount(labels)
        degree = np.asarray(adjacency.sum(axis=0)).reshape(-1)
        rows.append({"partition": partition_name(t.partition), "trip_id": t.trip_id,
            "gps_observations": len(t.gps), "nodes": len(t.nodes), "arcs": len(t.arcs),
            "route_arcs": 0 if t.route is None else len(t.route),
            "raw_gps_rows": t.raw_gps_rows, "invalid_gps_rows_removed": t.invalid_gps_rows,
            "timestamp_non_increasing": int(np.sum(gaps <= 0)),
            "median_gap_s": float(np.median(gaps)) if len(gaps) else None,
            "max_gap_s": float(np.max(gaps)) if len(gaps) else None,
            "isolated_nodes": int(np.sum(degree == 0)),
            "weak_components": int(component_count),
            "largest_component_fraction": float(component_sizes.max() / len(t.nodes)),
            "longitude_min": float(t.gps[:, 0].min()), "longitude_max": float(t.gps[:, 0].max()),
            "latitude_min": float(t.gps[:, 1].min()), "latitude_max": float(t.gps[:, 1].max())})
    frame = pd.DataFrame(rows)
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    counts = frame.groupby("partition").agg(trips=("trip_id", "count"), observations=("gps_observations", "sum"), nodes=("nodes", "sum"), arcs=("arcs", "sum")).to_dict(orient="index")
    (output.parent / "dataset_summary.json").write_text(json.dumps(counts, indent=2))
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", default="results/dataset_inventory.csv")
    args = parser.parse_args()
    frame = inventory(args.data_dir, args.output)
    print(frame.groupby("partition")["trip_id"].count().to_string())
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
