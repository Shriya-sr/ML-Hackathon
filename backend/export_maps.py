from __future__ import annotations

import argparse
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
from .data import discover_trips, partition_name
from .features import candidate_features
from .decoder import viterbi
from .maps import save_trip_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--artifact", default="artifacts/pathsnap.joblib")
    parser.add_argument("--output-dir", default="results/maps")
    args = parser.parse_args()
    model_pack = joblib.load(args.artifact); model = model_pack["model"]
    trips = [t for t in discover_trips(args.data_dir) if partition_name(t.partition) == "student_test"]
    if not trips:
        raise ValueError("No student_test trips are available for development maps")
    trip = min(trips, key=lambda t: len(t.gps))
    ids, feats, labels, misses = candidate_features(trip, model_pack.get("candidate_k", 8))
    probs = [model.predict_proba(x)[:, 1] for x in feats]
    pred = viterbi(ids, probs, trip.arcs)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    save_trip_map(trip, pred, out / f"trip_{trip.trip_id}.html", include_reference=True)

    fig, ax = plt.subplots(figsize=(9, 7), dpi=170)
    gps = trip.gps[:, :2]
    ax.plot(gps[:, 0], gps[:, 1], color="#1769aa", linewidth=1.5, label="GPS observations", zorder=3)
    ref = trip.nodes[trip.arcs[labels]]
    pred_edges = trip.nodes[trip.arcs[pred]]
    ax.add_collection(LineCollection(ref, colors="#2e8b57", linewidths=1.2, label="Reference route", zorder=2))
    ax.add_collection(LineCollection(pred_edges, colors="#e53935", linewidths=1.0, label="Predicted route", zorder=1))
    ax.scatter([gps[0, 0], gps[-1, 0]], [gps[0, 1], gps[-1, 1]], c=["#222", "#222"], s=16, zorder=4)
    ax.set_title(f"Trip {trip.trip_id}: GPS and route overlay")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude"); ax.grid(alpha=.2)
    ax.legend(loc="best"); ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout(); fig.savefig(out / f"trip_{trip.trip_id}.png", bbox_inches="tight"); plt.close(fig)
    print(f"Wrote interactive map and report figure for trip {trip.trip_id}; candidate miss rate={misses.mean():.3f}")


if __name__ == "__main__":
    main()
