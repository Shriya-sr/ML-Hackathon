"""Generate Folium maps from real data and real model outputs.

Usage:  python scripts/generate_maps.py
Outputs: maps/html/*.html  (+ maps/map_index.csv)

Maps:
  eda_train_<id>_raw.html            raw GPS + road network + reference (training partition, EDA)
  student_<id>_comparison.html       GPS, network, reference, classifier-only and HMM routes
Reference routes appear ONLY in these offline development maps, never in the API/UI.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathsnap.data.dataset import partition_dir, process_trip  # noqa: E402
from pathsnap.data.loader import load_trip  # noqa: E402
from pathsnap.inference.pipeline import MapMatcher, RouteEvaluator  # noqa: E402
from pathsnap.visualisation.folium_maps import save_map, trip_map  # noqa: E402

OUT = ROOT / "maps" / "html"
EDA_TRIPS = ["00000001", "00000054"]
STUDENT_TRIPS = ["00000052", "00000009", "00000077", "00000065"]


def main():
    rows = []
    for tid in EDA_TRIPS:
        trip = load_trip(partition_dir("train") / tid, partition="train")
        ctx = process_trip(trip)
        near = np.unique(ctx.cands["arc"].to_numpy())
        dt = np.diff(ctx.track.t)
        gaps = [(int(i + 1), float(dt[i])) for i in np.where(dt > 30)[0]]
        m = trip_map(ctx.track.lat, ctx.track.lon, ctx.net, f"Training trip {tid}: raw GPS, road network, reference route",
                     near_arcs=near, reference=trip.route, gaps=gaps)
        p = save_map(m, OUT / f"eda_train_{tid}_raw.html")
        rows.append({"map": p.name, "partition": "train", "trip_id": tid, "content": "raw GPS + network + reference"})
        print("saved", p.name, flush=True)

    matcher = MapMatcher()
    ev = RouteEvaluator()
    for tid in STUDENT_TRIPS:
        tdir = partition_dir("student_test") / tid
        pred = matcher.predict(tdir)
        met = ev.evaluate(tdir, pred)
        ref = load_trip(tdir).route          # offline development map only
        ctx = pred.ctx
        near = np.unique(ctx.cands["arc"].to_numpy())
        dt = np.diff(ctx.track.t)
        gaps = [(int(i + 1), float(dt[i])) for i in np.where(dt > 30)[0]]
        brk = [(int(k), v) for k, v in pred.breaks.items() if v != "no_candidates"]
        title = (f"Student-test trip {tid}: Top-1 classifier {met['classifier']['top1_accuracy']:.3f} "
                 f"-> HMM {met['hmm']['top1_accuracy']:.3f}")
        m = trip_map(ctx.track.lat, ctx.track.lon, ctx.net, title, near_arcs=near, reference=ref,
                     clf_route=pred.clf_route, hmm_route=pred.route, gaps=gaps, breaks=brk)
        p = save_map(m, OUT / f"student_{tid}_comparison.html")
        rows.append({"map": p.name, "partition": "student_test", "trip_id": tid,
                     "content": "GPS + network + reference + classifier-only + HMM",
                     "clf_top1": met["classifier"]["top1_accuracy"], "hmm_top1": met["hmm"]["top1_accuracy"]})
        print("saved", p.name, flush=True)
    pd.DataFrame(rows).to_csv(ROOT / "maps" / "map_index.csv", index=False)


if __name__ == "__main__":
    main()
