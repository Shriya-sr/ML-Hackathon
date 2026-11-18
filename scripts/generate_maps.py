"""Generate Folium maps from real data and real model outputs.

Usage:  python scripts/generate_maps.py
Outputs: maps/html/*.html  (+ maps/map_index.csv)

Maps:
  eda_train_<id>_raw.html            raw GPS + road network + reference (training partition, EDA)
  student_<id>_comparison.html       GPS, network, reference, classifier-only and HMM routes
  zoom_student_<id>.html             40-fix stretch where the HMM corrects the most classifier errors (small GPS markers)
Reference routes appear ONLY in these offline development maps, never in the API/UI.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathsnap.config import load_config  # noqa: E402
from pathsnap.data.dataset import partition_dir, process_trip  # noqa: E402
from pathsnap.data.labeling import align_to_route  # noqa: E402
from pathsnap.evaluation.route_metrics import collapse  # noqa: E402
from pathsnap.data.loader import load_trip  # noqa: E402
from pathsnap.inference.pipeline import MapMatcher, RouteEvaluator  # noqa: E402
from pathsnap.visualisation.folium_maps import save_map, trip_map  # noqa: E402

OUT = ROOT / "maps" / "html"
EDA_TRIPS = ["00000001", "00000054"]
STUDENT_TRIPS = ["00000052", "00000009", "00000077", "00000065"]
ZOOM_TRIPS = ["00000077", "00000009"]
ZOOM_FIXES, ZOOM_MIN_EXTENT_M = 40, 150.0


def zoom_window(clf_ok, hmm_ok, lab_ok, px, py, w=ZOOM_FIXES, min_extent=ZOOM_MIN_EXTENT_M) -> int:
    """Start of the w-fix stretch (all labelled, vehicle moving) where the HMM corrects the most classifier errors."""
    best, best_i = None, 0
    for i in range(len(clf_ok) - w):
        sl = slice(i, i + w)
        if lab_ok[sl].sum() < w or max(np.ptp(px[sl]), np.ptp(py[sl])) < min_extent:
            continue
        score = (int(hmm_ok[sl].sum()) - int(clf_ok[sl].sum()), int(hmm_ok[sl].sum()))
        if best is None or score > best:
            best, best_i = score, i
    return best_i


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
    lc = load_config("data")["labelling"]
    for tid in ZOOM_TRIPS:
        tdir = partition_dir("student_test") / tid
        pred = matcher.predict(tdir)
        ctx = pred.ctx
        ref = load_trip(tdir).route          # offline development map only
        lab, _, _ = align_to_route(ctx.px, ctx.py, ctx.net, ref, lc["distance_cap_m"], lc["max_label_distance_m"])
        lab_ok = lab >= 0
        clf_ok, hmm_ok = (pred.clf_arc == lab) & lab_ok, (pred.hmm_arc == lab) & lab_ok
        i0 = zoom_window(clf_ok, hmm_ok, lab_ok, ctx.px, ctx.py)
        sl = slice(i0, i0 + ZOOM_FIXES)
        c = ctx.cands
        near = np.unique(c.loc[(c.obs >= i0) & (c.obs < i0 + ZOOM_FIXES), "arc"].to_numpy())
        nc, nh = int(clf_ok[sl].sum()), int(hmm_ok[sl].sum())
        title = (f"Zoom, student-test trip {tid}, fixes {i0}-{i0 + ZOOM_FIXES - 1}: correct road classifier-only "
                 f"{nc}/{ZOOM_FIXES} -> HMM {nh}/{ZOOM_FIXES}")
        m = trip_map(ctx.track.lat[sl], ctx.track.lon[sl], ctx.net, title, near_arcs=near, reference=collapse(lab[sl]),
                     clf_route=collapse(pred.clf_arc[sl]), hmm_route=collapse(pred.hmm_arc[sl]), gps_radius=1.6, gps_opacity=0.55)
        p = save_map(m, OUT / f"zoom_student_{tid}.html")
        rows.append({"map": p.name, "partition": "student_test", "trip_id": tid,
                     "content": f"zoom fixes {i0}-{i0 + ZOOM_FIXES - 1}: reference + classifier-only + HMM",
                     "clf_top1": nc / ZOOM_FIXES, "hmm_top1": nh / ZOOM_FIXES})
        print("saved", p.name, nc, "->", nh, flush=True)
    pd.DataFrame(rows).to_csv(ROOT / "maps" / "map_index.csv", index=False)


if __name__ == "__main__":
    main()
