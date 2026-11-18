"""Generate the EDA Folium maps from the real training data.

Usage:  python scripts/generate_maps.py
Outputs: maps/html/eda_train_<id>_raw.html  (raw GPS + road network near the trajectory + reference route)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathsnap.data.dataset import partition_dir, process_trip  # noqa: E402
from pathsnap.data.loader import load_trip  # noqa: E402
from pathsnap.visualisation.folium_maps import save_map, trip_map  # noqa: E402

OUT = ROOT / "maps" / "html"
EDA_TRIPS = ["00000001", "00000054"]


def main():
    for tid in EDA_TRIPS:
        trip = load_trip(partition_dir("train") / tid, partition="train")
        ctx = process_trip(trip)
        near = np.unique(ctx.cands["arc"].to_numpy())
        dt = np.diff(ctx.track.t)
        gaps = [(int(i + 1), float(dt[i])) for i in np.where(dt > 30)[0]]
        m = trip_map(ctx.track.lat, ctx.track.lon, ctx.net, f"Training trip {tid}: raw GPS, road network, reference route",
                     near_arcs=near, reference=trip.route, gaps=gaps)
        print("saved", save_map(m, OUT / f"eda_train_{tid}_raw.html").name)


if __name__ == "__main__":
    main()
