"""Build candidate/feature tables for the train and student_test partitions.

Usage:  python scripts/build_features.py [--jobs 6]
Outputs (gitignored, fully reproducible):
  data/processed/<partition>_candidates.pkl   one row per (fix, candidate arc)
  data/processed/<partition>_obs.pkl          one row per cleaned GPS fix
  reports/eda/candidate_coverage_<partition>.csv
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathsnap.config import processed_dir  # noqa: E402
from pathsnap.data.dataset import partition_dir, process_trip  # noqa: E402
from pathsnap.data.loader import list_trips, load_trip  # noqa: E402


def one(part: str, tid: str):
    t0 = time.perf_counter()
    trip = load_trip(partition_dir(part) / tid, with_route=True, partition=part)
    ctx = process_trip(trip, with_labels=True)
    c = ctx.cands
    has = np.zeros(len(ctx.track), bool)
    has[c["obs"].to_numpy()] = True
    hit = np.zeros(len(ctx.track), bool)
    hit[c.loc[c["label"] == 1, "obs"].to_numpy()] = True
    obs = pd.DataFrame({
        "trip_id": tid, "obs": np.arange(len(ctx.track)), "orig_index": ctx.track.orig_index,
        "t": ctx.track.t, "lon": ctx.track.lon, "lat": ctx.track.lat, "x": ctx.px, "y": ctx.py,
        "label": ctx.obs_labels, "route_pos": ctx.route_pos, "label_dist_m": ctx.label_dist,
        "has_candidates": has, "true_in_candidates": hit,
    })
    cov = {"partition": part, "trip_id": tid, **ctx.coverage, "n_rows": len(c),
           "seconds": time.perf_counter() - t0, **ctx.timings}
    return c, obs, cov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--partitions", nargs="+", default=["train", "student_test"])
    args = ap.parse_args()
    out = processed_dir()
    for part in args.partitions:
        tids = list_trips(partition_dir(part))
        res = Parallel(n_jobs=args.jobs, verbose=5)(delayed(one)(part, t) for t in tids)
        cands = pd.concat([r[0] for r in res], ignore_index=True)
        obs = pd.concat([r[1] for r in res], ignore_index=True)
        cov = pd.DataFrame([r[2] for r in res])
        cands.to_pickle(out / f"{part}_candidates.pkl")
        obs.to_pickle(out / f"{part}_obs.pkl")
        cov.to_csv(ROOT / "reports" / "eda" / f"candidate_coverage_{part}.csv", index=False)
        lab = obs["label"] >= 0
        print(f"[{part}] trips={len(tids)} fixes={len(obs)} labelled={lab.sum()} rows={len(cands)} "
              f"pos={int(cands['label'].sum())} miss_rate={1 - obs.loc[lab, 'true_in_candidates'].mean():.4f}")


if __name__ == "__main__":
    main()
